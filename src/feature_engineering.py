import argparse
import datetime
import errno
import logging
import os.path
import sys
from collections import namedtuple
from typing import Optional, List

from pyspark.ml import PipelineModel, Transformer, Pipeline
from pyspark.ml.feature import SQLTransformer, StringIndexer, OneHotEncoder
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


class LoggingAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        msg_string = f"[{self.extra['id']}] {timestamp}: {msg}"
        return msg_string, kwargs

    @staticmethod
    def create(id: str):
        adapter = LoggingAdapter(logging.getLogger(), {"id": id})
        adapter.logger.setLevel(logging.INFO)
        if not adapter.logger.hasHandlers():
            adapter.logger.addHandler(logging.StreamHandler())
        return adapter


class EuclideanDistanceTransformer(
    Transformer, DefaultParamsWritable, DefaultParamsReadable, HasOutputCol
):
    def _transform(self, df: DataFrame) -> DataFrame:
        # calculate 2D Euclidean distance: sqrt(x^2 + y^2)
        transformation = F.sqrt(
            F.pow(F.col("x_shipping_distance"), 2)
            + F.pow(F.col("y_shipping_distance"), 2)
        )
        return df.withColumn(self.getOutputCol(), transformation)

    def getOutputCol(self) -> str:
        return "euclidean_distance"

    def get_param_value(self, param):
        return self.extractParamMap().get(param)


CategoriesReducerWithOutputColumn = namedtuple(
    "CategoriesReducerWithOutputColumns", ["reducer", "output_column"]
)


def feature_engineering(
    s3_dataframe_dir: str,
    s3_pipeline_model_dir: Optional[str],
    s3_output_dir: str,
    overwrite_output: bool,
    spark: SparkSession,
    logger: LoggingAdapter,
):
    logger.info(f"Reading dataframe from {s3_dataframe_dir}")
    df = spark.read.parquet(s3_dataframe_dir)
    if s3_pipeline_model_dir is not None:
        transform_using_pretrained_pipeline(
            df, s3_pipeline_model_dir, s3_output_dir, overwrite_output, logger
        )
    else:
        train_pipeline_and_transform(df, s3_output_dir, overwrite_output, logger)


def transform_using_pretrained_pipeline(
    df: DataFrame,
    s3_pipeline_model_dir: str,
    s3_output_dir: str,
    overwrite_output: bool,
    logger: LoggingAdapter,
):
    logger.info(
        f"Transforming dataframe using pretrained pipeline model from {s3_pipeline_model_dir}"
    )
    pipeline_model = PipelineModel.load(s3_pipeline_model_dir)
    df_transformed = pipeline_model.transform(df)
    save_dataframe(df_transformed, s3_output_dir, overwrite_output, logger)


def train_pipeline_and_transform(
    df: DataFrame, s3_output_dir: str, overwrite_output: bool, logger: LoggingAdapter
):
    pipeline: Pipeline = create_pipeline(logger)
    logger.info("Fitting pipeline to provided dataframe")
    pipeline_model = pipeline.fit(df)
    logger.info("Fitting successful, transforming using fitted pipeline model")
    df_transformed = pipeline_model.transform(df)
    save_artefacts(
        df_transformed, pipeline_model, s3_output_dir, overwrite_output, logger
    )


def save_artefacts(
    df: DataFrame,
    pipeline_model: PipelineModel,
    s3_output_dir: str,
    overwrite_output: bool,
    logger: LoggingAdapter,
):
    logger.info("Saving pretrained artefacts (dataframe, pipeline model)")
    save_pipeline_model(pipeline_model, s3_output_dir, overwrite_output, logger)
    save_dataframe(df, s3_output_dir, overwrite_output, logger)


def save_pipeline_model(
    pipeline_model: PipelineModel,
    s3_output_dir: str,
    overwrite_output: bool,
    logger: LoggingAdapter,
):
    s3_pipeline_model_path = os.path.join(s3_output_dir, "pipeline_model")
    logger.info(f"Saving pipeline model to {s3_pipeline_model_path}")
    writer = (
        pipeline_model.write().overwrite()
        if overwrite_output
        else pipeline_model.write()
    )
    writer.save(s3_pipeline_model_path)


def save_dataframe(
    df: DataFrame, s3_output_dir: str, overwrite_output: bool, logger: LoggingAdapter
):
    s3_dataframe_output_path = os.path.join(s3_output_dir, "dataframe")
    number_of_rows = df.count()
    chunk_approx_size = 1000
    number_of_chunks = max(1, int(number_of_rows / chunk_approx_size))
    # split dataframe to chunks for concurrent write
    logger.info(
        f"Saving dataframe to {s3_dataframe_output_path} in {number_of_chunks} chunks"
    )
    df = df.repartition(number_of_chunks)
    writer = df.write.mode("overwrite") if overwrite_output else df.write
    writer.parquet(s3_dataframe_output_path)


def create_pipeline(logger: LoggingAdapter):
    logger.info("Assembling dataframe transformations pipeline")

    shipping_days_columns = ["expected_shipping_days", "actual_shipping_days"]

    category_reducers_wrapped: List[
        CategoriesReducerWithOutputColumn
    ] = create_category_reducers(logger)

    string_indexer_input_columns = [
        *[red.output_column for red in category_reducers_wrapped],
        "shipping_priority",
    ]
    string_indexer = create_string_indexer(logger, *string_indexer_input_columns)

    one_hot_encoder = create_one_hot_encoder(logger, *string_indexer.getOutputCols())

    string_indexer_in_bulk_order = create_string_indexer(logger, "in_bulk_order")

    euclidean_distance_transformer = EuclideanDistanceTransformer()

    columns_keeper = create_columns_keeper(
        *one_hot_encoder.getOutputCols(),
        *string_indexer_in_bulk_order.getOutputCols(),
        euclidean_distance_transformer.getOutputCol(),
        *shipping_days_columns,
        "order_id"
    )

    pipeline_stages = [
        *[red.reducer for red in category_reducers_wrapped],
        string_indexer,
        one_hot_encoder,
        string_indexer_in_bulk_order,
        euclidean_distance_transformer,
        columns_keeper,
    ]
    logger.info(
        f"Creating pipeline with stages: {to_decimal_separated_string(pipeline_stages)}"
    )
    return Pipeline(stages=pipeline_stages)


def create_category_reducers(
    logger: LoggingAdapter,
) -> List[CategoriesReducerWithOutputColumn]:
    result = []
    categories_to_keep_by_column = {
        "carrier": ["BigBird"],
        "shipping_origin": ["Chicago", "Seattle", "San Francisco", "Salt Late City"],
    }
    for column, categories_to_keep in categories_to_keep_by_column.items():
        reducer_wrapped: CategoriesReducerWithOutputColumn = create_categories_reducer(
            column, categories_to_keep, logger
        )
        result.append(reducer_wrapped)
    return result


def create_categories_reducer(
    column: str, keep_categories: List[str], logger: LoggingAdapter
) -> CategoriesReducerWithOutputColumn:
    logger.info(
        f"Reducing categories for {column}, keeping {to_decimal_separated_string(keep_categories)}, "
        f"changing remaining to 'OTHER'"
    )
    keep_categories_str = to_decimal_separated_string(
        [f"'{cat}'" for cat in keep_categories]
    )
    output_column = f"{column}_reduced"
    statement = f"SELECT CASE WHEN {column} IN ({keep_categories_str}) THEN {column} ELSE 'OTHER' END as {output_column}, * FROM __THIS__"
    return CategoriesReducerWithOutputColumn(
        SQLTransformer(statement=statement), output_column
    )


def create_string_indexer(logger: LoggingAdapter, *input_columns):
    logger.info(
        f"Creating string indexer for: {to_decimal_separated_string(input_columns)}"
    )
    string_indexer_output_columns = [f"{c}_indexed" for c in input_columns]
    return StringIndexer(
        inputCols=input_columns, outputCols=string_indexer_output_columns
    )


def create_one_hot_encoder(logger: LoggingAdapter, *input_columns):
    logger.info(
        f"Creating one-hot-encoder for: {to_decimal_separated_string(input_columns)}"
    )
    ohe_output_columns = [f"{c}_ohe" for c in input_columns]
    return OneHotEncoder(inputCols=input_columns, outputCols=ohe_output_columns)


def create_columns_keeper(*column_names):
    statement = f"SELECT {','.join(column_names)} from __THIS__"
    return SQLTransformer(statement=statement)


def to_decimal_separated_string(list_):
    return ", ".join([str(it) for it in list_])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-dataframe-dir", type=str, required=True)
    parser.add_argument("--s3-pipeline-model-dir", type=str, required=False)
    parser.add_argument("--s3-output-dir", type=str, required=True)
    parser.add_argument("--overwrite-output", type=bool, required=False, default=True)
    args, _ = parser.parse_known_args()
    return (
        args.s3_dataframe_dir,
        args.s3_pipeline_model_dir,
        args.s3_output_dir,
        args.overwrite_output,
    )


def main():
    (
        s3_dataframe_dir,
        s3_pipeline_model_dir,
        s3_output_dir,
        overwrite_output,
    ) = parse_arguments()
    app_name = "ml-on-emr-feature-engineering"
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )
    logger = LoggingAdapter.create(app_name)

    try:
        feature_engineering(
            s3_dataframe_dir,
            s3_pipeline_model_dir,
            s3_output_dir,
            overwrite_output,
            spark,
            logger,
        )
    except Exception as exc:
        logger.error(f"Application {app_name} exited with error:")
        logger.error(exc, exc_info=True)
        sys.exit(errno.EPERM)

    logger.info(f"App {app_name} completed successfully.")


if __name__ == "__main__":
    main()
