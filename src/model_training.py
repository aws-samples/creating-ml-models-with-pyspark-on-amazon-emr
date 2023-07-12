import argparse
import ast
import datetime
import errno
import json
import logging
import os.path
import sys
import time
from typing import Optional

import boto3 as boto3
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, FMRegressor
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType


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


supported_models_by_name = {"gbt": GBTRegressor, "factorization_machine": FMRegressor}

target_column = "actual_shipping_days"


def train_model(
    s3_training_dataset_dir: str,
    s3_validation_dataset_dir: str,
    s3_output_dir: str,
    overwrite_output: bool,
    model_type: str,
    model_params: dict,
    spark: SparkSession,
    logger: LoggingAdapter,
):
    training_df = read_dataframe(spark, s3_training_dataset_dir, logger)
    validation_df = read_dataframe(spark, s3_validation_dataset_dir, logger)

    validate_model_type_is_supported(model_type)

    vector_assembler = VectorAssembler(
        inputCols=[
            c for c in training_df.columns if c != target_column and not is_id_column(c)
        ],
        outputCol="training_cols",
    )

    logger.info(
        f"Using {', '.join(vector_assembler.getInputCols())} columns for {model_type} model training"
    )

    model_class = supported_models_by_name[model_type]
    model = model_class(
        **model_params,
        labelCol=target_column,
        featuresCol=vector_assembler.getOutputCol(),
    )

    pipeline_stages = [vector_assembler, model]
    training_pipeline = Pipeline(stages=pipeline_stages)

    logger.info("Model training start")
    training_start = time.time()
    pipeline_model = training_pipeline.fit(training_df)
    training_stop = time.time()
    logger.info(
        f"Model training finished successfully after {(training_stop - training_start):.1f} seconds"
    )

    training_predictions = postprocess_predictions(
        pipeline_model.transform(training_df)
    )
    training_metrics = calculate_and_log_metrics(
        training_predictions, "training", logger
    )
    save_artefacts(
        training_predictions,
        training_metrics,
        s3_output_dir,
        "training",
        overwrite_output,
        logger,
    )

    validation_predictions = postprocess_predictions(
        pipeline_model.transform(validation_df)
    )
    validation_metrics = calculate_and_log_metrics(
        validation_predictions, "validation", logger
    )
    save_artefacts(
        validation_predictions,
        validation_metrics,
        s3_output_dir,
        "validation",
        overwrite_output,
        logger,
    )


def read_dataframe(spark: SparkSession, path: str, logger: LoggingAdapter):
    logger.info(f"Reading dataframe from {path}")
    return spark.read.parquet(path)


def is_id_column(column_name: str):
    return column_name.endswith("_id")


def validate_model_type_is_supported(model_type: str):
    if model_type not in supported_models_by_name:
        available_types_substr = ", ".join(
            f"'{name}' ({clazz})" for name, clazz in supported_models_by_name.items()
        )
        raise ValueError(f"Model type '{model_type}' not supported, available types: {available_types_substr}")


def postprocess_predictions(prediction_df: DataFrame):
    prediction_df = prediction_df.withColumnRenamed("prediction", "raw_prediction")
    prediction_df = prediction_df.withColumn(
        "rounded_prediction", F.round(F.col("raw_prediction"))
    )
    id_columns = [c for c in prediction_df.columns if is_id_column(c)]
    select_columns = [
        target_column,
        "rounded_prediction",
        "raw_prediction",
        *id_columns,
    ]
    return prediction_df.select(select_columns)


def calculate_and_log_metrics(
    predictions_df: DataFrame, predictions_identifier: str, logger: LoggingAdapter
):
    def get_metrics(prediction_column: str):
        return RegressionMetrics(
            predictions_df.select(
                F.col(target_column).cast(FloatType()),
                F.col(prediction_column).cast(FloatType()),
            ).rdd
        )

    raw_prediction_metrics = get_metrics("raw_prediction")
    rounded_prediction_metrics = get_metrics("rounded_prediction")

    logger.info(f"Metrics for {predictions_identifier} raw predictions:")
    logger.info(f"Mean absolute error: {raw_prediction_metrics.meanAbsoluteError}")
    logger.info(f"Mean squared error: {raw_prediction_metrics.meanSquaredError}")

    logger.info(f"Metrics for {predictions_identifier} rounded predictions:")
    logger.info(f"Mean absolute error: {rounded_prediction_metrics.meanAbsoluteError}")
    logger.info(f"Mean squared error: {rounded_prediction_metrics.meanSquaredError}")
    return {
        "raw_predictions_mean_absolute_error": raw_prediction_metrics.meanAbsoluteError,
        "raw_predictions_mean_squared_error": raw_prediction_metrics.meanSquaredError,
        "rounded_predictions_mean_absolute_error": rounded_prediction_metrics.meanAbsoluteError,
        "rounded_predictions_mean_squared_error": rounded_prediction_metrics.meanSquaredError,
    }


def save_artefacts(
    predictions_df: DataFrame,
    metrics: dict,
    s3_output_dir: str,
    artefacts_identifier: str,
    overwrite_output: bool,
    logger: LoggingAdapter,
):
    artefacts_dir = os.path.join(s3_output_dir, artefacts_identifier)

    predictions_path = os.path.join(artefacts_dir, "predictions.csv")
    logger.info(f"Saving {artefacts_identifier} predictions to {predictions_path}")
    writer = predictions_df.write.options(header="True", delimiter=",")
    writer = writer.mode("overwrite") if overwrite_output else writer
    writer.csv(predictions_path)

    metrics_path = os.path.join(artefacts_dir, "metrics.json")
    logger.info(f"Saving {artefacts_identifier} metrics to {metrics_path}")
    bucket, key = extract_s3_bucket_and_key(metrics_path)
    s3 = boto3.client("s3")
    s3.put_object(Body=json.dumps(metrics), Bucket=bucket, Key=key)


def extract_s3_bucket_and_key(path):
    path_parts = path.replace("s3://", "").split("/")
    return path_parts[0], "/".join(path_parts[1:])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-training-dataset-dir", type=str, required=True)
    parser.add_argument("--s3-validation-dataset-dir", type=str, required=True)
    parser.add_argument("--s3-output-dir", type=str, required=True)
    parser.add_argument(
        "--add-timestamp-to-output", required=False, default=False, action="store_true"
    )
    parser.add_argument("--overwrite-output", type=bool, required=False, default=True)
    parser.add_argument("--model-type", type=str, required=False, default="gbt")
    parser.add_argument("--model-params", type=str, required=False, default=None)
    args, _ = parser.parse_known_args()
    return (
        args.s3_training_dataset_dir,
        args.s3_validation_dataset_dir,
        args.s3_output_dir,
        args.add_timestamp_to_output,
        args.overwrite_output,
        args.model_type,
        args.model_params,
    )


def main():
    (
        s3_training_dataset_dir,
        s3_validation_dataset_dir,
        s3_output_dir,
        add_timestamp_to_output,
        overwrite_output,
        model_type,
        model_params,
    ) = parse_arguments()
    model_params, s3_output_dir = preprocess_arguments(model_params, add_timestamp_to_output, s3_output_dir)

    app_name = "ml-on-emr-model-training"
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )
    logger = LoggingAdapter.create(app_name)

    try:
        train_model(
            s3_training_dataset_dir,
            s3_validation_dataset_dir,
            s3_output_dir,
            overwrite_output,
            model_type,
            model_params,
            spark,
            logger,
        )
    except Exception as exc:
        logger.error(f"Application {app_name} exited with error:")
        logger.error(exc, exc_info=True)
        sys.exit(errno.EPERM)

    logger.info(f"App {app_name} completed successfully.")


def preprocess_arguments(model_params, add_timestamp_to_output, s3_output_dir):
    model_params = parse_model_parameters(model_params)
    if add_timestamp_to_output:
        current_time_utc = datetime.datetime.utcnow().strftime("%Y_%m_%d_UTC_%H_%M_%S")
        s3_output_dir = os.path.join(s3_output_dir, current_time_utc)
    return model_params, s3_output_dir


def parse_model_parameters(model_parameters: Optional[str]):
    if model_parameters is None:
        return {}
    return ast.literal_eval(model_parameters)


if __name__ == "__main__":
    main()
