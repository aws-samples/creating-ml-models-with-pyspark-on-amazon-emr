import argparse
import bisect
import datetime
import errno
import logging
import os.path
import sys

import inflection as inflection
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


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


def preprocess_data(
    s3_shipping_logs_dir: str,
    s3_product_description_dir: str,
    s3_output_dir: str,
    overwrite_output: bool,
    training_fraction: float,
    spark: SparkSession,
    logger: LoggingAdapter,
):
    shipping_logs_df = read_csv_to_dataframe(s3_shipping_logs_dir, spark, logger)
    product_description_df = read_csv_to_dataframe(
        s3_product_description_dir, spark, logger
    )

    df = preprocess_dataframe(shipping_logs_df, product_description_df, logger)
    training_df, validation_df = train_val_split(df, training_fraction, logger)

    save_dataframes(
        s3_output_dir,
        logger,
        overwrite_output=overwrite_output,
        training=training_df,
        validation=validation_df,
    )


def preprocess_dataframe(shipping_logs_df, product_description_df, logger):
    join_column = "ProductId"
    logger.info(
        f"Joining shipping logs and product description dataframes on {join_column}"
    )
    df = shipping_logs_df.join(
        product_description_df, on="ProductId", how="left"
    ).cache()
    logger.info("Renaming column names to match snake_case")
    df = rename_columns_to_camel_case(df)
    logger.info(f"Dataframe columns after renaming: {', '.join(df.columns)}")

    logger.info("Formatting order_date to be of DateType type")
    reformat_date_udf = F.udf(reformat_date, StringType())
    df = df.withColumn("order_date", F.to_date(reformat_date_udf(F.col("order_date"))))
    return df


def train_val_split(df, training_fraction, logger):
    def get_cumulative_data_fraction_and_date():
        logger.info("Calculating cumulative data fractions from start to each date")
        window = Window.orderBy(F.col("order_date").asc())
        cumulative_fraction_by_date = (
            df.groupby("order_date")
            .count()
            .select("order_date", F.sum((F.col("count") / df.count())).over(window))
            .collect()
        )
        return list(zip(*cumulative_fraction_by_date))

    def determine_cutoff_date():
        logger.info("Determining training dataset cutoff date")
        dates, cumulative_fraction = get_cumulative_data_fraction_and_date()
        cutoff_idx = bisect.bisect_left(cumulative_fraction, training_fraction)
        if cutoff_idx >= len(dates) - 1:
            cutoff_idx = (
                len(dates) - 2
            )  # make sure validation dataset covers at least one date
        return dates[cutoff_idx]

    validate_at_least_two_distinct_dates_present(df)
    logger.info("Performing train/validation split")
    cutoff_date = determine_cutoff_date()
    training_df = df.filter(F.col("order_date") <= cutoff_date)
    validation_df = df.filter(F.col("order_date") > cutoff_date)
    training_dataset_data_fraction = training_df.count() / df.count()
    logger.info(
        f"Cutoff date calcualted to be {cutoff_date}. Training dataset data fraction: "
        f"{training_dataset_data_fraction:.4f}, validation dataset data fraction: "
        f"{1 - training_dataset_data_fraction:.4f}"
    )
    return training_df.cache(), validation_df.cache()


def save_dataframes(s3_output_dir, logger, overwrite_output, **dfs_by_name):
    for name, df in dfs_by_name.items():
        number_of_rows = df.count()
        chunk_approx_size = 1000
        number_of_chunks = max(1, int(number_of_rows / chunk_approx_size))
        s3_df_output_path = os.path.join(s3_output_dir, name)
        # split dataframe to chunks for concurrent write
        logger.info(
            f"Saving dataframe to {s3_df_output_path} in {number_of_chunks} chunks"
        )
        df = df.repartition(number_of_chunks)
        writer = df.write.mode("overwrite") if overwrite_output else df.write
        writer.parquet(s3_df_output_path)


def rename_columns_to_camel_case(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, inflection.underscore(c))
    return df


def validate_at_least_two_distinct_dates_present(df):
    distinct_dates = df.select("order_date").distinct().count()
    if distinct_dates < 2:
        raise ValueError(
            f"There must be at least two distinct dates for a valid train/validation split, found {distinct_dates}"
        )


def read_csv_to_dataframe(path: str, spark: SparkSession, logger: LoggingAdapter):
    logger.info(f"Reading dataframe from {path}")
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    logger.info(
        f"Successfully read dataframe with {df.count()} rows and {len(df.columns)} columns ({', '.join(df.columns)})"
    )
    return df


def reformat_date(date):
    if date is None:
        return date
    month, day, year = date.split("/")
    month = f"0{month}" if len(month) == 1 else month
    day = f"0{day}" if len(day) == 1 else day
    year = f"20{year}"
    return f"{year}-{month}-{day}"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-shipping-logs-dir", type=str, required=True)
    parser.add_argument("--s3-product-description-dir", type=str, required=True)
    parser.add_argument("--s3-output-dir", type=str, required=True)
    parser.add_argument("--overwrite-output", type=bool, required=False, default=True)
    parser.add_argument("--training-fraction", type=float, required=False, default=0.9)
    args, _ = parser.parse_known_args()
    return (
        args.s3_shipping_logs_dir,
        args.s3_product_description_dir,
        args.s3_output_dir,
        args.overwrite_output,
        args.training_fraction,
    )


def main():
    (
        s3_shipping_logs_dir,
        s3_product_description_dir,
        s3_output_dir,
        overwrite_output,
        training_fraction,
    ) = parse_arguments()
    validate_training_fraction_in_correct_range(training_fraction)

    app_name = "ml-on-emr-data-preprocessing"
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )
    logger = LoggingAdapter.create(app_name)

    try:
        preprocess_data(
            s3_shipping_logs_dir,
            s3_product_description_dir,
            s3_output_dir,
            overwrite_output,
            training_fraction,
            spark,
            logger,
        )
    except Exception as exc:
        logger.error(f"Application {app_name} exited with error:")
        logger.error(exc, exc_info=True)
        sys.exit(errno.EPERM)

    logger.info(f"App {app_name} completed successfully.")


def validate_training_fraction_in_correct_range(training_fraction: float):
    if not 0 < training_fraction < 1:
        raise ValueError("Training fraction must be in ]0, 1[ range.")


if __name__ == "__main__":
    main()
