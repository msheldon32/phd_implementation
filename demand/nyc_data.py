from pyspark.sql import SparkSession
import pandas as pd

def get_spark_session():
    return SparkSession.builder.appName("NYC_POD").getOrCreate()

def get_data():
    spark = get_spark_session()
    df = spark.read.parquet("../../data/nyc_yellow")
    return df
