from pyspark.sql import SparkSession
import sparknlp

spark = sparknlp.start()

df = spark.read.option("header", True).csv("data/raw/prdect_id.csv")

df.printSchema()
df.show(5)
