from pyspark.sql.functions import col
import sparknlp

spark = sparknlp.start()

df = spark.read.option("header", True).csv("data/raw/prdect_id.csv")

df = df.select(
    col("Customer Review").alias("text"),
    col("Sentiment").alias("label")
)

df = df.dropna().dropDuplicates()

df.show(5)
