import sparknlp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower

spark = SparkSession.builder \
    .appName("Prepare PRDECT-ID Data") \
    .config("spark.driver.memory", "4G") \
    .getOrCreate()

sparknlp.start(spark)

# Mengambil data raw
df_raw = spark.read.option("header", True).csv(
    "data/raw/prdect_id.csv"
)

# Standarisasi nama kolom
df = df_raw.select(
    col("Customer Review").alias("text"),
    col("Sentiment").alias("label")
)

# Standarisasi label
df = df.withColumn(
    "label",
    when(lower(col("label")) == "positive", 1)
    .when(lower(col("label")) == "negative", 0)
    .otherwise(None)
)

# Membersihkan data
df = df.dropna()
df = df.filter(col("text") != "")

# Menampilkan data
df.printSchema()
df.show(5, truncate=False)
df.groupBy("label").count().show()

# Menyimpan paraquet dengan data yang sudah standar
df.write.mode("overwrite").parquet("data/processed/reviews.parquet")

print("Clean Parquet saved successfully")

spark.stop()
