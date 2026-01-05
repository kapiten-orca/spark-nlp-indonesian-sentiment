from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sparknlp

PARQUET_PATH = "data/processed/reviews.parquet"

def main():
    #Inisialisasi Spark
    spark = SparkSession.builder \
        .appName("Parquet Checker") \
        .config("spark.driver.memory", "4G") \
        .getOrCreate()

    sparknlp.start(spark)

    print("\n Spark started successfully\n")

    #Cek parquet
    try:
        df = spark.read.parquet(PARQUET_PATH)
        print(f" Parquet loaded from: {PARQUET_PATH}\n")
    except Exception as e:
        print(" Failed to load Parquet")
        raise e

    # Menampilkan skema
    print(" Schema:")
    df.printSchema()

    # Cek kolom
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)

    if missing:
        print(f"\n Missing columns: {missing}")
        return
    else:
        print("\n Required columns exist")

    # Menampilkan jumlah baris
    total_rows = df.count()
    print(f"\nTotal rows: {total_rows}")

    if total_rows == 0:
        print(" Dataset is empty")
        return

    # Cek baris kosong
    null_text = df.filter(col("text").isNull()).count()
    empty_text = df.filter(col("text") == "").count()

    print(f" Null text rows: {null_text}")
    print(f" Empty text rows: {empty_text}")

    # Cek distribusi label
    print("\n Label distribution:")
    df.groupBy("label").count().show()

    # Data sampel
    print("\n Sample data:")
    df.select("text", "label").show(5, truncate=False)

    print("\n Parquet validation completed successfully")

    spark.stop()

if __name__ == "__main__":
    main()
