from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
import sparknlp

#inisialisasi spark
spark = sparknlp.start(memory="8G")

#menggunakan model yang sudah disimpan sebelumnya
loaded_model = PipelineModel.load("models/sentiment_model")

# mengambil data raw untuk di analisis
data = spark.read.option("header", "true") \
    .csv("data/raw/prdect_id.csv")

# menyesuaikan kolom yang akan digunakan untuk analisis
prepared_data = data \
    .withColumnRenamed("Customer Review", "text") \
    .withColumn(
        "true_label",
        when(col("Sentiment") == "Positive", 1.0)
        .when(col("Sentiment") == "Negative", 0.0)
        .otherwise(None)
    ) \
    .filter(col("true_label").isNotNull())

# mengukur sentimen
predictions = loaded_model.transform(prepared_data)

predictions = predictions.withColumn(
    "prediction",
    col("prediction").cast("double")
)

predictions = predictions.withColumn(
    "predicted_sentiment",
    when(col("prediction") == 1.0, "Positive").otherwise("Negative")
)

# data distribusi sentimen
sentiment_distribution = predictions.groupBy("predicted_sentiment") \
    .count()

# data hubungan sentimen per kategori
category_sentiment = predictions.groupBy(
    "Category", "predicted_sentiment"
).count()

# data hubungan sentimen dan lokasi
location_sentiment_score = predictions.withColumn(
    "sentiment_score",
    when(col("prediction") == 1.0, 1).otherwise(-1)
).groupBy("Location") \
 .agg(
     F.avg("sentiment_score").alias("avg_sentiment_score"),
     F.count("*").alias("total_reviews")
 )

# menyimpan hasil analisis ke csv
sentiment_distribution.write.mode("overwrite") \
    .option("header", "true") \
    .csv("data/results/sentiment_distribution")

category_sentiment.write.mode("overwrite") \
    .option("header", "true") \
    .csv("data/results/category_sentiment")

location_sentiment_score.write.mode("overwrite") \
    .option("header", "true") \
    .csv("data/results/location_sentiment")

print("Aggregated sentiment data saved successfully!")
