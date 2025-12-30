from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
import sparknlp

# 1. Menjalankan Spark
spark = sparknlp.start(memory="8G")

# 2. Load model
loaded_model = PipelineModel.load("models/sentiment_model")

# 3. Mengambil data csv untuk dibandingkan
new_data = spark.read.option("header", "true") \
    .csv("data/raw/prdect_id.csv") # Make sure filename matches yours

# 4. Standarisasi kolom
# A. Mengubah 'Customer Review' menjadi 'text'
# B. Mengubah 'Sentiment' ("Positive"/"Negative") menjadi angka (1.0/0.0)
prepared_data = new_data \
    .withColumnRenamed("Customer Review", "text") \
    .withColumn("true_label", 
        when(col("Sentiment") == "Positive", 1.0)
        .when(col("Sentiment") == "Negative", 0.0)
        .otherwise(None) 
    )

# 5. Memfilter baris yang gagal diubah (optional safety)
prepared_data = prepared_data.filter(col("true_label").isNotNull())

# 6. Menjalankan prediksi dari model yang dibuat
predictions = loaded_model.transform(prepared_data)

# 7. Menghitung akurasi
evaluator = MulticlassClassificationEvaluator(
    labelCol="true_label",    # Jawaban yang benar
    predictionCol="prediction", # Hasil prediksi model
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("="*30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("="*30)

# 8. Show side-by-side comparison (First 10 rows)
predictions.select("text", "Sentiment", "true_label", "prediction").show(10, truncate=50)