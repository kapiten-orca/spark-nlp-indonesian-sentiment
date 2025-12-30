from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
import sparknlp

#Menginisialisasi Spark
spark = sparknlp.start(memory="8G")

#Menginisialisasi
loaded_model = PipelineModel.load("models/sentiment_model")

#Mengambil data csv untuk dibandingkan
new_data = spark.read.option("header", "true") \
    .csv("data/raw/prdect_id.csv") # Make sure filename matches yours

#Standarisasi kolom
#Mengubah 'Customer Review' menjadi 'text'
#Mengubah 'Sentiment' ("Positive"/"Negative") menjadi angka (1.0/0.0)
prepared_data = new_data \
    .withColumnRenamed("Customer Review", "text") \
    .withColumn("true_label", 
        when(col("Sentiment") == "Positive", 1.0)
        .when(col("Sentiment") == "Negative", 0.0)
        .otherwise(None) 
    )

#Memfilter baris yang gagal diubah (optional safety)
prepared_data = prepared_data.filter(col("true_label").isNotNull())

#Menjalankan prediksi dari model yang dibuat
predictions = loaded_model.transform(prepared_data)

#Show side-by-side comparison (First 10 rows)
predictions.select("text", "Sentiment", "true_label", "prediction").show(10, truncate=50)

#Menghitung total baris
total_count = predictions.count()

#Agregasi berdasarkan hasil prediksi
stats = predictions.groupBy("prediction").count() \
    .withColumn("percentage", (F.col("count") / total_count) * 100) \
    .withColumn("sentiment_name", 
        F.when(F.col("prediction") == 1.0, "Positive")
        .otherwise("Negative"))

print("Sentiment Prediction Summary:")
stats.select("sentiment_name", "count", "percentage").show()

#Memfilter kolom yang akan disimpan
output_df = predictions.select("text", "Sentiment", "true_label", "prediction")

#Simpan ke folder
output_path = "data/results/prediction_results"
output_df.repartition(1).write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)

print(f"Prediction result saved successfully at: {output_path}")