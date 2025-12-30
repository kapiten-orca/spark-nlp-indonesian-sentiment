import sparknlp
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import SQLTransformer  
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparknlp.annotator import (
    Tokenizer,
    Normalizer,
    StopWordsCleaner,
    BertEmbeddings,
    SentenceEmbeddings
)

#Inisialisasi SPARK & Data:
spark = sparknlp.start(gpu=True,memory="8G")
df = spark.read.parquet("data/processed/reviews.parquet")

#Mengubah kolom teks mentah menjadi format dokumen yang dipahami oleh Spark NLP.
document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

#Memecah teks menjadi potongan kata (token).
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

#Membersihkan teks (menghapus tanda baca, mengubah ke huruf kecil).
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

#Menghapus kata-kata umum yang kurang bermakna (seperti "dan", "yang", "di").
stopwords = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("clean_tokens")

#Menggunakan model pre-trained BERT spesifik bahasa Indonesi untuk mengubah teks menjadi representasi numerik (vektor)
embeddings = BertEmbeddings.pretrained(
    "bert_embeddings_base_indonesian_1.5g", "id"
).setInputCols(["document", "clean_tokens"]) \
 .setOutputCol("bert")

#Menggabungkan vektor kata-kata menjadi satu vektor tunggal yang mewakili seluruh kalimat
sentence = SentenceEmbeddings() \
    .setInputCols(["document", "bert"]) \
    .setOutputCol("features") \
    .setPoolingStrategy("AVERAGE")

#Mengubah format data dari objek internal Spark NLP menjadi vector yang bisa dibaca oleh pustaka Machine Learning standar (MLlib).
finisher = EmbeddingsFinisher() \
    .setInputCols(["features"]) \
    .setOutputCols(["features_vec"]) \
    .setOutputAsVector(True)

#Menggunakan perintah SQL untuk mengekstrak elemen pertama dari array fitur agar siap digunakan oleh model klasifikasi.
vector_extractor = SQLTransformer(
    statement="SELECT *, features_vec[0] as features_final FROM __THIS__"
)

#Algoritma klasifikasi yang memprediksi label (misalnya: positif/negatif) berdasarkan fitur dari BERT.
classifier = LogisticRegression(
    featuresCol="features_final", 
    labelCol="label",
    maxIter=20
)

#Menggabungkan semua tahap di atas menjadi satu alur kerja yang rapi.
pipeline = Pipeline(stages=[
    document,
    tokenizer,
    normalizer,
    stopwords,
    embeddings,
    sentence,
    finisher,
    vector_extractor,
    classifier
])

#Melatih model menggunakan data training (80% dari total data) dan melakukan prediksi pada data tes (20%).
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
predictions = model.transform(test)

# Inisialisasi evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

# Hitung akurasi
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy:.2%}")

#Menampilkan 5 hasil pertama
predictions.select("text", "label", "prediction").show(5)


#Model disimpan agar bisa digunakan tanpa perlu melatih ulang.
model.write().overwrite().save("models/sentiment_model")

print("Model saved successfully!")