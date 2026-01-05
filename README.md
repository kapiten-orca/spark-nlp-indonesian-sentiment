### System Requirements
- Java 11
- Python 3.10+

### Python Dependencies
- pyspark 3.5.1
- spark-nlp 5.2.2
- numpy
- pandas

## Langkah-langkah Menjalankan

```bash
python save_parquet.py      # Menyimpan data dalam format parquet
python check_parquet.py     # Mengecek file parquet
python train_model.py       # Melatih model
python acuracy.py           # Mengecek akurasi model
python prediction.py        # Prediksi dan simpan ke CSV
