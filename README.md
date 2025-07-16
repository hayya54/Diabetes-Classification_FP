# Diabetes-Classification_FP

## 1. Pemahaman Bisnis
Tujuan utama dari proyek ini adalah untuk membangun model klasifikasi yang dapat memprediksi apakah seseorang memiliki diabetes atau tidak, berdasarkan serangkaian fitur kesehatan. Hal ini sangat berguna untuk deteksi dini dan intervensi medis.

## 2. Dataset
Dataset yang digunakan bersumber dari Kaggle dan diberi nama diabetes.csv. Dataset ini memiliki 768 entri (baris) dan 9 kolom fitur.
Fitur-fitur dalam Dataset:
a. Pregnancies: Jumlah kehamilan.
b. Glucose: Kadar glukosa plasma oral test 2 jam.
c. BloodPressure: Tekanan darah diastolik (mm Hg).
d. SkinThickness: Ketebalan lipatan kulit trisep (mm).
e Insulin: Kadar insulin serum 2 jam (mu U/ml).
f.BMI: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2).
g. DiabetesPedigreeFunction: Fungsi riwayat diabetes.
h. Age: Usia (tahun).
i. Outcome: Variabel target, menunjukkan apakah seseorang memiliki diabetes (1) atau tidak (0).

## 3. Pra-pemrosesan Data
3.1 Pengecekan Missing Values: Dilakukan pengecekan nilai null untuk setiap kolom, dan ditemukan bahwa tidak ada nilai null eksplisit dalam dataset. Namun, seperti yang disebutkan dalam statistik deskriptif, ada nilai 0 pada fitur-fitur yang seharusnya tidak memiliki nilai 0 (misalnya Glucose, BloodPressure), yang perlu ditangani sebagai missing values pada analisis lebih lanjut (meskipun tidak disebutkan penanganannya dalam kode yang diberikan, hal ini penting untuk dicatat dalam rincian analisis).

3.2 Penanganan Duplikat: Data duplikat diperiksa dan dihapus (df = df.dropDuplicates()). Setelah penghapusan, jumlah baris tetap 768, menunjukkan tidak ada duplikasi dalam dataset.

3.3 Pengubahan Nama Kolom: Nama kolom diubah menjadi huruf kecil (pregnancies, glucose, bloodPressure, skinThickness, insulin, bodyMassIndex, diabetesPedigreeFunction, age, outcome) untuk konsistensi.

3.4 Rekayasa Fitur (Feature Engineering): Semua kolom fitur (pregnancies hingga age) digabungkan menjadi satu kolom vektor bernama features menggunakan VectorAssembler. Ini merupakan langkah penting untuk mempersiapkan data bagi model machine learning yang mengharapkan input dalam format vektor.

3.5 Penskalaan Fitur (Feature Scaling): Kolom features kemudian diskalakan menggunakan StandardScaler dan disimpan dalam kolom scaledFeatures. Penskalaan ini membantu model machine learning bekerja lebih baik, terutama bagi algoritma yang sensitif terhadap skala fitur.

## 4. Pemodelan
Data dibagi menjadi set pelatihan (70%) dan set pengujian (30%) (training_data, test_data) untuk melatih dan mengevaluasi model.
Beberapa model klasifikasi diimplementasikan dan dievaluasi:
a. Logistic Regression: Model regresi logistik dilatih pada data pelatihan dan digunakan untuk membuat prediksi pada data pengujian.
b. Decision Tree Classifier: Model pohon keputusan dilatih dan dievaluasi.
c. Gaussian Naive Bayes: Model Naive Bayes Gaussian diimplementasikan.
d. Support Vector Machine (SVM): Model SVM, khususnya LinearSVC, digunakan untuk klasifikasi.
e. Gradient Boosting Classifier: Model penguat gradien (GBTClassifier) dilatih dan dievaluasi.
f. Random Forest Classifier: Model hutan acak (RandomForestClassifier) dilatih dan dievaluasi.

5. Evaluasi Model
Setiap model dievaluasi menggunakan metrik berikut:
a. Accuracy (Akurasi): Proporsi total prediksi yang benar.
b. Precision (Presisi): Proporsi instans positif yang diprediksi dengan benar dari total instans yang diprediksi positif.
c. Recall (Sensitivitas): Proporsi instans positif yang diprediksi dengan benar dari total instans positif aktual.
d. F1-Score: Rata-rata harmonik dari presisi dan recall.
e. Confusion Matrix: Tabel yang menampilkan jumlah true positives (TP), true negatives (TN), false positives (FP), dan false negatives (FN).

Hasil Evaluasi (Contoh dari Logistik Regresi):
Untuk Logistic Regression, metrik evaluasi yang diperoleh adalah:
a. Accuracy: 0.7705
b. Precision: 0.7093
c. Recall: 0.5052
d. F1-Score: 0.5909

Confusion matrix untuk Logistic Regression menunjukkan:
a. True Positive (TP): 49
b. True Negative (TN): 128
c. False Positive (FP): 20
d. False Negative (FN): 48

Perbandingan dan pemilihan model terbaik akan didasarkan pada metrik-metrik ini, dengan mempertimbangkan keseimbangan antara presisi dan recall, tergantung pada prioritas bisnis (misalnya, apakah lebih penting untuk tidak melewatkan kasus diabetes (recall tinggi) atau meminimalkan prediksi positif palsu (presisi tinggi)).

##  6. Library yang Digunakan
Berikut adalah library Python yang diimpor untuk analisis ini:

6.1 pyspark.sql.SparkSession: Untuk menginisialisasi sesi Spark dan bekerja dengan DataFrame Spark.

6.2 urllib.request, os, matplotlib.pyplot: Untuk pengambilan data, operasi sistem, dan visualisasi dasar.

6.3 pyspark.sql.functions.col, format_number: Untuk manipulasi dan pemformatan kolom DataFrame Spark.

6.4 pyspark.ml.feature.VectorAssembler: Untuk menggabungkan kolom fitur menjadi satu kolom vektor.

6.5 pyspark.ml.feature.StandardScaler: Untuk menskalakan fitur.

6.6 pyspark.ml.classification.LogisticRegression: Untuk model regresi logistik.

6.7 pyspark.ml.classification.DecisionTreeClassifier: Untuk model pohon keputusan.

6.8 pyspark.ml.classification.NaiveBayes: Untuk model Naive Bayes (implisit, karena Gaussian Naive Bayes digunakan).

6.9 pyspark.ml.classification.LinearSVC: Untuk model Support Vector Machine.

6.10 pyspark.ml.classification.GBTClassifier: Untuk model Gradient Boosting.

6.11 pyspark.ml.classification.RandomForestClassifier: Untuk model Random Forest.

6.12pyspark.ml.evaluation.BinaryClassificationEvaluator: Untuk mengevaluasi model klasifikasi biner.

6.13 pyspark.mllib.evaluation.MulticlassMetrics: Untuk menghitung metrik multi-kelas (seperti Confusion Matrix, Precision, Recall, F1-score).
