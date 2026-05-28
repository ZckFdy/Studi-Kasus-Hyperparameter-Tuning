# Proyek Advanced Hyperparameter Tuning: Implementasi ML Workflow

Repositori ini memuat implementasi praktis dari *Machine Learning Workflow* pada tahap **Model Tuning & Optimization**. Proyek ini mendemonstrasikan bagaimana meningkatkan performa model dasar (*baseline*) secara signifikan menggunakan metode **Bayesian Optimization** untuk mencari kombinasi hyperparameter paling optimal secara cerdas dan efisien.

---

## Kasus 1: Klasifikasi Risiko Kredit (German Credit Dataset)

Kasus ini bertujuan untuk memprediksi kelayakan kredit seorang nasabah (`Good` atau `Bad`) menggunakan dataset *German Credit* dari OpenML.

### Alur Kerja & Pra-pemrosesan Data
* **Pemuatan Data:** Dataset diunduh secara otomatis menggunakan fungsi `fetch_openml(name='credit-g')` yang menghasilkan 700 data latih (*Training Data*) dan 300 data uji (*Testing Data*).
* **Target Encoding:** Mengubah variabel target kategori menjadi format numerik biner menggunakan `LabelEncoder`.

### Optimisasi Model Klasifikasi (Bayesian Optimization)
Dibandingkan dengan *Grid Search* yang memakan waktu lama atau *Random Search* yang berbasis faktor keberuntungan, proyek ini menerapkan **Bayesian Optimization** (menggunakan pustaka seperti `scikit-optimize` atau sejenisnya):
* **Algoritma Target:** Mengoptimalkan model ensemble klasifikasi.
* **Mekanisme Kerja:** Algoritma ini membangun model probabilitas dari fungsi tujuan (*surrogate model*) untuk memprediksi hyperparameter mana yang akan memberikan hasil terbaik berdasarkan hasil eksperimen sebelumnya. 
* **Output:** Menghasilkan kombinasi parameter terbaik (`Best parameters (Bayesian Optimization)`) yang meminimalkan salah klasifikasi pada penentuan risiko kredit nasabah.

---

## Kasus 2: Regresi Prediksi Harga Rumah (California Housing Dataset)

Kasus ini bertujuan untuk memprediksi harga median rumah di wilayah California menggunakan *California Housing Dataset* (terdiri dari 14.448 data latih dan 6.192 data uji).

### Alur Kerja & Pra-pemrosesan Data
* **Data Splitting:** Membagi dataset dengan proporsi ketat: 70% sebagai *training set* dan 30% sebagai *testing set*.
* **Feature Scaling:** Seluruh fitur numerik distandardisasi menggunakan `StandardScaler` untuk memastikan performa stabilitas algoritma saat proses optimisasi berjalan.

### Optimisasi Model Regresi (Hyperparameter Tuning via Cross-Validation)
Proyek ini mengoptimalkan algoritma pohon tingkat lanjut seperti **Random Forest Regressor** dengan memantau metrik evaluasi secara berlapis:
* **Ruang Pencarian Hyperparameter:** Melakukan pencarian kombinasi dinamis pada parameter kunci, termasuk:
    * `n_estimators`: Jumlah pohon keputusan yang akan dibangun (misal: 132, 138).
    * `max_depth`: Batas kedalaman maksimum pohon untuk menghindari overfitting (misal: 17, 18, 19).
    * `min_samples_split` & `min_samples_leaf`: Batas minimal sampel untuk memecah simpul daun.
* **Metode Evaluasi:** Menggunakan **3-Fold Cross-Validation** (`Fitting 3 folds for each of 1 candidates`) untuk memastikan bahwa parameter yang terpilih tidak hanya unggul pada satu sub-sampel data saja, melainkan memiliki performa generalisasi yang stabil dan konsisten di lingkungan produksi.

---

## Library Python Utama yang Digunakan
* `sklearn.datasets` (`fetch_openml`, `fetch_california_housing`) - Mengunduh dataset benchmark publik berskala besar.
* `sklearn.preprocessing` (`LabelEncoder`, `StandardScaler`) - Melakukan standardisasi skala dan encoding data target.
* `sklearn.model_selection` (`train_test_split`, `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score`) - Mengelola validasi silang dan pembagian data secara konsisten.
* `skopt` / `scikit-optimize` - Implementasi algoritma optimisasi berbasis teori Bayes (*Bayesian Optimization*).
