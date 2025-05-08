# implementation-of-prediction-model-for-credit-risk


[Final task] Data Scientist project based internship at ID/X Partners x Rakamin Academy
----------------------------
## About Company
Id/x partners didirikan pada tahun 2002 oleh para mantan bankir dan konsultan manajemen yang memiliki pengalaman luas dalam siklus kredit dan manajemen proses, pengembangan penilaian, dan manajemen kinerja. Gabungan pengalaman kami telah melayani perusahaan-perusahaan di seluruh wilayah Asia dan Australia dan di berbagai industri, khususnya jasa keuangan, telekomunikasi, manufaktur, dan ritel. 

Id/x partners menyediakan layanan konsultasi yang mengkhususkan diri dalam memanfaatkan solusi data analytic and decisioning (DAD) yang dikombinasikan dengan manajemen risiko dan disiplin pemasaran yang terintegrasi untuk membantu klien mengoptimalkan profitabilitas portofolio dan proses bisnis.

Layanan konsultasi yang komprehensif dan solusi teknologi yang ditawarkan oleh id/x partners menjadikannya sebagai penyedia layanan satu atap.


## Background Information
Dalam industri keuangan, terutama pada lembaga pemberi pinjaman seperti bank atu fintech, penilaian risiko kredit merupakan aspek krusial dalam pengambilan keputusan. Pemberian pinjaman yang tidak terukur akan membuat banyak kerugian finansial dan kestabilan perusahaan. Oleh karena itu, pengendalian risiko pinjaman sangat penting untuk dilakukan agar dapat melakukan evaluasi kelayakan pinjaman.

Untuk mengatasi tantangan ini, penerapan Machine Learning (ML) sangat diperlukan untuk dapat melakukan fungsi otomatisasi dalam melakukan evaluasi kredit dan meningkatkan akurasi. Dengan demikian, proyek ini bertujuan untuk membangun model machine learning yang mampu mengklasifikasikan risiko pinjaman secara lebih akurat, membantu lembaga keuangan dalam mengurangi kredit macet, serta mempercepat dan mengoptimalkan proses pengambilan keputusan dalam pemberian kredit.



## Project

Sebagai Data Scientist intern di ID/X Partners, dalam sebuah proyek dari perusahaan pemberi pinjaman (multifinance), dimana client ingin meningkatkan keakuratan dalam menilai dan mengelola risiko kredit, sehingga dapat mengoptimalkan keputusan bisnis mereka dan mengurangi potensi kerugian.

Tugas proyek ini adalah mengembangkan model machine learning yang dapat memprediksi risiko kredit (credit risk) berdasarkan dataset yang disediakan, yang mencakup data pinjaman yang disetujui dan ditolak. Dalam pengembangan modelnya juga perlu melakukan beberapa tahap dimulai dengan Data Understanding, Exploratory Data Analysis (EDA), Data Preparation, Data Modelling, dan Evaluation.


### Folder
    ├── data/ # berisi datasets 
    ├── outputs/ # visualisai hasil sebelum SMOTE
    ├── src code/ # source code Final_Task
    └── README.md # Dokumentasi

### Library

* pandas
* numpy
* scikit-learn
* SMOTE
* XGB
* matplotlib
* seaborn

### Datasets
Datasets yang digunakan dalam project ini adalah merupakah dataset yang berisi informasi mengenai pinjaman kredit, mulai dari jumlah pinjaman, total yang lunas dan belum lunas, hingga sisa pembayaran selanjutnya. 

Dimensi dataset ini terdapat 468225 baris dan 75 fitur dengan tipe data terdiri dari time series, kategorik, dan numberik yang diantaranya adalah :

Time Series :
* issue_d
* earlist_cr_line
* last_pymnt_d
* next_pymnt_d
* last_credit_pull_d

Kategorik :
* term
* grade
* sub_grade
* loan_status
* home_ownership
* application_type
* purpose
* intial_list_status

Numberik :
* funded_amnt
* last_pymnt_amnt
* total_pymnt_amnt
* total_pymnt_inv
* total_pymnt
* total_rec_int
* tot_cur_bal
* tot_coll_amnt

### Model
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "Logistic Regression": LogisticRegression(solver='saga', penalty='l2', C=10, max_iter=500)
    }

## Evaluation
### Confusion Matrix
*Hasil Confusion Matrix Menggunakan SMOTE Oversampling*
![Hasil Confusion Matrix Menggunakan SMOTE Oversampling](outputs/Hasil%20Confusion%20Matrix%20SMOTE%20Oversampling.png)

*Hasil Confusion Matrix Sebelum SMOTE Oversampling*
![Hasil Confusion Matrix sebelum SMOTE](outputs/Hasil%20Confusion%20Matrix%20tanpa%20SMOTE%20Oversampling.png)

Jika dibandingkan nilai confusion matrix pada model dengan akurasi dan presisi yang tinggi yaitu Logistic Regression dan Random Forest sangat jauh dibandingkan dengan menggunakan SMOTE, terlihat bahwa model masih tidak memiliki nilai yang baik dalam memprediksi pinjaman yang berisiko dan tidak berisiko.

### Feature Importance
*Hasil Feature Importance Antara XGBoost dan Random Forest*

*XGBoost*
![Hasil Feature Importance](outputs/Hasil%20Feature%20Importance%20XGB.png)
*Random Forest*
![Hasil Feature Importance](outputs/Hasil%20Feature%20Importance%20RF.png)
Pada kedua model fitur yang sangat penting dalam pemodelan terdapat pada fitur **Recoveries**.

## Conclusion
Antara akurasi dan presisi dalam model ini dapat dilihat dari bagaimana XGBoost lebih baik dalam menyeimbangkan presisi dan recall dibandingkan Random Forest. XGBoost memiliki FP dan FN yang lebih kecil, sehingga memberikan akurasi keseluruhan yang lebih tinggi. Namun, jika kita ingin menyesuaikan model sesuai kebutuhan bisnis—misalnya lebih fokus pada recall agar tidak terlalu banyak pinjaman buruk yang lolos—kita bisa menyesuaikan threshold atau menggunakan metode balancing tambahan."

5 Februari 2025

Khirana Dwicahyo,.