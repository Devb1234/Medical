# 📅 Medical Appointment No-Show Prediction

This project aims to predict whether a patient will **miss or attend their medical appointment** using historical data. It includes:

* Exploratory Data Analysis (EDA)
* Machine Learning modeling
* A modern and interactive Streamlit web app for predictions

---

### 📌 Project Description

Missed appointments affect healthcare systems by wasting resources and delaying treatments. This model uses various features such as age, health conditions, and waiting time to predict no-shows.

---

### 📁 Dataset

* Source: [Kaggle - Medical Appointment No Shows](https://www.kaggle.com/joniarroba/noshowappointments)
* Size: 110,000+ records
* Features include:

  * Age, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap
  * SMS Received, Waiting Days
  * Scheduled & Appointment Weekdays
  * Target: `No-show`

---

### 📊 Exploratory Data Analysis (EDA)

Key insights include:

* Distribution of age and waiting days
* No-show rate trends
* Day-of-week influence on attendance
* Correlation between features

Plots used:

* Bar plots
* Count plots
* Heatmaps

---

### 🤖 Model Building

* **Preprocessing:**

  * Encoded categorical values
  * Scaled `age` and `waiting_days` using `StandardScaler`
* **Model Used:**

  * Random Forest Classifier
* **Metrics:**

  * Accuracy, Precision, Recall, F1-score
* **Files Saved:**

  * `no_show_model.pkl` (model)
  * `scaler.pkl` (standard scaler)

---

### 🖥️ Streamlit App

Interactive dashboard built using Streamlit.

#### Features:

* Sidebar for inputting patient details
* Predict button with clear results
* Probability score
* Modern and responsive layout

#### Run Locally:

1. Clone this repo:

   ```bash
   git clone https://github.com/Devb1234/medical-no-show-predictor.git
   cd medical-no-show-predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

---

### 🗃️ Project Structure

```
📦 medical-no-show-predictor/
├── app.py                 # Streamlit dashboard
├── dataset/               # Dataset folder
    ├── KaggleV2-May-2016.csv.zip
    ├── processed_no_show.csv
├── notebooks/
    ├── EDA and FE.ipynb   # Jupyter notebook with full EDA and FE
    ├── Model Training.ipynb  # Jupyter notebook with full ML
├── models/
    ├── no_show_model.pkl      # Trained ML model
    ├── scaler.pkl             # Scaler for preprocessing
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

---

### ✅ Requirements

* Python 3.8+
* pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, pickle

```bash
pip install -r requirements.txt
```

---

### 💡 Future Improvements

* Add SHAP-based model explainability
* Bulk predictions using CSV upload
* Deploy online using Streamlit Cloud or HuggingFace Spaces

---

### 🙌 Acknowledgements

* Dataset by [Joni Arroba on Kaggle](https://www.kaggle.com/joniarroba/noshowappointments)
* Inspired by real-world problems in public healthcare systems

---

### 🔗 Connect

Made with ❤️ by **Dev**

Feel free to connect on [GitHub](https://github.com/Devb1234)
