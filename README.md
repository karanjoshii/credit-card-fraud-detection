# Credit Card Fraud Detection Pipeline & Interactive Dashboard

An end-to-end Machine Learning pipeline and interactive Streamlit web dashboard built using **Apache Spark (PySpark 4.2.0)** and **Streamlit** to detect fraudulent transactions based on spending behavior.

---

## 🚀 Key Features & Completed Tasks

1. **Transaction Data Ingestion & Preprocessing**:
   - Ingests credit card transaction records using PySpark DataFrame API.
   - Extracts behavioral signals:
     - Transaction Hour (`trans_hour`)
     - Day of Week (`trans_dayofweek`)
     - High-Risk Nighttime Flag (`is_night`, 22:00 to 05:00)
     - Geodesic Distance (`distance_km`) between cardholder and merchant
     - Cardholder Age (`age`) derived from date of birth
2. **Outlier Detection & Removal**:
   - Detects spending amount anomalies using the **Interquartile Range (IQR)** method ($Q_3 + 3.0 \times IQR$) and Percentile Capping.
   - Preserves high-value fraud cases while filtering out illegitimate anomalies and recording errors.
3. **Categorical Feature Encoding**:
   - Converts categorical features (Merchant Category `category`, Gender `gender`) into numerical indices using PySpark's `StringIndexer`.
   - Combines numerical features and indexed categories into a feature vector using `VectorAssembler`.
4. **Decision Tree Classification**:
   - Trains a PySpark MLlib `DecisionTreeClassifier` with stratified sampling and minority class oversampling to effectively learn fraud patterns despite severe class imbalance (~0.36% fraud).
5. **Model Performance & AUC-ROC Evaluation**:
   - Evaluated via `BinaryClassificationEvaluator(metricName="areaUnderROC")`.
   - Generates full Receiver Operating Characteristic (ROC) curve coordinates and calculates Area Under Precision-Recall (AUC-PR).
   - Generates test set Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives).
6. **Fraud Distribution Visualization by Category**:
   - Compares total volume, fraud incident count, and fraud rate across merchant types (`shopping_net`, `misc_net`, `grocery_pos`, `health_fitness`, etc.).
7. **Interactive Streamlit Web Dashboard**:
   - **Tab 1: Category Distribution**: Interactive bar charts for fraud count, fraud rate (%), and transaction volume.
   - **Tab 2: Outlier & Spending Analysis**: Box plots (log scale), nighttime vs daytime fraud frequency, and distance vs amount scatter plot.
   - **Tab 3: Model Performance & AUC-ROC**: Visual ROC curve, confusion matrix heatmap, feature importance breakdown, and decision tree rule logic.
   - **Tab 4: Real-Time Fraud Predictor**: Live risk assessment simulator for custom transaction amounts, merchant types, and time.
   - **Tab 5: Data Explorer**: Searchable, filterable transaction table with CSV download.

---

## 📁 Project Structure

```
credit_card_fraud_detection/
├── data/
│   └── credit_card_transactions.csv   # Cleaned transaction dataset (5,000 records)
├── models/
│   ├── auc_roc_curve.png              # Generated ROC curve plot
│   ├── category_summary.csv           # Category fraud summary table
│   ├── confusion_matrix.png           # Confusion matrix heatmap
│   ├── feature_importance.csv         # Decision Tree feature importances
│   ├── feature_importance.png         # Feature importance chart
│   ├── fraud_distribution_by_category.png # Category distribution plot
│   ├── metrics.json                   # Pipeline evaluation metrics
│   ├── roc_data.json                  # FPR, TPR, thresholds for ROC curve
│   └── spending_behavior.png          # Spending distribution plots
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data loader, feature engineering, and outlier removal
│   ├── model.py                       # PySpark pipeline, Decision Tree training, and evaluation
│   ├── spark_session.py               # Reusable, optimized SparkSession configuration
│   └── visualization.py               # Analytical plotting functions
├── app.py                             # Interactive Streamlit Web Dashboard
├── run_pipeline.py                    # Standalone PySpark pipeline execution script
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## 🛠️ How to Run

### 1. Execute the End-to-End PySpark Pipeline (CLI)

Run the pipeline from the command line:
```bash
python run_pipeline.py
```
*(On Windows with Python 3.11: `& "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" run_pipeline.py`)*

Outputs:
- Data loading and outlier removal statistics
- Pipeline training logs and Decision Tree feature importances
- AUC-ROC score, Accuracy, F1-score, Precision, Recall, and Confusion Matrix
- Generated visual plots saved directly into `models/`

### 2. Launch the Interactive Streamlit Dashboard

Run the Streamlit application:
```bash
streamlit run app.py
```
*(On Windows with Python 3.11: `& "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📈 Model Performance Summary

| Metric | Score |
| :--- | :--- |
| **Area Under ROC (AUC-ROC)** | **1.0000** |
| **Area Under PR (AUC-PR)** | **1.0000** |
| **Accuracy** | **99.25%** |
| **Weighted F1-Score** | **0.9942** |
| **Fraud Recall (Sensitivity)** | **100.00%** (4/4 test frauds caught) |
| **Non-Fraud Specificity** | **99.25%** (924/931 legit approved) |

### Top Predictors Identified by Decision Tree:
1. **Transaction Amount (`amt`)** (~86.3% importance): Fraudulent transactions average \$632.45 vs \$66.52 for legitimate purchases.
2. **Nighttime Flag (`is_night`)** (~9.7% importance): 100% of recorded fraudulent transactions occurred between 22:00 and 05:00.
3. **Merchant Category (`category_idx`)** (~3.9% importance): `shopping_net` and `misc_net` carry the highest historical risk.
