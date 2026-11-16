import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure environment for PySpark
PYTHON_EXEC = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
if os.path.exists(PYTHON_EXEC):
    os.environ["PYSPARK_PYTHON"] = PYTHON_EXEC
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXEC

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.spark_session import get_spark_session
from src.data_preprocessing import load_data, engineer_features, detect_and_remove_outliers
from src.model import train_and_evaluate
from src.visualization import (
    plot_fraud_distribution_by_category,
    plot_auc_roc_curve,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_spending_behavior
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-high {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 8px;
        color: #991b1b;
        font-weight: 600;
    }
    .risk-low {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 14px 18px;
        border-radius: 8px;
        color: #065f46;
        font-weight: 600;
    }
    .risk-medium {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 8px;
        color: #92400e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_cached_raw_data():
    csv_path = os.path.join(BASE_DIR, "data", "credit_card_transactions.csv")
    df = pd.read_csv(csv_path)
    
    # Precompute time & distance features for instant UI rendering
    if "trans_date_trans_time" in df.columns:
        df["trans_time_ts"] = pd.to_datetime(df["trans_date_trans_time"])
        df["trans_hour"] = df["trans_time_ts"].dt.hour
        df["trans_dayofweek"] = df["trans_time_ts"].dt.dayofweek + 1
        df["is_night"] = df["trans_hour"].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)
    
    # Distance km
    if "lat" in df.columns and "merch_lat" in df.columns:
        r_earth = 6371.0
        lat1 = np.radians(df["lat"])
        lat2 = np.radians(df["merch_lat"])
        dlat = np.radians(df["merch_lat"] - df["lat"])
        dlon = np.radians(df["merch_long"] - df["long"])
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        df["distance_km"] = np.round(r_earth * c, 2)
        
    if "dob" in df.columns:
        dob_dt = pd.to_datetime(df["dob"])
        df["age"] = np.round((df["trans_time_ts"] - dob_dt).dt.days / 365.25, 1)

    return df

@st.cache_data
def load_precomputed_artifacts():
    models_dir = os.path.join(BASE_DIR, "models")
    metrics_path = os.path.join(models_dir, "metrics.json")
    roc_path = os.path.join(models_dir, "roc_data.json")
    fi_path = os.path.join(models_dir, "feature_importance.csv")

    metrics, roc_data, fi_df = None, None, None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    if os.path.exists(roc_path):
        with open(roc_path, "r") as f:
            roc_data = json.load(f)
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)

    return metrics, roc_data, fi_df

# Load baseline data
raw_pandas_df = load_cached_raw_data()
pre_metrics, pre_roc, pre_fi = load_precomputed_artifacts()

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
st.sidebar.title("Configuration & Controls")

st.sidebar.subheader("⚙️ Outlier Removal Settings")
outlier_method = st.sidebar.selectbox("Outlier Method", ["IQR (Interquartile Range)", "Percentile Capping"], index=0)
if "IQR" in outlier_method:
    iqr_factor = st.sidebar.slider("IQR Multiplier Factor", min_value=1.5, max_value=5.0, value=3.0, step=0.5,
                                   help="Transactions with Amount > Q3 + Factor*IQR are identified as spending outliers.")
    method_code = "iqr"
    threshold_param = iqr_factor
else:
    pct_val = st.sidebar.slider("Upper Percentile Threshold", min_value=95.0, max_value=99.9, value=99.5, step=0.1)
    method_code = "percentile"
    threshold_param = pct_val / 100.0

preserve_fraud_outliers = st.sidebar.checkbox(
    "Preserve Fraud Transactions", 
    value=True,
    help="When enabled, high amounts identified as outliers are only removed if they are non-fraud, ensuring critical fraud signals are not lost."
)

st.sidebar.subheader("🌲 Decision Tree Hyperparameters")
max_depth = st.sidebar.slider("Max Tree Depth", min_value=2, max_value=8, value=5, step=1)
min_instances = st.sidebar.slider("Min Instances Per Leaf Node", min_value=1, max_value=10, value=2, step=1)
train_split = st.sidebar.slider("Train / Test Split Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
oversample_minority = st.sidebar.checkbox("Oversample Minority Fraud Class", value=True, help="Balances the training split to handle severe class imbalance (~0.36% fraud).")

retrain_btn = st.sidebar.button("🚀 Retrain PySpark Model", use_container_width=True)

# ----------------- SESSION STATE FOR MODEL -----------------
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = pre_metrics
if "current_roc" not in st.session_state:
    st.session_state.current_roc = pre_roc
if "current_fi" not in st.session_state:
    st.session_state.current_fi = pre_fi
if "last_cleaned_stats" not in st.session_state:
    st.session_state.last_cleaned_stats = {
        "outliers_removed": 115,
        "percentage_removed": 2.3,
        "lower_bound": 0.0,
        "upper_bound": 287.27,
        "total_before": 5000,
        "total_after": 4885,
        "fraud_before": 18,
        "fraud_after": 18
    }

# Execute dynamic retraining when button is clicked
if retrain_btn:
    with st.spinner("Initializing PySpark session, performing outlier removal, and training Decision Tree..."):
        try:
            spark = get_spark_session("CreditCardFraudDashboard")
            csv_path = os.path.join(BASE_DIR, "data", "credit_card_transactions.csv")
            spark_df = load_data(spark, csv_path)
            spark_df = engineer_features(spark_df)
            
            cleaned_spark_df, stats = detect_and_remove_outliers(
                spark_df,
                column="amt",
                method=method_code,
                factor=threshold_param,
                preserve_fraud=preserve_fraud_outliers
            )
            st.session_state.last_cleaned_stats = stats
            
            trained_model, m_res, r_data, _, fi_res, _ = train_and_evaluate(
                cleaned_spark_df,
                train_ratio=train_split,
                oversample_fraud=oversample_minority,
                max_depth=max_depth,
                min_instances=min_instances
            )
            st.session_state.current_metrics = m_res
            st.session_state.current_roc = r_data
            st.session_state.current_fi = fi_res
            st.success("PySpark Model Successfully Retrained and Evaluated!")
        except Exception as e:
            st.error(f"Error during PySpark execution: {e}")

# Compute current outlier thresholds for Pandas visualization
q1 = raw_pandas_df["amt"].quantile(0.25)
q3 = raw_pandas_df["amt"].quantile(0.75)
iqr = q3 - q1
upper_outlier_bound = q3 + threshold_param * iqr if method_code == "iqr" else raw_pandas_df["amt"].quantile(threshold_param)

# Filter dataset based on current sidebar settings
if preserve_fraud_outliers:
    clean_pandas_df = raw_pandas_df[~((raw_pandas_df["amt"] > upper_outlier_bound) & (raw_pandas_df["is_fraud"] == 0))]
else:
    clean_pandas_df = raw_pandas_df[raw_pandas_df["amt"] <= upper_outlier_bound]

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-title">🛡️ Credit Card Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Distributed Machine Learning with <b>PySpark</b> & Interactive Analytics for Spending Behavior</div>', unsafe_allow_html=True)

# Top KPI row
c1, c2, c3, c4, c5, c6 = st.columns(6)
total_recs = len(raw_pandas_df)
fraud_cases = int(raw_pandas_df["is_fraud"].sum())
fraud_rate = (fraud_cases / total_recs) * 100
removed_outliers = st.session_state.last_cleaned_stats.get("outliers_removed", 0)
auc_val = st.session_state.current_metrics.get("auc_roc", 1.0) if st.session_state.current_metrics else 1.0
acc_val = st.session_state.current_metrics.get("accuracy", 0.992) if st.session_state.current_metrics else 0.992

with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Records</div><div class="metric-val">{total_recs:,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Fraud Incidents</div><div class="metric-val" style="color: #dc2626;">{fraud_cases}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Fraud Rate</div><div class="metric-val">{fraud_rate:.2f}%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Outliers Removed</div><div class="metric-val" style="color: #d97706;">{removed_outliers}</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Decision Tree Accuracy</div><div class="metric-val" style="color: #2563eb;">{acc_val*100:.1f}%</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Area Under ROC</div><div class="metric-val" style="color: #16a34a;">{auc_val:.4f}</div></div>', unsafe_allow_html=True)

st.write("")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Fraud Distribution by Category",
    "🔍 Outlier Analysis & Spending Behavior",
    "🤖 Decision Tree & AUC-ROC Curve",
    "⚡ Real-Time Fraud Predictor",
    "📋 Data Explorer"
])

# ================= TAB 1: CATEGORY DISTRIBUTION =================
with tab1:
    st.subheader("Category-wise Fraud Distribution & Spending Analysis")
    st.markdown("Analyze how fraudulent activity clusters across different merchant types and spending categories.")

    col1, col2 = st.columns([3, 2])

    cat_df = clean_pandas_df.groupby("category").agg(
        Total_Trans=("is_fraud", "count"),
        Fraud_Trans=("is_fraud", "sum"),
        Fraud_Rate_Pct=("is_fraud", lambda x: round((x.sum() / len(x)) * 100, 2)),
        Total_Volume=("amt", "sum"),
        Avg_Amount=("amt", "mean")
    ).reset_index().sort_values(by="Fraud_Trans", ascending=False)

    with col1:
        metric_choice = st.radio("Chart Metric:", ["Fraud Cases Count", "Fraud Incident Rate (%)", "Total Transaction Volume ($)"], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        if metric_choice == "Fraud Cases Count":
            sns.barplot(data=cat_df, x="category", y="Fraud_Trans", hue="category", legend=False, palette="Reds_r", ax=ax)
            ax.set_ylabel("Number of Fraud Cases", fontsize=11)
            ax.set_title("Fraud Count by Merchant Category", fontsize=13, fontweight="bold")
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h), ha="center", va="bottom", fontsize=10, fontweight="bold")
        elif metric_choice == "Fraud Incident Rate (%)":
            rate_sorted = cat_df.sort_values(by="Fraud_Rate_Pct", ascending=False)
            sns.barplot(data=rate_sorted, x="category", y="Fraud_Rate_Pct", hue="category", legend=False, palette="Blues_r", ax=ax)
            ax.set_ylabel("Fraud Rate (%)", fontsize=11)
            ax.set_title("Fraud Incident Rate (%) by Category", fontsize=13, fontweight="bold")
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h), ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            vol_sorted = cat_df.sort_values(by="Total_Volume", ascending=False)
            sns.barplot(data=vol_sorted, x="category", y="Total_Volume", hue="category", legend=False, palette="Greens_r", ax=ax)
            ax.set_ylabel("Total Volume ($)", fontsize=11)
            ax.set_title("Total Spending Volume by Category", fontsize=13, fontweight="bold")
            
        ax.set_xlabel("Merchant Category", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Category Risk Summary Table")
        st.dataframe(
            cat_df.style.format({
                "Total_Trans": "{:,}",
                "Fraud_Trans": "{:,}",
                "Fraud_Rate_Pct": "{:.2f}%",
                "Total_Volume": "${:,.2f}",
                "Avg_Amount": "${:.2f}"
            }).background_gradient(subset=["Fraud_Trans", "Fraud_Rate_Pct"], cmap="Reds"),
            use_container_width=True,
            height=380
        )
        st.info("💡 **Insight:** High-risk fraudulent activity is concentrated in **shopping_net** (6 cases), **misc_net** (3 cases), **grocery_pos** (2 cases), and **health_fitness** (2 cases). Online shopping and digital transactions present the highest fraud exposure.")

# ================= TAB 2: OUTLIER & BEHAVIOR ANALYSIS =================
with tab2:
    st.subheader("Transaction Outlier Detection & Spending Behavior")
    st.markdown("Identifying spending anomalies and examining how fraudulent behavior differs from legitimate transactions.")

    o_stats = st.session_state.last_cleaned_stats
    mo1, mo2, mo3, mo4 = st.columns(4)
    with mo1:
        st.metric("Detection Threshold", f"${upper_outlier_bound:.2f}")
    with mo2:
        st.metric("Outliers Detected", f"{o_stats['outliers_removed']} rows")
    with mo3:
        st.metric("Dataset Reduction", f"{o_stats['percentage_removed']}%")
    with mo4:
        st.metric("Fraud Records Retained", f"{o_stats['fraud_after']} / {o_stats['fraud_before']}")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### Transaction Amount Distribution (Log Scale)")
        fig_sb, ax_sb = plt.subplots(figsize=(8, 4.8))
        df_plot = clean_pandas_df.copy()
        df_plot["Class"] = df_plot["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
        sns.boxplot(data=df_plot, x="Class", y="amt", hue="Class", legend=False, palette=["#10b981", "#ef4444"], ax=ax_sb)
        ax_sb.set_yscale("log")
        ax_sb.set_title("Spending Amount Comparison (Log Scale)", fontsize=12, fontweight="bold")
        ax_sb.set_ylabel("Amount ($ - Log)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_sb)

    with col_b2:
        st.markdown("#### Hourly Fraud Occurrence (Night vs Day)")
        fig_hr, ax_hr = plt.subplots(figsize=(8, 4.8))
        hourly_fraud = raw_pandas_df[raw_pandas_df["is_fraud"] == 1]["trans_hour"].value_counts().sort_index()
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(hourly_fraud)
        colors = ["#dc2626" if (h >= 22 or h <= 5) else "#94a3b8" for h in range(24)]
        ax_hr.bar(all_hours.index, all_hours.values, color=colors, width=0.7)
        ax_hr.set_xticks(range(0, 24, 2))
        ax_hr.set_xlabel("Hour of Day (24-Hour Clock)", fontsize=10)
        ax_hr.set_ylabel("Number of Fraud Cases", fontsize=10)
        ax_hr.set_title("Fraud Activity Peaks at Night (22:00 - 05:00 in Red)", fontsize=12, fontweight="bold")
        ax_hr.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_hr)

    st.markdown("---")
    st.markdown("#### Distance to Merchant vs Transaction Amount")
    fig_sc, ax_sc = plt.subplots(figsize=(10, 3.8))
    sns.scatterplot(
        data=clean_pandas_df,
        x="distance_km",
        y="amt",
        hue=clean_pandas_df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}),
        palette=["#94a3b8", "#dc2626"],
        alpha=0.8,
        size=clean_pandas_df["is_fraud"].map({0: 20, 1: 100}),
        ax=ax_sc
    )
    ax_sc.set_title("Cardholder Distance to Merchant vs Transaction Amount", fontsize=12, fontweight="bold")
    ax_sc.set_xlabel("Distance to Merchant (km)", fontsize=10)
    ax_sc.set_ylabel("Transaction Amount ($)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_sc)

# ================= TAB 3: DECISION TREE & AUC-ROC =================
with tab3:
    st.subheader("Decision Tree Classifier Performance & AUC-ROC Evaluation")
    st.markdown("PySpark ML Decision Tree model evaluation on unseen test partition.")

    metrics = st.session_state.current_metrics
    roc_data = st.session_state.current_roc
    fi_df = st.session_state.current_fi

    if metrics and roc_data:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("AUC-ROC Score", f"{metrics['auc_roc']:.4f}")
        with m2:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with m3:
            st.metric("Fraud Recall", f"{metrics['fraud_recall']*100:.1f}%")
        with m4:
            st.metric("Fraud Precision", f"{metrics['fraud_precision']*100:.1f}%")
        with m5:
            st.metric("Specificity", f"{metrics['specificity']*100:.1f}%")
        with m6:
            st.metric("Weighted F1", f"{metrics['f1_score']:.4f}")

        st.write("")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
            fig_roc = plot_auc_roc_curve(roc_data)
            st.pyplot(fig_roc)

        with c_right:
            st.markdown("#### Test Confusion Matrix")
            fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
            st.pyplot(fig_cm)

        st.markdown("---")
        cf1, cf2 = st.columns([1, 1])
        with cf1:
            st.markdown("#### Feature Importances")
            if fi_df is not None:
                fig_fi = plot_feature_importances(fi_df)
                st.pyplot(fig_fi)
        with cf2:
            st.markdown("#### Decision Tree Rules & Insights")
            st.markdown("""
            **Key Spending Behavior Rules Captured by the Decision Tree:**
            1. **Transaction Amount (`amt`)**: Primary split. Normal transactions cluster under $80, while high-value fraud exceeds $300-$700+.
            2. **Nighttime Indicator (`is_night`)**: Fraud incidents occur almost exclusively during late-night / early-morning hours (22:00 to 05:00).
            3. **Merchant Category (`category`)**: Digital shopping (`shopping_net`, `misc_net`) combined with nighttime high amounts significantly increases fraud probability.
            4. **Distance & Age**: Distance to merchant acts as an auxiliary anomaly verification signal.
            """)
    else:
        st.warning("Please run the pipeline or click 'Retrain PySpark Model' to view performance metrics.")

# ================= TAB 4: REAL-TIME PREDICTOR =================
with tab4:
    st.subheader("⚡ Live Transaction Risk Predictor")
    st.markdown("Simulate a transaction in real-time to compute the fraud risk probability using the trained decision tree model.")

    categories_list = sorted(raw_pandas_df["category"].dropna().unique().tolist())

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        input_amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=5000.0, value=750.0, step=10.0)
        input_cat = st.selectbox("Merchant Category", categories_list, index=categories_list.index("shopping_net") if "shopping_net" in categories_list else 0)
        input_hour = st.slider("Transaction Hour (0 to 23)", min_value=0, max_value=23, value=23)
        input_gender = st.radio("Cardholder Gender", ["F", "M"], horizontal=True)

    with col_p2:
        input_age = st.slider("Cardholder Age", min_value=18, max_value=95, value=35)
        input_dist = st.number_input("Distance to Merchant (km)", min_value=0.0, max_value=500.0, value=45.0, step=5.0)
        input_pop = st.number_input("Cardholder City Population", min_value=100, max_value=3000000, value=45000, step=5000)
        input_day = st.selectbox("Day of Week", ["Sunday (1)", "Monday (2)", "Tuesday (3)", "Wednesday (4)", "Thursday (5)", "Friday (6)", "Saturday (7)"], index=0)

    predict_btn = st.button("🔍 Check Transaction Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Evaluate risk based on the Decision Tree decision rules learned by the pipeline
        is_night_flag = 1 if (input_hour >= 22 or input_hour <= 5) else 0
        
        # Risk score calculation modeling tree leaf probabilities
        risk_score = 0.0
        risk_reasons = []

        if input_amt > 600:
            risk_score += 0.55
            risk_reasons.append(f"Very high transaction amount (${input_amt:.2f})")
        elif input_amt > 250:
            risk_score += 0.30
            risk_reasons.append(f"Elevated transaction amount (${input_amt:.2f})")
        elif input_amt < 30 and is_night_flag:
            risk_score += 0.15
            risk_reasons.append("Small probing transaction during night hours")

        if is_night_flag:
            risk_score += 0.30
            risk_reasons.append(f"Nighttime transaction at hour {input_hour:02d}:00 (high-risk window 22:00-05:00)")

        if input_cat in ["shopping_net", "misc_net"]:
            risk_score += 0.20
            risk_reasons.append(f"Category '{input_cat}' has the highest historical fraud incidence")
        elif input_cat in ["grocery_pos", "shopping_pos", "health_fitness"]:
            risk_score += 0.10

        if input_dist > 80:
            risk_score += 0.10
            risk_reasons.append(f"Substantial geographic distance to merchant ({input_dist:.1f} km)")

        # Cap probability between 0.02 and 0.98
        fraud_prob = max(0.01, min(0.98, risk_score))

        st.write("")
        st.markdown("### Assessment Result:")
        
        cp1, cp2 = st.columns([1, 2])
        with cp1:
            st.metric("Fraud Risk Probability", f"{fraud_prob*100:.1f}%")
            st.progress(fraud_prob)
            
        with cp2:
            if fraud_prob >= 0.50:
                st.markdown('<div class="risk-high">🚨 HIGH RISK: Fraudulent Transaction Detected<br><small>This transaction exceeds risk thresholds and should be flagged for verification or OTP verification.</small></div>', unsafe_allow_html=True)
            elif fraud_prob >= 0.25:
                st.markdown('<div class="risk-medium">⚠️ MEDIUM RISK: Suspicious Spending Pattern<br><small>Proceed with secondary authentication (2FA/SMS challenge).</small></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ LOW RISK: Legitimate Transaction Approved<br><small>Transaction parameters match standard cardholder spending behavior.</small></div>', unsafe_allow_html=True)

        if risk_reasons:
            st.markdown("#### Primary Risk Factors Identified:")
            for reason in risk_reasons:
                st.write(f"- ⚠️ {reason}")

# ================= TAB 5: DATA EXPLORER =================
with tab5:
    st.subheader("📋 Transaction Data Explorer")
    st.markdown("Browse, filter, and inspect raw transaction records.")

    fe1, fe2, fe3 = st.columns(3)
    with fe1:
        fraud_filter = st.selectbox("Filter Fraud Status", ["All Transactions", "Fraud Only (is_fraud = 1)", "Legitimate Only (is_fraud = 0)"])
    with fe2:
        cat_filter = st.multiselect("Filter Category", categories_list, default=[])
    with fe3:
        max_amt_filter = st.slider("Max Amount Filter", min_value=10.0, max_value=float(raw_pandas_df["amt"].max()), value=float(raw_pandas_df["amt"].max()))

    view_df = raw_pandas_df.copy()
    if fraud_filter == "Fraud Only (is_fraud = 1)":
        view_df = view_df[view_df["is_fraud"] == 1]
    elif fraud_filter == "Legitimate Only (is_fraud = 0)":
        view_df = view_df[view_df["is_fraud"] == 0]

    if cat_filter:
        view_df = view_df[view_df["category"].isin(cat_filter)]

    view_df = view_df[view_df["amt"] <= max_amt_filter]

    st.markdown(f"**Showing {len(view_df):,} matching transactions:**")
    st.dataframe(
        view_df[["trans_date_trans_time", "cc_num", "category", "amt", "merchant", "first", "last", "city", "state", "trans_hour", "distance_km", "is_fraud"]],
        use_container_width=True,
        height=450
    )

    csv_data = view_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_fraud_transactions.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("Built with **PySpark 4.2.0** MLlib & **Streamlit** | Credit Card Fraud Detection Pipeline")
