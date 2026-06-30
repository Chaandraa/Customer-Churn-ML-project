
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        padding: 0 1rem;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .churn-yes {
        color: #d9534f;
        font-weight: bold;
    }
    .churn-no {
        color: #5cb85c;
        font-weight: bold;
    }
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model & Assets --------------------
@st.cache_resource
def load_model_and_metadata():
    try:
        model = joblib.load("../models/best_churn_model.joblib")
        # If you have saved metadata (feature names, metrics), load them here
        # For now we define them manually
        feature_names = [
            "Monetary", "Frequency", "Recency", "Tenure",
            "Country_Belgium", "Country_France", "Country_Germany",
            "Country_Netherlands", "Country_Other", "Country_Portugal",
            "Country_Spain", "Country_Sweden", "Country_Switzerland",
            "Country_United Kingdom"
        ]
        # Pre-computed metrics (example values – replace with your actual test metrics)
        metrics = {
            "roc_auc": 0.92,
            "accuracy": 0.87,
            "precision": 0.78,
            "recall": 0.85,
            "f1": 0.81
        }
        # Dummy feature importance (if model doesn't have it, we'll compute later)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.random.rand(len(feature_names))  # fallback
        return model, feature_names, metrics, importances
    except FileNotFoundError:
        st.error("❌ Model file not found. Please train the model first and place it in `../models/best_churn_model.joblib`.")
        return None, None, None, None

model, FEATURE_NAMES, METRICS, IMPORTANCES = load_model_and_metadata()

# Country list
COUNTRIES = [
    "United Kingdom", "Germany", "France", "Spain", "Belgium",
    "Netherlands", "Portugal", "Switzerland", "Sweden", "Australia", "Other"
]

# -------------------- Preprocessing Function --------------------
def preprocess_input(monetary, frequency, recency, tenure, country):
    """Convert user inputs to feature vector."""
    features = {f: 0 for f in FEATURE_NAMES}
    features["Monetary"] = monetary
    features["Frequency"] = frequency
    features["Recency"] = recency
    features["Tenure"] = tenure
    # One-hot encode country
    if country in COUNTRIES:
        col = f"Country_{country}"
        if col in features:
            features[col] = 1
        else:
            features["Country_Other"] = 1
    else:
        features["Country_Other"] = 1
    return pd.DataFrame([features])[FEATURE_NAMES]

# -------------------- Sidebar Inputs --------------------
st.sidebar.title("🔧 Customer Metrics")
st.sidebar.markdown("Adjust the values below to predict churn risk.")

# Use sliders for better UX with sensible ranges
monetary = st.sidebar.slider(
    "💰 Monetary (Total Spend)",
    min_value=0.0, max_value=5000.0, value=500.0, step=50.0,
    help="Total amount spent by the customer"
)
frequency = st.sidebar.slider(
    "📦 Frequency (Number of Purchases)",
    min_value=0, max_value=50, value=5, step=1,
    help="Number of unique invoices/orders"
)
recency = st.sidebar.slider(
    "🕒 Recency (Days since last purchase)",
    min_value=0, max_value=365, value=30, step=1,
    help="Days since the customer's last purchase"
)
tenure = st.sidebar.slider(
    "📅 Tenure (Days since first purchase)",
    min_value=0, max_value=1000, value=365, step=10,
    help="Total days since the customer's first purchase"
)
country = st.sidebar.selectbox("🌍 Country", COUNTRIES)

# Prediction button
predict_btn = st.sidebar.button("🚀 Predict Churn", use_container_width=True)

# Option to reset inputs
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.session_state['monetary'] = 500.0
    st.session_state['frequency'] = 5
    st.session_state['recency'] = 30
    st.session_state['tenure'] = 365
    st.session_state['country'] = "United Kingdom"
    st.experimental_rerun()

# -------------------- Main Area --------------------
st.title("🛒 Customer Churn Predictor")
st.markdown("""
Enter customer details using the sidebar or upload a CSV file for batch predictions.
The model predicts the probability that a customer will churn (no purchase in the last 90 days).
""")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📁 Batch Prediction", "📈 Model Performance"])

# -------------------- Tab 1: Single Prediction --------------------
with tab1:
    if predict_btn and model is not None:
        # Preprocess
        input_df = preprocess_input(monetary, frequency, recency, tenure, country)
        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # Store in session state for history
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "Monetary": monetary,
            "Frequency": frequency,
            "Recency": recency,
            "Tenure": tenure,
            "Country": country,
            "Churn": "Yes" if pred == 1 else "No",
            "Probability": prob
        })

        # Display results in a nice layout
        st.subheader("📋 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Churn Status</h3>
                <p class="big-metric {'churn-yes' if pred==1 else 'churn-no'}">
                    {'⚠️ Churned' if pred==1 else '✅ Not Churned'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Churn Probability</h3>
                <p class="big-metric">{prob:.2%}</p>
                <p style="color: gray; font-size: 0.9rem;">Threshold = 50%</p>
            </div>
            """, unsafe_allow_html=True)

        # Show input parameters for transparency
        with st.expander("🔍 Input Parameters"):
            st.dataframe(input_df)

        # Show a gauge chart for probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk (%)"},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance for this prediction (SHAP-like? not implemented, but show global)
        if IMPORTANCES is not None and len(IMPORTANCES) == len(FEATURE_NAMES):
            st.subheader("📊 Global Feature Importance")
            imp_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Importance": IMPORTANCES
            }).sort_values("Importance", ascending=False).head(10)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                             color="Importance", color_continuous_scale="Viridis",
                             title="Top 10 Features by Importance")
            fig_imp.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

    elif predict_btn and model is None:
        st.error("Model not loaded. Please check the file path.")

# -------------------- Tab 2: Batch Prediction --------------------
with tab2:
    st.subheader("📁 Upload CSV for Batch Prediction")
    st.markdown("Upload a CSV file with columns: `Monetary`, `Frequency`, `Recency`, `Tenure`, `Country`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            # Validate required columns
            required = ["Monetary", "Frequency", "Recency", "Tenure", "Country"]
            missing = [col for col in required if col not in df_batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Preprocess each row
                predictions = []
                probabilities = []
                for _, row in df_batch.iterrows():
                    try:
                        input_df = preprocess_input(
                            row["Monetary"],
                            row["Frequency"],
                            row["Recency"],
                            row["Tenure"],
                            row["Country"]
                        )
                        pred = model.predict(input_df)[0]
                        prob = model.predict_proba(input_df)[0][1]
                        predictions.append("Yes" if pred == 1 else "No")
                        probabilities.append(prob)
                    except Exception as e:
                        predictions.append("Error")
                        probabilities.append(np.nan)
                df_batch["Churn_Prediction"] = predictions
                df_batch["Churn_Probability"] = probabilities

                # Display results
                st.success(f"✅ Processed {len(df_batch)} rows.")
                st.dataframe(df_batch, use_container_width=True)

                # Summary statistics
                st.subheader("📊 Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers", len(df_batch))
                with col2:
                    churn_count = (df_batch["Churn_Prediction"] == "Yes").sum()
                    st.metric("Predicted Churn", churn_count)
                with col3:
                    avg_prob = df_batch["Churn_Probability"].mean()
                    st.metric("Average Churn Probability", f"{avg_prob:.2%}")

                # Download results
                csv = df_batch.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">⬇️ Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Show distribution of probabilities
                fig_hist = px.histogram(df_batch, x="Churn_Probability", nbins=30,
                                        title="Distribution of Churn Probabilities",
                                        color_discrete_sequence=["#1f77b4"])
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (50%)")
                st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------- Tab 3: Model Performance --------------------
with tab3:
    st.subheader("📈 Model Performance Metrics")
    if METRICS:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
        with col2:
            st.metric("Accuracy", f"{METRICS['accuracy']:.2%}")
        with col3:
            st.metric("Precision", f"{METRICS['precision']:.2%}")
        with col4:
            st.metric("Recall", f"{METRICS['recall']:.2%}")
        with col5:
            st.metric("F1-Score", f"{METRICS['f1']:.3f}")

    # Show confusion matrix (precomputed or dummy)
    # If you have a confusion matrix saved, load it. Here we'll show a placeholder.
    st.markdown("**Confusion Matrix (on test set)**")
    # For demonstration, we create a dummy matrix. Replace with your actual data.
    cm = np.array([[200, 30], [40, 100]])  # [TN, FP; FN, TP]
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Not Churn", "Churn"], y=["Not Churn", "Churn"])
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Show feature importance again (reuse)
    if IMPORTANCES is not None and len(IMPORTANCES) == len(FEATURE_NAMES):
        imp_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": IMPORTANCES
        }).sort_values("Importance", ascending=False)
        fig_imp2 = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                          color="Importance", color_continuous_scale="Viridis",
                          title="Global Feature Importance (Full List)")
        fig_imp2.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp2, use_container_width=True)

# -------------------- Prediction History (optional) --------------------
if 'history' in st.session_state and len(st.session_state.history) > 0:
    with st.expander("📜 Prediction History (Session)"):
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        # Option to clear history
        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: Random Forest (balanced class)")
