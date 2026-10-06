import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WDBC XGBoost AI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model.joblib")

    with open("features.json", "r") as f:
        features = json.load(f)

    with open("labels.json", "r") as f:
        labels = json.load(f)

    return model, features, labels


model, features, labels = load_model()


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():

    try:
        results = pd.read_csv(
            "WDBC_XGBoost_Ablation_Results.csv"
        )
    except:
        results = pd.DataFrame()

    try:
        shap_ranking = pd.read_csv(
            "WDBC_SHAP_Feature_Ranking.csv"
        )
    except:
        shap_ranking = pd.DataFrame()

    return results, shap_ranking


results, shap_ranking = load_results()


# ============================================================
# HEADER
# ============================================================

st.title("🧬 WDBC Breast Cancer Prediction")

st.markdown(
    """
    ### XGBoost + SHAP Explainable AI

    This application uses an **XGBoost classifier** trained on
    the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

    The deployed model uses the **Top-15 SHAP-ranked features**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔬 Model Information")

st.sidebar.success(
    "XGBoost + SHAP"
)

st.sidebar.write(
    f"**Number of Features:** {len(features)}"
)

st.sidebar.write(
    "**Feature Selection:** SHAP Top-15"
)

st.sidebar.write(
    "**Dataset:** WDBC"
)

st.sidebar.divider()

st.sidebar.info(
    "This application is intended for research/demo purposes "
    "and is not a medical diagnostic system."
)


# ============================================================
# FEATURE INPUT
# ============================================================

st.header("🔢 Enter Patient Features")

st.write(
    "Enter the values for the 15 SHAP-selected WDBC features."
)

input_data = {}

# Create columns for cleaner UI

cols = st.columns(3)

for i, feature in enumerate(features):

    with cols[i % 3]:

        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=feature
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Breast Cancer Class",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Create DataFrame in EXACT feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    malignant_probability = probabilities[1]
    benign_probability = probabilities[0]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.header("🧠 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Prediction: MALIGNANT"
        )

    else:

        st.success(
            "✅ Prediction: BENIGN"
        )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Benign Probability",
            f"{benign_probability:.2%}"
        )

    with col2:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability:.2%}"
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame(
        {
            "Class": [
                "Benign",
                "Malignant"
            ],
            "Probability": [
                benign_probability,
                malignant_probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Class")
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header("🔍 SHAP Explanation")

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer(input_df)

        st.subheader(
            "Feature Contribution to Prediction"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    except Exception as e:

        st.warning(
            f"SHAP visualization could not be generated: {e}"
        )


    # ========================================================
    # INPUT VALUES
    # ========================================================

    st.divider()

    st.subheader("📋 Input Feature Values")

    st.dataframe(
        input_df.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )


# ============================================================
# ABLATION STUDY
# ============================================================

st.divider()

st.header("📊 SHAP-Guided Ablation Study")

if not results.empty:

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ACCURACY GRAPH
    # --------------------------------------------------------

    if "Accuracy" in results.columns:

        st.subheader(
            "Accuracy Comparison"
        )

        accuracy_chart = results[
            ["Experiment", "Accuracy"]
        ].copy()

        accuracy_chart = accuracy_chart.set_index(
            "Experiment"
        )

        st.bar_chart(
            accuracy_chart
        )


    # --------------------------------------------------------
    # ROC-AUC GRAPH
    # --------------------------------------------------------

    if "ROC-AUC" in results.columns:

        st.subheader(
            "ROC-AUC Comparison"
        )

        auc_chart = results[
            ["Experiment", "ROC-AUC"]
        ].copy()

        auc_chart = auc_chart.set_index(
            "Experiment"
        )

        st.bar_chart(
            auc_chart
        )

else:

    st.info(
        "Ablation results file not available."
    )


# ============================================================
# SHAP FEATURE RANKING
# ============================================================

st.divider()

st.header("⭐ SHAP Feature Ranking")

if not shap_ranking.empty:

    st.dataframe(
        shap_ranking,
        use_container_width=True,
        hide_index=True
    )

    if (
        "Feature" in shap_ranking.columns
        and
        "Mean_Absolute_SHAP" in shap_ranking.columns
    ):

        st.subheader(
            "Global SHAP Feature Importance"
        )

        ranking_chart = shap_ranking[
            ["Feature", "Mean_Absolute_SHAP"]
        ].head(15)

        ranking_chart = ranking_chart.set_index(
            "Feature"
        )

        st.bar_chart(
            ranking_chart
        )

else:

    st.info(
        "SHAP feature ranking file not available."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "WDBC XGBoost + SHAP | Streamlit Community Cloud | "
    "Research Demonstration"
)
