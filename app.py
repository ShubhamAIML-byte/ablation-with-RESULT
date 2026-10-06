import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WDBC AI Cancer Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #f7f9fc;
}

[data-testid="stSidebar"] h1 {
    color: #1f4e79;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    text-align: center;
}

.big-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1f4e79;
}

.subtitle {
    font-size: 1.1rem;
    color: #555;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="big-title">🧬 WDBC AI Cancer Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">XGBoost + SHAP Explainable Breast Cancer Classification</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🧬 WDBC AI")

page = st.sidebar.radio(
    "Select a section",
    [
        "🏠 Prediction",
        "📊 Ablation Study",
        "🔍 SHAP Explainability",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dataset:** Wisconsin Diagnostic Breast Cancer

    **Model:** XGBoost

    **Explainability:** SHAP

    **Task:** Binary Classification
    """
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model.joblib")

    with open("features.json", "r") as f:
        features = json.load(f)

    return model, features


model, selected_features = load_model()


# ============================================================
# CLASS LABELS
# IMPORTANT:
# sklearn breast cancer dataset:
# 0 = malignant
# 1 = benign
# ============================================================

CLASS_NAMES = {
    0: "Malignant",
    1: "Benign"
}


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():

    results = pd.read_csv(
        "WDBC_XGBoost_Ablation_Results.csv"
    )

    shap_ranking = pd.read_csv(
        "WDBC_SHAP_Feature_Ranking.csv"
    )

    return results, shap_ranking


results_df, shap_ranking_df = load_results()


# ============================================================
# DYNAMIC SHAP BASELINE MODEL
# ============================================================

@st.cache_resource
def create_shap_baseline():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        data.target,
        name="target"
    )

    X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    baseline_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    baseline_model.fit(
        X_train_shap,
        y_train_shap
    )

    return (
        baseline_model,
        X_train_shap,
        X_test_shap,
        y_train_shap,
        y_test_shap
    )


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

@st.cache_data
def calculate_shap_values(model, X_data):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_data)

    return shap_values


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🏠 Prediction":

    st.header("🏠 Breast Cancer Prediction")

    st.write(
        "Enter the selected WDBC features below and use the trained "
        "XGBoost model to predict the cancer class."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    input_values = {}

    cols = st.columns(3)

    for i, feature in enumerate(selected_features):

        with cols[i % 3]:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

    st.write("")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔬 Predict Cancer Class",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(
            input_df
        )[0]

        predicted_class = CLASS_NAMES[prediction]

        st.markdown("---")

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if predicted_class == "Benign":

            st.success(
                f"### Prediction: {predicted_class}"
            )

        else:

            st.error(
                f"### Prediction: {predicted_class}"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Malignant Probability",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Benign Probability",
                f"{probabilities[1] * 100:.2f}%"
            )

        st.write("### Prediction Probability")

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "Malignant",
                    "Benign"
                ],
                "Probability": [
                    probabilities[0],
                    probabilities[1]
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        st.write("### Input Features")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SHAP EXPLANATION FOR PREDICTION
        # ----------------------------------------------------

        st.write("### 🔍 Prediction Explanation")

        try:

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                input_df
            )

            if isinstance(shap_values, list):

                shap_values_single = shap_values[
                    prediction
                ]

            else:

                shap_values_single = shap_values[0]

            fig = plt.figure(figsize=(10, 6))

            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_single,
                    base_values=explainer.expected_value,
                    data=input_df.iloc[0],
                    feature_names=input_df.columns
                ),
                max_display=15,
                show=False
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

        except Exception as e:

            st.warning(
                f"SHAP prediction explanation could not be generated: {e}"
            )


# ============================================================
# ABLATION STUDY
# ============================================================

elif page == "📊 Ablation Study":

    st.header("📊 XGBoost Ablation Study")

    st.write(
        "Comparison of different feature-selection experiments "
        "performed on the WDBC dataset."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.subheader("Ablation Results")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    st.subheader("Accuracy Comparison")

    accuracy_df = results_df[
        ["Experiment", "Accuracy"]
    ].set_index("Experiment")

    st.bar_chart(
        accuracy_df
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    st.subheader("ROC-AUC Comparison")

    auc_df = results_df[
        ["Experiment", "ROC-AUC"]
    ].set_index("Experiment")

    st.bar_chart(
        auc_df
    )

    # --------------------------------------------------------
    # F1 SCORE
    # --------------------------------------------------------

    st.subheader("F1 Score Comparison")

    f1_df = results_df[
        ["Experiment", "F1"]
    ].set_index("Experiment")

    st.bar_chart(
        f1_df
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    st.subheader("Recall Comparison")

    recall_df = results_df[
        ["Experiment", "Recall"]
    ].set_index("Experiment")

    st.bar_chart(
        recall_df
    )


# ============================================================
# SHAP EXPLAINABILITY PAGE
# ============================================================

elif page == "🔍 SHAP Explainability":

    st.header("🔍 Dynamic SHAP Explainability")

    st.write(
        "The SHAP summary plots below are generated dynamically "
        "from the XGBoost model rather than using a static image."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CREATE BASELINE MODEL
    # --------------------------------------------------------

    with st.spinner(
        "Preparing XGBoost model and SHAP values..."
    ):

        (
            shap_model,
            X_train_shap,
            X_test_shap,
            y_train_shap,
            y_test_shap
        ) = create_shap_baseline()

        shap_values_dynamic = calculate_shap_values(
            shap_model,
            X_train_shap
        )

    st.success(
        "Dynamic SHAP model is ready."
    )

    # --------------------------------------------------------
    # FEATURE NUMBER SELECTOR
    # --------------------------------------------------------

    max_features = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=30,
        value=20,
        step=5
    )

    # --------------------------------------------------------
    # GENERATE SUMMARY PLOT
    # --------------------------------------------------------

    generate_plot = st.button(
        "🔄 Generate SHAP Summary Plot",
        type="primary",
        use_container_width=True
    )

    if generate_plot:

        with st.spinner(
            "Generating SHAP Summary Plot..."
        ):

            fig = plt.figure(
                figsize=(10, 8)
            )

            shap.summary_plot(
                shap_values_dynamic,
                X_train_shap,
                max_display=max_features,
                show=False
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

    # --------------------------------------------------------
    # DYNAMIC SHAP BAR PLOT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Dynamic SHAP Feature Importance"
    )

    generate_bar = st.button(
        "📊 Generate SHAP Bar Plot",
        use_container_width=True
    )

    if generate_bar:

        with st.spinner(
            "Generating SHAP Feature Importance..."
        ):

            fig = plt.figure(
                figsize=(10, 8)
            )

            shap.summary_plot(
                shap_values_dynamic,
                X_train_shap,
                plot_type="bar",
                max_display=max_features,
                show=False
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

    # --------------------------------------------------------
    # SHAP FEATURE RANKING FROM NOTEBOOK
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📌 SHAP Feature Ranking"
    )

    st.dataframe(
        shap_ranking_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TOP SHAP FEATURES
    # --------------------------------------------------------

    if len(shap_ranking_df) > 0:

        st.subheader(
            "Top SHAP Features"
        )

        top_n = st.slider(
            "Number of top features",
            min_value=5,
            max_value=min(20, len(shap_ranking_df)),
            value=min(10, len(shap_ranking_df))
        )

        top_features = shap_ranking_df.head(
            top_n
        )

        feature_column = top_features.columns[0]

        value_column = top_features.columns[1]

        chart_df = top_features[
            [feature_column, value_column]
        ].set_index(
            feature_column
        )

        st.bar_chart(
            chart_df
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Performance metrics obtained from the WDBC XGBoost "
        "ablation experiments."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # BEST EXPERIMENT
    # --------------------------------------------------------

    best_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{best_row['Accuracy']:.4f}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{best_row['Precision']:.4f}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{best_row['Recall']:.4f}"
        )

    with col4:

        st.metric(
            "F1",
            f"{best_row['F1']:.4f}"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{best_row['ROC-AUC']:.4f}"
        )

    st.markdown("---")

    st.subheader(
        f"Best Accuracy Experiment: {best_row['Experiment']}"
    )

    st.dataframe(
        pd.DataFrame(
            [best_row]
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # METRIC SELECTOR
    # --------------------------------------------------------

    metric = st.selectbox(
        "Select performance metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Specificity"
        ]
    )

    metric_df = results_df[
        ["Experiment", metric]
    ].set_index(
        "Experiment"
    )

    st.bar_chart(
        metric_df
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About the Project")

    st.markdown("""
    ### WDBC XGBoost Explainable AI

    This application demonstrates a breast cancer classification
    pipeline using the Wisconsin Diagnostic Breast Cancer dataset.

    **Machine Learning Model**

    - XGBoost Classifier
    - Binary classification
    - Malignant vs Benign

    **Explainable AI**

    - SHAP feature importance
    - SHAP summary plot
    - SHAP bar plot
    - SHAP waterfall explanation

    **Research Components**

    - Baseline experiment
    - Feature ablation
    - SHAP-guided feature selection
    - Top-20 feature selection
    - Top-15 feature selection
    - Top-10 feature selection

    **Evaluation Metrics**

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Specificity

    **Deployment**

    - Streamlit
    - GitHub
    - Streamlit Community Cloud
    """)

    st.markdown("---")

    st.info(
        "This application is intended for academic and research demonstration purposes."
    )
