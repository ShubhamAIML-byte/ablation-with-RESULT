import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WDBC XGBoost + SHAP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        text-align: center;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .small-text {
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model.joblib")

    with open("features.json", "r") as f:
        features = json.load(f)

    return model, features


model, features = load_model()


# ============================================================
# IMPORTANT CLASS MAPPING
# ============================================================
# Based on sklearn load_breast_cancer():
#
# 0 = malignant
# 1 = benign
#
# This matches your uploaded notebook.
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

    try:
        results = pd.read_csv(
            "WDBC_XGBoost_Ablation_Results.csv"
        )
    except Exception:
        results = pd.DataFrame()

    try:
        shap_ranking = pd.read_csv(
            "WDBC_SHAP_Feature_Ranking.csv"
        )
    except Exception:
        shap_ranking = pd.DataFrame()

    return results, shap_ranking


results, shap_ranking = load_results()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🧬 WDBC AI")

st.sidebar.markdown(
    "### Navigation"
)

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


# ============================================================
# SIDEBAR MODEL INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 🔬 Model Information"
)

st.sidebar.success(
    "XGBoost + SHAP"
)

st.sidebar.write(
    f"**Features:** {len(features)}"
)

st.sidebar.write(
    "**Feature Selection:** SHAP Top-15"
)

st.sidebar.write(
    "**Dataset:** WDBC"
)

st.sidebar.write(
    "**Model:** XGBoost"
)

st.sidebar.divider()

st.sidebar.warning(
    "Research/demo purposes only. "
    "This application is not a medical diagnostic system."
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.markdown(
        '<div class="main-title">🧬 WDBC Breast Cancer Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'XGBoost classification with SHAP-guided feature selection'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter the 15 SHAP-selected WDBC feature values below "
        "and click Predict."
    )

    st.markdown(
        "### 🔢 Patient Feature Input"
    )

    # --------------------------------------------------------
    # FEATURE INPUTS
    # --------------------------------------------------------

    input_data = {}

    cols = st.columns(3)

    for i, feature in enumerate(features):

        with cols[i % 3]:

            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"input_{feature}"
            )


    st.divider()


    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:

        predict_button = st.button(
            "🔍 Predict",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear_button = st.button(
            "🔄 Clear Inputs",
            use_container_width=True
        )

    with col3:

        st.button(
            "📋 15 SHAP Features",
            use_container_width=True
        )


    # --------------------------------------------------------
    # CLEAR INPUTS
    # --------------------------------------------------------

    if clear_button:

        st.rerun()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )

        try:

            prediction = int(
                model.predict(input_df)[0]
            )

            probabilities = model.predict_proba(
                input_df
            )[0]

            malignant_probability = float(
                probabilities[0]
            )

            benign_probability = float(
                probabilities[1]
            )

            predicted_class = CLASS_NAMES[
                prediction
            ]


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )

            if prediction == 0:

                st.error(
                    "⚠️ Predicted Class: MALIGNANT"
                )

            else:

                st.success(
                    "✅ Predicted Class: BENIGN"
                )


            # ------------------------------------------------
            # PROBABILITY METRICS
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Prediction",
                    predicted_class
                )

            with c2:

                st.metric(
                    "Malignant Probability",
                    f"{malignant_probability:.2%}"
                )

            with c3:

                st.metric(
                    "Benign Probability",
                    f"{benign_probability:.2%}"
                )


            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Probability"
            )

            probability_df = pd.DataFrame(
                {
                    "Class": [
                        "Malignant",
                        "Benign"
                    ],
                    "Probability": [
                        malignant_probability,
                        benign_probability
                    ]
                }
            )

            st.bar_chart(
                probability_df.set_index("Class")
            )


            # ------------------------------------------------
            # SHAP LOCAL EXPLANATION
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔍 Why did the model make this prediction?"
            )

            try:

                explainer = shap.TreeExplainer(
                    model
                )

                shap_explanation = explainer(
                    input_df
                )

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                shap.plots.waterfall(
                    shap_explanation[0],
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
                    "SHAP waterfall plot could not be generated."
                )

                st.code(
                    str(e)
                )


            # ------------------------------------------------
            # INPUT TABLE
            # ------------------------------------------------

            with st.expander(
                "📋 View Input Feature Values"
            ):

                st.dataframe(
                    input_df.T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# PAGE 2 — ABLATION STUDY
# ============================================================

elif page == "📊 Ablation Study":

    st.title(
        "📊 SHAP-Guided Ablation Study"
    )

    st.markdown(
        """
        Comparison of the baseline model, feature-group ablations,
        and SHAP-guided feature selection experiments.
        """
    )

    if results.empty:

        st.warning(
            "WDBC_XGBoost_Ablation_Results.csv was not found."
        )

    else:

        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Experiment Results"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        if "Accuracy" in results.columns:

            st.subheader(
                "🎯 Accuracy Comparison"
            )

            accuracy_data = results[
                ["Experiment", "Accuracy"]
            ].copy()

            accuracy_data = accuracy_data.set_index(
                "Experiment"
            )

            st.bar_chart(
                accuracy_data
            )


        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if "ROC-AUC" in results.columns:

            st.subheader(
                "📈 ROC-AUC Comparison"
            )

            auc_data = results[
                ["Experiment", "ROC-AUC"]
            ].copy()

            auc_data = auc_data.set_index(
                "Experiment"
            )

            st.bar_chart(
                auc_data
            )


        # ----------------------------------------------------
        # F1 SCORE
        # ----------------------------------------------------

        if "F1" in results.columns:

            st.subheader(
                "⚖️ F1 Score Comparison"
            )

            f1_data = results[
                ["Experiment", "F1"]
            ].copy()

            f1_data = f1_data.set_index(
                "Experiment"
            )

            st.bar_chart(
                f1_data
            )


        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        if "Recall" in results.columns:

            st.subheader(
                "🎯 Recall Comparison"
            )

            recall_data = results[
                ["Experiment", "Recall"]
            ].copy()

            recall_data = recall_data.set_index(
                "Experiment"
            )

            st.bar_chart(
                recall_data
            )


# ============================================================
# PAGE 3 — SHAP EXPLAINABILITY
# ============================================================

elif page == "🔍 SHAP Explainability":

    st.title(
        "🔍 SHAP Explainability"
    )

    st.markdown(
        """
        SHAP identifies which features contribute most strongly
        to the XGBoost predictions.
        """
    )

    if shap_ranking.empty:

        st.warning(
            "WDBC_SHAP_Feature_Ranking.csv was not found."
        )

    else:

        # ----------------------------------------------------
        # SHAP RANKING TABLE
        # ----------------------------------------------------

        st.subheader(
            "⭐ Global SHAP Feature Ranking"
        )

        st.dataframe(
            shap_ranking,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # TOP 15
        # ----------------------------------------------------

        if (
            "Feature" in shap_ranking.columns
            and
            "Mean_Absolute_SHAP" in shap_ranking.columns
        ):

            st.subheader(
                "🔥 Top 15 SHAP Features"
            )

            top15 = shap_ranking.head(15).copy()

            chart_data = top15[
                [
                    "Feature",
                    "Mean_Absolute_SHAP"
                ]
            ].set_index(
                "Feature"
            )

            st.bar_chart(
                chart_data
            )


        # ----------------------------------------------------
        # TOP 10
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Top 10 SHAP Features"
        ):

            st.dataframe(
                shap_ranking.head(10),
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # TOP 15 FEATURE NAMES
        # ----------------------------------------------------

        with st.expander(
            "🧬 View Features Used by Deployed Model"
        ):

            for i, feature in enumerate(
                features,
                start=1
            ):

                st.write(
                    f"**{i}.** {feature}"
                )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance"
    )

    if results.empty:

        st.warning(
            "Model results are unavailable."
        )

    else:

        # ----------------------------------------------------
        # BEST RESULTS
        # ----------------------------------------------------

        st.subheader(
            "🏆 Performance Summary"
        )

        best_accuracy = results[
            "Accuracy"
        ].max()

        best_auc = results[
            "ROC-AUC"
        ].max()

        best_f1 = results[
            "F1"
        ].max()

        best_recall = results[
            "Recall"
        ].max()


        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Best Accuracy",
                f"{best_accuracy:.4f}"
            )

        with c2:

            st.metric(
                "Best ROC-AUC",
                f"{best_auc:.4f}"
            )

        with c3:

            st.metric(
                "Best F1",
                f"{best_f1:.4f}"
            )

        with c4:

            st.metric(
                "Best Recall",
                f"{best_recall:.4f}"
            )


        st.divider()


        # ----------------------------------------------------
        # FULL PERFORMANCE TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Complete Performance Table"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # METRIC COMPARISON
        # ----------------------------------------------------

        metric_options = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Specificity"
        ]

        available_metrics = [
            m for m in metric_options
            if m in results.columns
        ]

        selected_metric = st.selectbox(
            "Select metric to compare",
            available_metrics
        )

        metric_chart = results[
            [
                "Experiment",
                selected_metric
            ]
        ].set_index(
            "Experiment"
        )

        st.bar_chart(
            metric_chart
        )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About This Application"
    )

    st.markdown(
        """
        ## 🧬 WDBC XGBoost + SHAP

        This application demonstrates an explainable machine
        learning workflow for the Wisconsin Diagnostic Breast
        Cancer dataset.

        ### 🔬 Methodology

        **Dataset**

        Wisconsin Diagnostic Breast Cancer (WDBC)

        **Machine Learning Model**

        XGBoost Classifier

        **Feature Selection**

        SHAP-guided feature ranking

        **Deployed Feature Set**

        Top 15 SHAP-ranked features

        **Explainability**

        SHAP

        ### 🧪 Ablation Study

        The research workflow compares:

        - Baseline using all 30 features
        - Without Mean features
        - Without Standard Error features
        - Without Worst features
        - SHAP Top-20 features
        - SHAP Top-15 features
        - SHAP Top-10 features

        ### 🎯 Purpose

        The application is designed for:

        - Research demonstration
        - Explainable AI
        - Model evaluation
        - SHAP-based analysis
        - Feature ablation analysis
        - Human-in-the-loop exploration

        ### ⚠️ Disclaimer

        This application is for research and educational
        demonstration purposes only.

        It is **not a medical diagnostic system** and should
        not be used to make clinical decisions.
        """
    )

    st.divider()

    st.subheader(
        "📦 Deployment Information"
    )

    st.write(
        "**Deployment:** Streamlit Community Cloud"
    )

    st.write(
        "**Model:** XGBoost"
    )

    st.write(
        "**Explainability:** SHAP"
    )

    st.write(
        f"**Deployed Features:** {len(features)}"
    )
