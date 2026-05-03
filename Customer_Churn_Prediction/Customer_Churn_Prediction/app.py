import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score,
                             recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
)

st.title("📡 AI-Based Customer Churn Prediction System")
st.markdown("*Telecom Customer Relationship Management · Data Mining Project*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Telco CSV", type=["csv"])
    st.markdown("---")
    st.caption("Built with Streamlit · sklearn · Plotly")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    data_original = pd.read_csv(file)
    data = data_original.copy()

    data = data.drop('customerID', axis=1)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    data['AvgCharges'] = data['TotalCharges'] / (data['tenure'] + 1)

    data_encoded = pd.get_dummies(data, drop_first=True)

    X = data_encoded.drop('Churn_Yes', axis=1)
    y = data_encoded['Churn_Yes']

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except ImportError:
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return data_original, data_encoded, X, y, X_train_s, X_test_s, y_train, y_test

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
    }
    results, trained, preds = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model
        preds[name]   = y_pred
        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, y_pred),  4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall':    round(recall_score(y_test, y_pred),    4),
            'F1 Score':  round(f1_score(y_test, y_pred),        4),
        })
    return pd.DataFrame(results), trained, preds

# ── Main flow ─────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈  Upload the **Telco-Customer-Churn.csv** file in the sidebar to begin.", icon="📂")
    st.stop()

with st.spinner("Processing data and training models…"):
    data_original, data_enc, X, y, X_train, X_test, y_train, y_test = load_and_process(uploaded_file)
    results_df, trained_models, all_preds = train_models(X_train, X_test, y_train, y_test)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "🔍 EDA & Visuals", "🤖 Model Benchmarking", "🏆 Best Model & Insights"]
)

# ══ TAB 1 — Dataset Overview ══════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    churn_rate = data_original['Churn'].value_counts(normalize=True)['Yes'] * 100
    c1.metric("Total Customers",  f"{len(data_original):,}")
    c2.metric("Features",         data_original.shape[1] - 1)
    c3.metric("Churn Rate",       f"{churn_rate:.1f}%")
    c4.metric("Missing Values",   int(data_original.isnull().sum().sum()))

    st.markdown("#### First 10 Rows")
    st.dataframe(data_original.head(10), use_container_width=True)

    st.markdown("#### Statistical Summary")
    st.dataframe(data_original.describe(), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Class Distribution")
        dist = data_original['Churn'].value_counts().reset_index()
        dist.columns = ['Churn', 'Count']
        st.dataframe(dist, use_container_width=True)
    with col_right:
        st.markdown("#### Missing Values per Column")
        mv = data_original.isnull().sum().reset_index()
        mv.columns = ['Column', 'Missing']
        mv = mv[mv['Missing'] > 0]
        if mv.empty:
            st.success("No missing values found ✅")
        else:
            st.dataframe(mv, use_container_width=True)

# ══ TAB 2 — EDA ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        churn_counts = data_original['Churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values, names=['No Churn', 'Churn'],
            title="Customer Churn Distribution",
            color_discrete_sequence=['#64b5f6', '#ff8a65'], hole=0.35,
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.07])
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1c2:
        contract_churn = data_original.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn', barmode='group',
            title="Contract Type vs Churn", color_discrete_sequence=['#64b5f6', '#ff8a65'],
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        payment_churn = (
            pd.crosstab(data_original['PaymentMethod'], data_original['Churn'], normalize='index') * 100
        ).reset_index()
        payment_churn.columns = ['PaymentMethod', 'No Churn %', 'Churn %']
        fig_pay = px.bar(
            payment_churn, x='PaymentMethod', y='Churn %',
            title="Churn Rate by Payment Method (%)",
            color='Churn %',color_discrete_sequence=['#64b5f6', '#ff8a65'],
        )
        fig_pay.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_pay, use_container_width=True)

    with r2c2:
       seg = data_original[['tenure', 'MonthlyCharges', 'Churn']].copy()
       seg['Churn'] = seg['Churn'].map({'No': 'No Churn', 'Yes': 'Churn'})

    # 🔹 Create tenure bins (e.g., every 6 months)
    seg['tenure_bin'] = (seg['tenure'] // 6) * 6

    # 🔹 Aggregate (average MonthlyCharges per bin per churn group)
    seg_grouped = seg.groupby(['tenure_bin', 'Churn'], as_index=False)['MonthlyCharges'].mean()

    fig_seg = px.bar(
        seg_grouped,
        x='tenure_bin',
        y='MonthlyCharges',
        color='Churn',
        title="Customer Segmentation: Tenure vs Monthly Charges",
        labels={
            'tenure_bin': 'Tenure',
            'MonthlyCharges': 'Monthly Charges'
        },
        color_discrete_sequence=['#64b5f6', '#ff8a65'],
        opacity=0.5,
    )

    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("#### Numeric Feature Distributions by Churn")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=num_cols)
    for i, col in enumerate(num_cols, 1):
        for churn_val, label, color in [(0, 'No Churn', '#64b5f6'), (1, 'Churn', '#ff8a65')]:
            subset = data_enc[data_enc['Churn_Yes'] == churn_val][col].dropna()
            fig_dist.add_trace(
                go.Histogram(x=subset, name=label, marker_color=color,
                             opacity=0.65, showlegend=(i == 1), nbinsx=30),
                row=1, col=i,
            )
    fig_dist.update_layout(barmode='overlay', title_text='Feature Distributions', height=350)
    st.plotly_chart(fig_dist, use_container_width=True)


# ══ TAB 3 — Model Benchmarking ════════════════════════════════════════════════
with tab3:
    st.subheader("Model Training & Benchmarking")

    st.markdown("#### Model Comparison Table")

    def highlight_max_green(s):
        is_max = s == s.max()
        return ['background-color: #A9A9A9; color: #ffffff' if v else 'color: #ffffff' for v in is_max]

    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    display_df = results_df.copy()
    for c in metric_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}")

    styled = (
        display_df.style
        .apply(highlight_max_green, subset=metric_cols)
        .set_properties(**{'color': '#ffffff'})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig_bar = px.bar(
        melted, x='Metric', y='Score', color='Model', barmode='group',
        title="Model Comparison — All Metrics",
        color_discrete_sequence=px.colors.qualitative.Set2, text_auto='.3f',
    )
    fig_bar.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Radar Chart — Model Profiles")
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig_radar = go.Figure()
    for (_, row), color in zip(results_df.iterrows(), ['#636EFA', '#EF553B', '#00CC96']):
        vals = [row[m] for m in metrics_list] + [row[metrics_list[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metrics_list + [metrics_list[0]],
            fill='toself', name=row['Model'], line_color=color, opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Performance Radar",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### Confusion Matrices")
    cols_cm = st.columns(3)
    for idx, (model_name, y_pred) in enumerate(all_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
            color_continuous_scale='Blues', title=model_name,
        )
        fig_cm.update_layout(height=320, coloraxis_showscale=False)
        cols_cm[idx].plotly_chart(fig_cm, use_container_width=True)

# ══ TAB 4 — Best Model & Insights ════════════════════════════════════════════
with tab4:
    best_name  = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
    best_row   = results_df[results_df['Model'] == best_name].iloc[0]
    best_model = trained_models[best_name]
    best_pred  = all_preds[best_name]

    st.subheader(f"🏆 Best Model: {best_name}")
    st.caption("Selected based on highest F1 Score — appropriate for imbalanced churn classification.")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{best_row['Accuracy']:.2%}")
    mc2.metric("Precision", f"{best_row['Precision']:.2%}")
    mc3.metric("Recall",    f"{best_row['Recall']:.2%}")
    mc4.metric("F1 Score",  f"{best_row['F1 Score']:.2%}")

    st.markdown("#### Detailed Classification Report")
    report = classification_report(y_test, best_pred,
                                   target_names=['No Churn', 'Churn'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    if best_name in ["Random Forest", "Decision Tree"] and hasattr(best_model, 'feature_importances_'):
        st.markdown("#### 🔑 Top 15 Feature Importances")
        imp_df = pd.DataFrame({
            'Feature':    X.columns,
            'Importance': best_model.feature_importances_,
        }).sort_values('Importance', ascending=False).head(15)
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title=f"Top 15 Features — {best_name}",
            color='Importance', color_continuous_scale='Tealgrn',
        )
        fig_imp.update_layout(yaxis={'autorange': 'reversed'}, height=480)
        st.plotly_chart(fig_imp, use_container_width=True)
    elif best_name == "Logistic Regression":
        st.markdown("#### 🔑 Top 15 Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature':     X.columns,
            'Coefficient': np.abs(best_model.coef_[0]),
        }).sort_values('Coefficient', ascending=False).head(15)
        fig_coef = px.bar(
            coef_df, x='Coefficient', y='Feature', orientation='h',
            title="Top 15 Features by |Coefficient| — Logistic Regression",
            color='Coefficient', color_continuous_scale='Oryel',
        )
        fig_coef.update_layout(yaxis={'autorange': 'reversed'}, height=480)
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")
    st.subheader("💼 Business Insights & Conclusions")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""
**Key Findings**
- ~26 % of customers churned — significant revenue risk
- Month-to-month contract holders churn at the highest rate
- Electronic check users show the highest churn propensity
- Short-tenure customers (< 12 months) are most at risk
- Higher monthly charges correlate with increased churn
        """)
    with i2:
        st.markdown("""
**Business Value**
- Identify high-risk customers before they leave
- Target retention campaigns at short-tenure, high-charge segments
- Offer long-term contract incentives to at-risk cohorts
- Prioritise support for electronic-check payment customers
- Data-driven decisions to reduce churn and improve LTV
        """)
    st.markdown("""
---
**Future Improvements** · Hyperparameter tuning · XGBoost / LightGBM benchmarking · SHAP explainability · Real-time prediction API
    """)
