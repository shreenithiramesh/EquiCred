
import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ZYVANTIS | Credit Fairness Auditor",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #F4F7FB;}
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #DCE4EF;
}
</style>
""", unsafe_allow_html=True)

st.title("🔎 ZYVANTIS")
st.subheader("Intelligent Credit Risk & Fairness Auditor")
st.caption(
    "Offline-capable • Automated analytics • "
    "Model evaluation • Group disparity review"
)


# ---------------- LOAD DATA ----------------

# -------- LOAD EXCEL OR CSV DATA --------

uploaded = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["xlsx", "xls", "csv"]
)

# Optional local Excel file
default_file = "credit_card_clients_cleaned.xlsx"

try:
    if uploaded is not None:
        filename = uploaded.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded)
            selected_sheet = "CSV file"
        else:
            excel_file = pd.ExcelFile(uploaded)
            sheet_names = excel_file.sheet_names

            selected_sheet = st.sidebar.selectbox(
                "Select Excel sheet",
                sheet_names
            )

            df = pd.read_excel(
                excel_file,
                sheet_name=selected_sheet
            )

    elif os.path.exists(default_file):
        excel_file = pd.ExcelFile(default_file)

        selected_sheet = st.sidebar.selectbox(
            "Select local Excel sheet",
            excel_file.sheet_names
        )

        df = pd.read_excel(
            excel_file,
            sheet_name=selected_sheet
        )

    else:
        st.info(
            "Upload your Excel file to start the audit."
        )
        st.stop()

    df.columns = df.columns.astype(str).str.strip()

    st.sidebar.success("Dataset loaded successfully")
    st.sidebar.write(f"Selected sheet: {selected_sheet}")
    st.sidebar.write(f"Rows: {len(df):,}")
    st.sidebar.write(f"Columns: {len(df.columns)}")

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()



# ---------------- DATA QUALITY ----------------
st.header("1. Automatic Data Quality Audit")

missing_total = int(df.isna().sum().sum())
duplicate_total = int(df.duplicated().sum())

a, b, c, d = st.columns(4)
a.metric("Total records", f"{len(df):,}")
b.metric("Total columns", len(df.columns))
c.metric("Missing cells", f"{missing_total:,}")
d.metric("Duplicate rows", f"{duplicate_total:,}")

with st.expander("View dataset preview and column types"):
    st.dataframe(df.head(100), use_container_width=True)
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Missing values": df.isna().sum().values,
            "Unique values": df.nunique().values
        }),
        use_container_width=True
    )

if duplicate_total:
    st.warning(
        f"{duplicate_total} duplicate rows detected. "
        "They are flagged, not silently removed."
    )

if missing_total:
    st.info(
        "Missing values found. Numeric feature values will be "
        "imputed during model training."
    )


# ---------------- CONFIGURATION ----------------
st.header("2. Configure the Audit")

default_target = (
    "default_next_month"
    if "default_next_month" in df.columns
    else df.columns[-1]
)

target = st.selectbox(
    "Select actual outcome / target column",
    options=list(df.columns),
    index=list(df.columns).index(default_target)
)

# Do not use identifiers or sensitive group fields as model inputs.
excluded_names = {
    "customer_id", "id", "sex_code", "education_code",
    "marriage_code", "sex_group", "education_group",
    "marriage_group"
}

excluded = [
    col for col in df.columns
    if col == target or col.lower() in excluded_names
]

numeric_features = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in excluded
]

if not numeric_features:
    st.error(
        "No eligible numeric features found. "
        "Check the target and feature columns."
    )
    st.stop()

st.caption(
    "Protected-group columns and known ID/code columns are excluded "
    "from the model inputs in this prototype."
)

with st.expander("Review model input features"):
    st.write(numeric_features)

# Target must contain exactly two observed classes for this version.
valid = df[target].notna()
work = df.loc[valid].copy()

if work[target].nunique() != 2:
    st.error(
        "This starter version supports binary classification only. "
        "Choose a target with exactly two observed classes."
    )
    st.stop()

if len(work) < 20:
    st.error("Not enough records for a reliable train/test split.")
    st.stop()

X = work[numeric_features].replace([np.inf, -np.inf], np.nan)
y = work[target]

# Fairness group labels are kept separate from training features.
group_candidates = [
    col for col in [
        "sex_group", "education_group",
        "marriage_group", "age"
    ]
    if col in work.columns and col != target
]

group_col = st.selectbox(
    "Group for fairness review",
    options=["None"] + group_candidates
)

test_size = st.slider(
    "Test data percentage", 20, 40, 25, 5
) / 100

# Preserve group labels and row indices for test-set auditing.
indices = np.arange(len(work))

try:
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
except ValueError:
    st.error(
        "Cannot create a stratified split. "
        "Check the number of records in each target class."
    )
    st.stop()

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Numeric features only, with training-fold median imputation.
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

with st.spinner("Training and auditing the model..."):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

# Define the positive class as the second sorted target label.
classes = list(model.classes_)
positive_label = classes[-1]
actual_positive = (y_test.to_numpy() == positive_label).astype(int)
predicted_positive = (
    predictions == positive_label
).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions,
    pos_label=positive_label, zero_division=0
)
recall = recall_score(
    y_test, predictions,
    pos_label=positive_label, zero_division=0
)
f1 = f1_score(
    y_test, predictions,
    pos_label=positive_label, zero_division=0
)

# ---------------- MODEL PERFORMANCE ----------------
st.header("3. Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy:.1%}")
m2.metric("Precision", f"{precision:.1%}")
m3.metric("Recall", f"{recall:.1%}")
m4.metric("F1-score", f"{f1:.1%}")

st.caption(
    f"Positive class: {positive_label!r}. "
    "Metrics are calculated on held-out test data, not training data."
)

cm = confusion_matrix(
    y_test, predictions, labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {x}" for x in classes],
    columns=[f"Predicted {x}" for x in classes]
)

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Actual vs predicted outcomes"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_test, predictions,
        labels=classes,
        zero_division=0,
        output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )


# ---------------- ERROR ANALYSIS ----------------
st.header("4. Mistake Detective")

error_df = work.iloc[test_idx].copy()
error_df["actual_outcome"] = y_test.to_numpy()
error_df["predicted_outcome"] = predictions
error_df["error_type"] = np.select(
    [
        (actual_positive == 0) & (predicted_positive == 1),
        (actual_positive == 1) & (predicted_positive == 0)
    ],
    ["False Positive", "False Negative"],
    default="Correct"
)

e1, e2, e3 = st.columns(3)
e1.metric(
    "Correct predictions",
    int((predictions == y_test.to_numpy()).sum())
)
e2.metric(
    "False positives",
    int((error_df["error_type"] == "False Positive").sum())
)
e3.metric(
    "False negatives",
    int((error_df["error_type"] == "False Negative").sum())
)

st.bar_chart(error_df["error_type"].value_counts())

with st.expander("Inspect test-set prediction errors"):
    st.dataframe(
        error_df.loc[
            error_df["error_type"] != "Correct"
        ].head(200),
        use_container_width=True
    )


# ---------------- FAIRNESS REVIEW ----------------
st.header("5. Group Fairness Review")

fairness_df = pd.DataFrame({
    "group": (
        work.iloc[test_idx][group_col].astype(str).to_numpy()
        if group_col != "None"
        else ["All"] * len(test_idx)
    ),
    "actual_positive": actual_positive,
    "predicted_positive": predicted_positive
})

if group_col == "None":
    st.info("Select a group column to run group-wise fairness checks.")
else:
    fairness = fairness_df.groupby("group").agg(
        records=("group", "size"),
        actual_positive_rate=("actual_positive", "mean"),
        predicted_positive_rate=("predicted_positive", "mean")
    ).reset_index()

    fairness["prediction_minus_actual"] = (
        fairness["predicted_positive_rate"]
        - fairness["actual_positive_rate"]
    )

    fairness["records"] = fairness["records"].astype(int)

    st.dataframe(
        fairness.style.format({
            "actual_positive_rate": "{:.1%}",
            "predicted_positive_rate": "{:.1%}",
            "prediction_minus_actual": "{:+.1%}"
        }),
        use_container_width=True
    )

    fig = px.bar(
        fairness,
        x="group",
        y=["actual_positive_rate", "predicted_positive_rate"],
        barmode="group",
        title="Actual vs predicted positive outcome rates by group"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Descriptive disparity indicator, not a legal or ethical verdict.
    rate_gap = (
        fairness["predicted_positive_rate"].max()
        - fairness["predicted_positive_rate"].min()
    )

    st.metric(
        "Largest predicted-rate gap",
        f"{rate_gap:.1%}"
    )

    st.warning(
        "A group-rate difference is a review signal, not proof of bias. "
        "Check sample sizes, data quality, error rates, context, "
        "and relevant fairness criteria before drawing conclusions."
    )

    if (fairness["records"] < 30).any():
        st.info(
            "At least one group has fewer than 30 test records. "
            "Its rate may be unstable; interpret cautiously."
        )


# ---------------- PDF REPORT ----------------
st.header("6. Export Audit Report")

def make_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            "ZYVANTIS - Credit Risk Fairness Audit",
            styles["Title"]
        ),
        Spacer(1, 12),
        Paragraph(f"Data source: {data_source}", styles["Normal"]),
        Paragraph(f"Dataset rows: {len(df):,}", styles["Normal"]),
        Paragraph(f"Target: {target}", styles["Normal"]),
        Paragraph(
            f"Test-set positive class: {positive_label!r}",
            styles["Normal"]
        ),
        Spacer(1, 12),
        Paragraph("Data Quality", styles["Heading2"]),
        Paragraph(f"Missing cells: {missing_total}", styles["Normal"]),
        Paragraph(f"Duplicate rows: {duplicate_total}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Test-set Performance", styles["Heading2"])
    ]

    metric_table = [
        ["Metric", "Value"],
        ["Accuracy", f"{accuracy:.3f}"],
        ["Precision", f"{precision:.3f}"],
        ["Recall", f"{recall:.3f}"],
        ["F1-score", f"{f1:.3f}"]
    ]

    table = Table(metric_table)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(table)

    story += [
        Spacer(1, 12),
        Paragraph("Audit Limitations", styles["Heading2"]),
        Paragraph(
            "This report evaluates a locally trained prototype model "
            "on a held-out test set. Group-rate differences are "
            "descriptive signals and do not independently establish "
            "discrimination or causation. Review data provenance, "
            "group sizes, model design, and applicable requirements.",
            styles["Normal"]
        )
    ]

    if group_col != "None":
        story += [
            Spacer(1, 12),
            Paragraph("Group Rate Review", styles["Heading2"])
        ]
        rows = [[
            "Group", "Records",
            "Actual rate", "Predicted rate"
        ]]
        for _, r in fairness.iterrows():
            rows.append([
                str(r["group"])[:35],
                str(int(r["records"])),
                f'{r["actual_positive_rate"]:.1%}',
                f'{r["predicted_positive_rate"]:.1%}'
            ])

        ft = Table(rows, repeatRows=1)
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#183153")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))
        story.append(ft)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


if st.button("Generate PDF Audit Report"):
    try:
        pdf_bytes = make_pdf()
        st.download_button(
            "Download ZYVANTIS Audit Report",
            data=pdf_bytes,
            file_name="zyvantis_audit_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

st.divider()
st.caption(
    "ZYVANTIS is an audit-support prototype. "
    "It does not approve or reject real loan applications."
)
