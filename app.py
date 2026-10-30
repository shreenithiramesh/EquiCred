from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE + THEME (all CSS is local -> works 100% offline)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ZYVANTIS | Fairness Audit Studio", page_icon="⚖️", layout="wide")

ORANGE, AMBER, GOLD = "#ff8a1f", "#ffb347", "#ffd27a"
GREEN, RED, TEAL, PINK = "#2ecc71", "#ff4d4f", "#2dd4bf", "#f472b6"
PALETTE = [ORANGE, AMBER, TEAL, PINK, "#a78bfa", GOLD, "#60a5fa", GREEN]

st.markdown(
    f"""
<style>
.stApp {{background: radial-gradient(1200px 500px at 10% -10%, #3a1d05 0%, #0b0b0f 55%);}}
header[data-testid="stHeader"] {{background: transparent;}}
.hero {{padding: 22px 28px; border-radius: 18px; margin-bottom: 14px;
  background: linear-gradient(120deg, #7a3a05 0%, {ORANGE} 55%, {AMBER} 100%);
  box-shadow: 0 10px 30px rgba(255,138,31,.25);}}
.hero h1 {{margin:0; font-size: 2rem; color:#fff; letter-spacing:.5px;}}
.hero p {{margin:4px 0 0 0; color:#fff3e0; font-size:.95rem;}}
.badge {{display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:700;
  background:rgba(0,0,0,.35); color:#fff; margin-right:6px;}}
.kpi {{background:#16161c; border:1px solid #2b2b33; border-radius:16px; padding:14px 16px;
  border-top:3px solid {ORANGE};}}
.kpi .l {{color:#9a9aa5; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px;}}
.kpi .v {{color:#fff; font-size:1.75rem; font-weight:800;}}
.kpi .s {{color:{AMBER}; font-size:.75rem;}}
.card {{background:#16161c; border:1px solid #2b2b33; border-radius:16px; padding:16px 18px; margin-bottom:10px;}}
table.t {{width:100%; border-collapse:collapse; font-size:.88rem;}}
table.t th {{text-align:left; color:#9a9aa5; font-weight:600; padding:8px 10px; border-bottom:1px solid #2b2b33;}}
table.t td {{padding:9px 10px; border-bottom:1px solid #202028; color:#eee;}}
.pill {{padding:3px 12px; border-radius:6px; font-weight:800; font-size:.75rem;}}
.pass {{background:{GREEN}; color:#04210f;}} .fail {{background:{RED}; color:#2b0305;}}
.verdict {{border-left:5px solid {ORANGE}; background:#1b140d; padding:12px 16px; border-radius:8px;}}
div[data-baseweb="tab-list"] {{gap:6px;}}
button[data-baseweb="tab"] {{background:#16161c; border-radius:10px 10px 0 0; padding:8px 16px;}}
button[data-baseweb="tab"][aria-selected="true"] {{background:linear-gradient(90deg,{ORANGE},{AMBER}); color:#111 !important; font-weight:700;}}
</style>
""",
    unsafe_allow_html=True,
)


def style(fig, h=360, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, height=h, margin=dict(l=10, r=10, t=45 if title else 15, b=10),
        title=dict(text=title, font=dict(size=15, color="#fff")) if title else None,
        font=dict(family="Segoe UI, Arial, sans-serif", color="#e6e6e6"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#24242c")
    fig.update_yaxes(gridcolor="#24242c")
    return fig


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
TARGET = "default_next_month"
SENS_CODES = ["sex_code", "education_code", "marriage_code"]
GROUP_COLS = {"Gender": "sex_group", "Education": "education_group",
              "Marital status": "marriage_group", "Age band": "age_group"}
REQUIRED = ["credit_limit", "age", TARGET, "sex_group", "education_group", "marriage_group"]
DATA_FILE = Path(__file__).parent / "data" / "credit_card_clients_cleaned.xlsx"


@st.cache_data(show_spinner=False)
def read_bundled():
    return pd.read_excel(DATA_FILE)


def prep(df):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 200], right=False,
                             labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    return df


def base_features(df):
    num = df.select_dtypes("number").columns
    return [c for c in num if c not in SENS_CODES + [TARGET, "customer_id"]]


MODELS = {
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07, max_iter=200, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=42),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=600)),
}


@st.cache_data(show_spinner="Model train aaguthu... 🔥")
def train(df, model_name, use_sens, test_size):
    feats = base_features(df) + ([c for c in SENS_CODES if c in df] if use_sens else [])
    X, y = df[feats], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    model = MODELS[model_name]().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    groups = df.loc[Xte.index, list(GROUP_COLS.values())]
    return model, feats, Xte, yte.values, proba, groups


# ----------------------------------------------------------------------------
# FAIRNESS ENGINE
# ----------------------------------------------------------------------------
def group_metrics(y, pred, g, min_n=30):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred), "g": np.asarray(g)})
    rows = []
    for k, s in d.groupby("g"):
        if len(s) < min_n:
            continue
        tp = ((s.p == 1) & (s.y == 1)).sum(); fp = ((s.p == 1) & (s.y == 0)).sum()
        fn = ((s.p == 0) & (s.y == 1)).sum(); tn = ((s.p == 0) & (s.y == 0)).sum()
        rows.append(dict(group=k, n=len(s), actual_rate=s.y.mean(), flag_rate=s.p.mean(),
                         tpr=tp / (tp + fn) if tp + fn else np.nan,
                         fpr=fp / (fp + tn) if fp + tn else np.nan,
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         acc=(tp + tn) / len(s)))
    return pd.DataFrame(rows).set_index("group")


def run_tests(gm, tol, di_min):
    def spread(col):
        s = gm[col].dropna()
        if len(s) < 2:
            return 0.0, "-", "-"
        return s.max() - s.min(), s.idxmax(), s.idxmin()

    out = []
    specs = [("Demographic Parity", "flag_rate", "Gap in high-risk flag rate"),
             ("Equal Opportunity", "tpr", "Gap in true-positive rate (catching real defaulters)"),
             ("FPR Parity", "fpr", "Gap in false-positive rate (good customers wrongly flagged)"),
             ("Predictive Parity", "precision", "Gap in precision of a 'high-risk' flag"),
             ("Accuracy Parity", "acc", "Gap in overall accuracy")]
    for name, col, desc in specs:
        v, hi, lo = spread(col)
        out.append(dict(name=name, value=v, thr=f"≤ {tol:.2f}", hi=hi, lo=lo, ok=v <= tol,
                        pen=min(v / (2 * tol), 1), desc=desc))
    fr = gm["flag_rate"]
    di = (fr.min() / fr.max()) if len(fr) > 1 and fr.max() > 0 else 1.0
    out.insert(1, dict(name="Disparate Impact (80% rule)", value=di, thr=f"≥ {di_min:.2f}",
                       hi=fr.idxmax() if len(fr) else "-", lo=fr.idxmin() if len(fr) else "-",
                       ok=di >= di_min, pen=min(max(0, 1 - di) / (2 * (1 - di_min)), 1),
                       desc="Lowest flag rate ÷ highest flag rate"))
    score = 100 * (1 - np.mean([t["pen"] for t in out]))
    return out, score


def tests_table(tests):
    rows = "".join(
        f"<tr><td><b>{t['name']}</b><br><span style='color:#8a8a95;font-size:.75rem'>{t['desc']}</span></td>"
        f"<td>{t['value']:.3f}</td><td>{t['thr']}</td><td>{t['hi']}</td><td>{t['lo']}</td>"
        f"<td><span class='pill {'pass' if t['ok'] else 'fail'}'>{'PASS' if t['ok'] else 'FAIL'}</span></td></tr>"
        for t in tests)
    return ("<div class='card'><table class='t'><tr><th>Test</th><th>Value</th><th>Limit</th>"
            f"<th>Highest group</th><th>Lowest group</th><th>Status</th></tr>{rows}</table></div>")


def verdict_text(tests, score, attr):
    fails = [t for t in tests if not t["ok"]]
    if not fails:
        return (f"All {len(tests)} fairness tests passed for **{attr}** (score {score:.0f}/100). "
                "No major disparity found at the current thresholds, but keep monitoring - passing a test does not prove the model is fair.")
    worst = max(fails, key=lambda t: t["pen"])
    names = ", ".join(t["name"] for t in fails)
    return (f"{len(fails)} of {len(tests)} tests failed for **{attr}** ({names}). Biggest concern: **{worst['name']}** - "
            f"**{worst['hi']}** vs **{worst['lo']}** (value {worst['value']:.3f}). "
            "This highlights an area that needs review; it does not by itself prove the model is unfair.")


def apply_group_thresholds(y, s, g, base_t, mode, min_n=30):
    y, s, g = np.asarray(y), np.asarray(s), np.asarray(g)
    grid = np.linspace(0.02, 0.98, 97)
    target = ((s >= base_t).mean() if mode == "Demographic Parity"
              else ((s >= base_t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    th = {}
    for k in np.unique(g):
        m = g == k
        if m.sum() < min_n or (mode != "Demographic Parity" and (m & (y == 1)).sum() < 10):
            th[k] = base_t
            continue
        sc = s[m] if mode == "Demographic Parity" else s[m & (y == 1)]
        vals = np.array([(sc >= t).mean() for t in grid])
        th[k] = float(grid[np.argmin(np.abs(vals - target))])
    thr = pd.Series(g).map(th).values
    return (s >= thr).astype(int), th


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### ⚖️ <span style='color:{ORANGE}'>ZYVANTIS</span> Controls", unsafe_allow_html=True)
    st.caption("🟢 Offline mode: no internet / API needed")
    src = st.radio("Dataset", ["Bundled cleaned dataset", "Upload my own (.xlsx / .csv)"])
    if src.startswith("Bundled"):
        raw = read_bundled()
    else:
        up = st.file_uploader("Upload file", type=["xlsx", "csv"])
        if up is None:
            st.info("File upload pannunga, illana bundled dataset use pannunga.")
            st.stop()
        raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df = prep(raw)

    st.divider()
    model_name = st.selectbox("Model", list(MODELS))
    use_sens = st.toggle("Model-ku sensitive features kudukkanum? (sex / education / marriage)", value=False,
                         help="OFF = 'fairness through unawareness'. ON = model directly sensitive data paakkum.")
    test_size = st.slider("Test split", 0.2, 0.5, 0.3, 0.05)
    attr_label = st.selectbox("Audit attribute", list(GROUP_COLS))
    attr = GROUP_COLS[attr_label]

    st.divider()
    thresh = st.slider("Decision threshold (flag as HIGH RISK)", 0.05, 0.95, 0.40, 0.01)
    tol = st.slider("Fairness gap tolerance", 0.02, 0.30, 0.10, 0.01)
    di_min = st.slider("Disparate-impact minimum", 0.50, 1.00, 0.80, 0.05)
    min_n = st.slider("Min group size to audit", 10, 300, 30, 10)

model, feats, Xte, yte, proba, groups = train(df, model_name, use_sens, test_size)
pred = (proba >= thresh).astype(int)
gm = group_metrics(yte, pred, groups[attr], min_n)
tests, score = run_tests(gm, tol, di_min)

# ----------------------------------------------------------------------------
# HERO + KPIs
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><span class="badge">TEAM ZYVANTIS</span><span class="badge">DT-FIN-10</span>
<span class="badge">OFFLINE</span><h1>AI Bias &amp; Fairness Audit Studio</h1>
<p>Audit credit-risk predictions for disparities - accuracy, fairness, mitigation and auto-report in one dashboard.</p></div>""",
    unsafe_allow_html=True)

c = st.columns(5)
kpi(c[0], "Accuracy", f"{accuracy_score(yte, pred):.1%}", f"{len(yte):,} test customers")
kpi(c[1], "Recall", f"{recall_score(yte, pred):.1%}", "defaulters caught")
kpi(c[2], "Precision", f"{precision_score(yte, pred, zero_division=0):.1%}", "flags that were right")
kpi(c[3], "ROC-AUC", f"{roc_auc_score(yte, proba):.3f}", model_name)
kpi(c[4], "Fairness score", f"{score:.0f}/100", f"{sum(t['ok'] for t in tests)}/{len(tests)} tests passed")
st.write("")

tabs = st.tabs(["🏠 Overview", "🧪 Fairness Tests", "🔍 Group Explorer", "🛠️ Mitigation Lab",
                "🔁 Counterfactual & Proxy", "🎯 What-If Simulator", "📄 Audit Report"])

# ---- OVERVIEW -------------------------------------------------------------
with tabs[0]:
    l, r = st.columns([1, 1.4])
    with l:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": "/100", "font": {"color": "#fff"}},
            title={"text": "Fairness Score", "font": {"color": "#ddd"}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": ORANGE},
                   "steps": [{"range": [0, 50], "color": "#4a1618"}, {"range": [50, 75], "color": "#4a3510"},
                             {"range": [75, 100], "color": "#10361f"}]}))
        st.plotly_chart(style(g, 300), width="stretch")
        st.markdown(f"<div class='verdict'>{verdict_text(tests, score, attr_label)}</div>", unsafe_allow_html=True)
    with r:
        d = gm.reset_index()
        fig = go.Figure()
        for col, nm, colr in [("actual_rate", "Actual default rate", TEAL), ("flag_rate", "Model flag rate", ORANGE)]:
            fig.add_bar(x=d["group"], y=d[col], name=nm, marker_color=colr, text=d[col].map("{:.0%}".format))
        fig.update_layout(barmode="group")
        st.plotly_chart(style(fig, 360, f"Actual vs predicted risk by {attr_label}"), width="stretch")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for k in gm.index:
            m = (groups[attr] == k).values
            if m.sum() > 1 and len(np.unique(yte[m])) == 2:
                f, t, _ = roc_curve(yte[m], proba[m])
                fig.add_scatter(x=f, y=t, mode="lines", name=f"{k} (AUC {roc_auc_score(yte[m], proba[m]):.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#666"), showlegend=False)
        st.plotly_chart(style(fig, 340, "ROC curve per group"), width="stretch")
    with b:
        cnt = gm["n"].reset_index()
        fig = px.pie(cnt, names="group", values="n", hole=.62, color_discrete_sequence=PALETTE)
        st.plotly_chart(style(fig, 340, "Group composition (test set)"), width="stretch")

# ---- FAIRNESS TESTS -------------------------------------------------------
with tabs[1]:
    st.markdown(f"#### Test results - **{attr_label}** @ threshold {thresh:.2f}")
    st.markdown(tests_table(tests), unsafe_allow_html=True)
    tv = pd.DataFrame({"Test": [t["name"] for t in tests], "Value": [t["value"] for t in tests],
                       "ok": ["Pass" if t["ok"] else "Fail" for t in tests]})
    fig = px.bar(tv, x="Test", y="Value", color="ok", text=tv["Value"].map("{:.2f}".format),
                 color_discrete_map={"Pass": GREEN, "Fail": RED})
    st.plotly_chart(style(fig, 330, "Metric value per test (DI should be HIGH, others LOW)"), width="stretch")
    st.markdown("##### Fairness across ALL attributes (heatmap)")
    rows = []
    for lab, col in GROUP_COLS.items():
        t2, s2 = run_tests(group_metrics(yte, pred, groups[col], min_n), tol, di_min)
        rows.append({"Attribute": lab, **{t["name"]: t["value"] for t in t2}, "Score": s2})
    hm = pd.DataFrame(rows).set_index("Attribute")
    fig = px.imshow(hm.round(2), text_auto=True, aspect="auto",
                    color_continuous_scale=["#10361f", "#4a3510", "#ff8a1f", "#ff4d4f"])
    st.plotly_chart(style(fig, 300), width="stretch")
    st.caption("Score column = 0-100 fairness score. Baaki columns raw gap values (Disparate Impact thavira - adhu ratio).")

# ---- GROUP EXPLORER -------------------------------------------------------
with tabs[2]:
    d = gm.reset_index().melt("group", ["actual_rate", "flag_rate", "tpr", "fpr", "precision", "acc"],
                              var_name="metric", value_name="v")
    fig = px.bar(d, x="metric", y="v", color="group", barmode="group", color_discrete_sequence=PALETTE)
    st.plotly_chart(style(fig, 380, f"All metrics by {attr_label}"), width="stretch")
    st.dataframe(gm.style.format({c: "{:.3f}" for c in gm.columns if c != "n"}), width="stretch")

    st.markdown("##### 🧬 Intersectional bias heatmap")
    i1, i2, i3 = st.columns(3)
    ax1 = i1.selectbox("Row attribute", list(GROUP_COLS), index=0, key="ia")
    ax2 = i2.selectbox("Column attribute", list(GROUP_COLS), index=1, key="ib")
    met = i3.selectbox("Metric", ["flag_rate", "fpr", "tpr", "actual_rate"], key="im")
    if ax1 == ax2:
        st.info("Rendu different attribute select pannunga.")
    else:
        t = pd.DataFrame({"r": groups[GROUP_COLS[ax1]].values, "c": groups[GROUP_COLS[ax2]].values, "y": yte, "p": pred})

        def cell(s):
            if len(s) < min_n:
                return np.nan
            return {"flag_rate": s.p.mean(), "actual_rate": s.y.mean(),
                    "tpr": ((s.p == 1) & (s.y == 1)).sum() / max((s.y == 1).sum(), 1),
                    "fpr": ((s.p == 1) & (s.y == 0)).sum() / max((s.y == 0).sum(), 1)}[met]

        piv = t.groupby(["r", "c"]).apply(cell, include_groups=False).unstack()
        fig = px.imshow(piv, text_auto=".2f", aspect="auto", color_continuous_scale="YlOrRd")
        st.plotly_chart(style(fig, 380), width="stretch")
        st.caption(f"Blank cell = {min_n}-ku kammiya customers. Dark cell = highest {met}.")

# ---- MITIGATION LAB -------------------------------------------------------
with tabs[3]:
    st.markdown("#### 🛠️ Bias Mitigation Lab - per-group thresholds (post-processing)")
    st.caption("Model retrain panna thevai illa. Each group-ku thani threshold set panni gap-a kuraikkalam.")
    mode = st.radio("Goal", ["Demographic Parity", "Equal Opportunity"], horizontal=True)
    pred2, th = apply_group_thresholds(yte, proba, groups[attr], thresh, mode, min_n)
    gm2 = group_metrics(yte, pred2, groups[attr], min_n)
    t2, s2 = run_tests(gm2, tol, di_min)
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Fairness score", f"{s2:.0f}", f"before {score:.0f}  ({s2 - score:+.0f})")
    kpi(c2, "Accuracy", f"{accuracy_score(yte, pred2):.1%}", f"before {accuracy_score(yte, pred):.1%}")
    kpi(c3, "Recall", f"{recall_score(yte, pred2):.1%}", f"before {recall_score(yte, pred):.1%}")
    st.write("")
    cmp = pd.DataFrame({"Test": [t["name"] for t in tests] * 2,
                        "Value": [t["value"] for t in tests] + [t["value"] for t in t2],
                        "Version": ["Before"] * len(tests) + ["After"] * len(t2)})
    fig = px.bar(cmp, x="Test", y="Value", color="Version", barmode="group",
                 color_discrete_map={"Before": "#8a8a95", "After": ORANGE})
    st.plotly_chart(style(fig, 340, "Before vs After mitigation"), width="stretch")
    st.markdown("**Learned thresholds**")
    st.dataframe(pd.DataFrame({"group": th.keys(), "threshold": [round(v, 2) for v in th.values()]}).set_index("group").T,
                 width="stretch")
    st.warning("Group-wise thresholds legal / policy review-ku apram dhaan production-la use pannanum. "
               "Idhu oru analysis tool, final decision illa.")

# ---- COUNTERFACTUAL & PROXY ----------------------------------------------
with tabs[4]:
    st.markdown("#### 🔁 Counterfactual flip test (Gender)")
    if use_sens and "sex_code" in feats:
        Xf = Xte.copy()
        Xf["sex_code"] = 3 - Xf["sex_code"]
        pf = (model.predict_proba(Xf)[:, 1] >= thresh).astype(int)
        flips = pf != pred
        c1, c2 = st.columns(2)
        kpi(c1, "Decisions that flip", f"{flips.mean():.2%}", "when ONLY gender is swapped")
        kpi(c2, "Mean score change", f"{np.abs(model.predict_proba(Xf)[:, 1] - proba).mean():.4f}", "absolute")
        st.caption("Idhu 0% illana, model directly gender-a use pannudhu.")
    else:
        st.info("Model sensitive features use pannala (sidebar toggle OFF). Counterfactual flip = 0 by design. "
                "Toggle ON panna test run aagum. Keezha proxy check paarunga 👇")

    st.markdown("#### 🕵️ Proxy detector - hidden bias evidence")
    code_map = {"sex_group": "sex_code", "education_group": "education_code", "marriage_group": "marriage_code"}
    ref = groups[attr].astype("category")
    codes = pd.Series(ref.cat.codes.values, index=Xte.index)
    non_sens = [f for f in feats if f not in SENS_CODES]
    corr = Xte[non_sens].corrwith(codes).abs().sort_values(ascending=False).head(10)
    fig = px.bar(corr[::-1], orientation="h", color=corr[::-1], color_continuous_scale="YlOrRd")
    st.plotly_chart(style(fig, 360, f"Features most correlated with {attr_label} (|r|)"), width="stretch")
    try:
        Xa, Xb, ya, yb = train_test_split(Xte[non_sens], ref.cat.codes.values, test_size=0.3, random_state=1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xa, ya)
        pr = clf.predict_proba(Xb)
        auc = roc_auc_score(yb, pr[:, 1]) if pr.shape[1] == 2 else roc_auc_score(yb, pr, multi_class="ovr")
        kpi(st.columns(3)[0], "Leakage AUC", f"{auc:.2f}", "0.5 = no leakage, 1.0 = fully predictable")
    except Exception:
        pass
    st.caption("Sensitive column remove panna kooda, vera features moolama sensitive group-a guess panna mudinjaa - adhu PROXY bias.")

# ---- WHAT-IF --------------------------------------------------------------
with tabs[5]:
    st.markdown("#### 🎯 Single-customer What-If simulator")
    base = Xte.median(numeric_only=True)
    cols = st.columns(3)
    row = base.copy()
    row["credit_limit"] = cols[0].slider("Credit limit", int(df.credit_limit.min()), int(df.credit_limit.max()), int(base["credit_limit"]), 10000)
    row["age"] = cols[1].slider("Age", 18, 80, int(base["age"]))
    row["repayment_status_sep"] = cols[2].slider("Last month repayment status (-2 early … 8 very late)", -2, 8, int(base["repayment_status_sep"]))
    cols2 = st.columns(3)
    row["bill_amount_sep"] = cols2[0].number_input("Last bill amount", value=int(base["bill_amount_sep"]), step=1000)
    row["payment_amount_sep"] = cols2[1].number_input("Last payment amount", value=int(base["payment_amount_sep"]), step=500)
    if use_sens and "sex_code" in feats:
        row["sex_code"] = 1 if cols2[2].radio("Gender", ["Male", "Female"], horizontal=True) == "Male" else 2
    p = float(model.predict_proba(pd.DataFrame([row[feats]]))[:, 1][0])
    g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                               title={"text": "Default probability"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": ORANGE},
                                      "threshold": {"line": {"color": "#fff", "width": 4}, "value": thresh * 100},
                                      "steps": [{"range": [0, 30], "color": "#10361f"}, {"range": [30, 60], "color": "#4a3510"},
                                                {"range": [60, 100], "color": "#4a1618"}]}))
    st.plotly_chart(style(g, 300), width="stretch")
    st.markdown(f"<div class='verdict'>Decision: <b>{'HIGH RISK ⚠️' if p >= thresh else 'LOW RISK ✅'}</b> "
                f"(score {p:.2f} vs threshold {thresh:.2f}). Other features dataset median-la irukku.</div>", unsafe_allow_html=True)

# ---- REPORT ---------------------------------------------------------------
with tabs[6]:
    st.markdown("#### 📄 Auto-generated audit report (offline, 1-click)")
    rows = "".join(f"<tr><td>{t['name']}</td><td>{t['value']:.3f}</td><td>{t['thr']}</td><td>{t['hi']}</td>"
                   f"<td>{t['lo']}</td><td style='color:{GREEN if t['ok'] else RED};font-weight:700'>"
                   f"{'PASS' if t['ok'] else 'FAIL'}</td></tr>" for t in tests)
    gtab = gm.round(3).reset_index().to_html(index=False, border=0)
    report = f"""<html><head><meta charset="utf-8"><title>ZYVANTIS Audit</title><style>
body{{font-family:Segoe UI,Arial;background:#0b0b0f;color:#eee;padding:30px;max-width:980px;margin:auto}}
h1{{color:{ORANGE}}} table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{border-bottom:1px solid #2b2b33;padding:8px;text-align:left}}
th{{color:{AMBER}}} .box{{background:#16161c;padding:14px;border-radius:10px;border-left:5px solid {ORANGE}}}</style></head><body>
<h1>ZYVANTIS - Credit-Risk Fairness Audit</h1><p>Generated {datetime.now():%d %b %Y, %H:%M} | Problem DT-FIN-10</p>
<p><b>Model:</b> {model_name} | <b>Sensitive features in model:</b> {'Yes' if use_sens else 'No'} | <b>Audit attribute:</b> {attr_label} |
<b>Threshold:</b> {thresh:.2f} | <b>Test size:</b> {len(yte):,}</p>
<p><b>Accuracy</b> {accuracy_score(yte, pred):.1%} | <b>Recall</b> {recall_score(yte, pred):.1%} |
<b>Precision</b> {precision_score(yte, pred, zero_division=0):.1%} | <b>ROC-AUC</b> {roc_auc_score(yte, proba):.3f} |
<b>Fairness score</b> {score:.0f}/100</p>
<div class="box">{verdict_text(tests, score, attr_label).replace('**', '')}</div>
<h2>Fairness tests</h2><table><tr><th>Test</th><th>Value</th><th>Limit</th><th>Highest</th><th>Lowest</th><th>Status</th></tr>{rows}</table>
<h2>Group metrics</h2>{gtab}
<p style="color:#888;font-size:12px">This report highlights areas for review. It does not by itself establish that a model is unfair or lawful.</p>
</body></html>"""
    if hasattr(st, "iframe"):
        st.iframe(report, height=620)
    else:
        st.components.v1.html(report, height=620, scrolling=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download HTML report", report, "zyvantis_audit_report.html", "text/html", width="stretch")
    d2.download_button("⬇️ Download group metrics CSV", gm.to_csv().encode(), "group_metrics.csv", "text/csv", width="stretch")
# ---- ERROR ANALYSIS -------------------------------------------------------
with tabs[7]:
    st.markdown("#### 🚨 How many errors did the AI make?")
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    errors = fp + fn
    c = st.columns(3)
    kpi(c[0], "Total errors", f"{errors:,}", f"{errors/len(yte):.1%} of {len(yte):,} decisions")
    kpi(c[1], "Good customers rejected (FP)", f"{fp:,}", "lost business")
    kpi(c[2], "Defaulters approved (FN)", f"{fn:,}", "lost money")

    cm = np.array([[tp, fp], [fn, tn]])
    fig = px.imshow(cm, text_auto=True, x=["Actual: defaulted", "Actual: paid"],
                    y=["AI: risky", "AI: safe"], color_continuous_scale="YlOrRd")
    st.plotly_chart(style(fig, 340, "Confusion matrix"), width="stretch")
