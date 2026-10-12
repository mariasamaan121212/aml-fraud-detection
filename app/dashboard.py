import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from api.main import Transaction, score_transaction
from src.limits import HARD
from src.predict import load_model

st.set_page_config(page_title="AML Fraud Scoring", page_icon="🛡️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, #1b2440 0%, #11182b 60%, #0E1117 100%);
  border: 1px solid #263154;}
.hero-title {font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;}
.hero-sub {color: #9aa7c7; margin: .2rem 0 .9rem 0;}
.chip {display: inline-block; padding: .2rem .7rem; margin-right: .4rem; border-radius: 999px;
  background: #1f2a4a; color: #b9c6ee; font-size: .8rem; border: 1px solid #2d3c6b;}
.score-card {padding: 1.2rem 1.4rem; border-radius: 16px; background: #161B26;
  border: 1px solid #263154; margin-bottom: .8rem;}
.score-label {color: #8da0d0; font-size: .75rem; letter-spacing: 2px;}
.score-value {font-size: 3.2rem; font-weight: 800; line-height: 1.1; color: #8FA8FF;}
.score-sub {color: #9aa7c7; font-size: .9rem; margin-top: .3rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

_, meta = load_model()
BASE_RATE = meta["n_train_fraud"] / meta["n_train_rows"]
REPORTS = ROOT / "reports"


@st.cache_data
def load_json(name):
    return json.loads((REPORTS / name).read_text())


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CHANNELS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
FLAGS = ["is_cross_bank_transfer", "is_cross_currency_transfer", "is_new_receiver",
         "is_new_bank", "is_new_payment_format"]

PRESETS = {
    "Typical returning sender (illustrative)": dict(
        amount=1400.0, payment_channel="Cheque", time_since_last_transaction=720.0,
        velocity_score=17, spending_deviation_score=-0.2, hour=11, day_of_week=2,
        is_cross_bank_transfer=True, is_cross_currency_transfer=False,
        is_new_receiver=False, is_new_bank=False, is_new_payment_format=False),
    "First transaction of a new sender (illustrative)": dict(
        amount=1500.0, payment_channel="Credit Card", time_since_last_transaction=0.0,
        velocity_score=0, spending_deviation_score=0.0, hour=9, day_of_week=1,
        is_cross_bank_transfer=True, is_cross_currency_transfer=False,
        is_new_receiver=True, is_new_bank=True, is_new_payment_format=True),
    "Rapid ACH payment to a new receiver (illustrative)": dict(
        amount=11139.43, payment_channel="ACH", time_since_last_transaction=0.0,
        velocity_score=61, spending_deviation_score=0.5, hour=14, day_of_week=2,
        is_cross_bank_transfer=True, is_cross_currency_transfer=False,
        is_new_receiver=True, is_new_bank=True, is_new_payment_format=False),
}
for key, value in PRESETS[next(iter(PRESETS))].items():
    st.session_state.setdefault(key, value)


def apply_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


def fmt_pct(p):
    pct = p * 100
    if pct < 0.0001:
        return "< 0.0001%"
    return f"{pct:.2g}%" if pct < 1 else f"{pct:.1f}%"


def fmt_times(x):
    if x >= 10:
        return f"{x:,.0f}x"
    if x >= 1:
        return f"{x:.1f}x"
    return f"{x:.2g}x"


LABELS = {
    "amount": "Amount",
    "payment_channel": "Payment channel",
    "time_since_last_transaction": "Seconds since last transaction",
    "velocity_score": "Previous transactions",
    "spending_deviation_score": "Spending deviation",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "is_new_receiver": "New receiver",
    "is_new_bank": "New receiver bank",
    "is_new_payment_format": "New payment format",
    "is_cross_bank_transfer": "Cross-bank transfer",
    "is_cross_currency_transfer": "Cross-currency transfer",
    "burst": "Back-to-back transaction",
    "has_history": "Sender has history",
}
YES_NO = set(FLAGS) | {"burst", "has_history"}


def show_value(feature, v):
    if feature in YES_NO:
        return "Yes" if v else "No"
    if feature == "day_of_week":
        return DAYS[int(v)]
    if feature == "time_since_last_transaction":
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


st.markdown(
    f"""
<div class="hero">
<div class="hero-title">AML Fraud Scoring</div>
<div class="hero-sub">Explainable risk scores for bank transfers, built on a synthetic transaction dataset.</div>
<span class="chip">LightGBM</span>
<span class="chip">{meta['n_train_rows']:,} training rows</span>
<span class="chip">{BASE_RATE * 100:.3f}% fraud rate in training</span>
<span class="chip">synthetic data</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_score, tab_perf, tab_about = st.tabs(["Score a transaction", "Model performance", "About & limits"])

# ---------------------------------------------------------------- Score a transaction
with tab_score:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        with st.container(border=True):
            st.selectbox("Start from an example", list(PRESETS), key="preset", on_change=apply_preset)
            st.number_input("Amount", min_value=0.0, max_value=HARD["amount"][1],
                            step=100.0, format="%.2f", key="amount")
            st.selectbox("Payment channel", CHANNELS, key="payment_channel")
            c1, c2 = st.columns(2)
            c1.number_input("Seconds since sender's last transaction", min_value=0.0,
                            max_value=HARD["time_since_last_transaction"][1],
                            step=60.0, format="%.0f", key="time_since_last_transaction")
            c2.number_input("Sender's previous transactions", min_value=0,
                            max_value=HARD["velocity_score"][1], step=1, key="velocity_score")
            st.number_input("Spending deviation score", min_value=HARD["spending_deviation_score"][0],
                            max_value=HARD["spending_deviation_score"][1],
                            step=0.1, format="%.2f", key="spending_deviation_score")
            c3, c4 = st.columns(2)
            c3.slider("Hour of day", 0, 23, key="hour")
            c4.selectbox("Day of week", range(7), format_func=lambda i: DAYS[i], key="day_of_week")
            t1, t2 = st.columns(2)
            t1.toggle("New receiver", key="is_new_receiver")
            t1.toggle("New receiver bank", key="is_new_bank")
            t1.toggle("New payment format", key="is_new_payment_format")
            t2.toggle("Cross-bank transfer", key="is_cross_bank_transfer")
            t2.toggle("Cross-currency transfer", key="is_cross_currency_transfer")

    payload = {
        "amount": st.session_state["amount"],
        "payment_channel": st.session_state["payment_channel"],
        "time_since_last_transaction": st.session_state["time_since_last_transaction"],
        "velocity_score": st.session_state["velocity_score"],
        "spending_deviation_score": st.session_state["spending_deviation_score"],
        "hour": st.session_state["hour"],
        "day_of_week": st.session_state["day_of_week"],
    }
    payload.update({name: int(st.session_state[name]) for name in FLAGS})

    with right:
        try:
            transaction = Transaction(**payload)
        except ValidationError as err:
            st.markdown("#### This combination is not allowed")
            for item in err.errors():
                st.error(item["msg"].removeprefix("Value error, "))
        else:
            r = score_transaction(transaction)

            st.markdown(
                f"""
<div class="score-card">
<div class="score-label">RISK SCORE</div>
<div class="score-value">{fmt_pct(r['risk_score'])}</div>
<div class="score-sub">{fmt_times(r['times_train_base_rate'])} the {BASE_RATE * 100:.3f}% fraud rate of the training data</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.progress(min(r["risk_score"], 1.0))
            for warning in r["warnings"]:
                st.warning(warning)

            st.markdown("##### What pushed the score")
            rows = []
            for f in r["top_factors"]:
                name = LABELS.get(f["feature"], f["feature"])
                rows.append({"label": f"{name}: {show_value(f['feature'], f['value'])}",
                             "contribution": f["contribution"]})
            rows.append({"label": "all other inputs", "contribution": r["other_contribution"]})
            chart_df = pd.DataFrame(rows)
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("contribution:Q", title="Effect on the model score (log-odds)"),
                    y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=280)),
                    color=alt.condition(alt.datum.contribution > 0,
                                        alt.value("#FF6B6B"), alt.value("#4DA3FF")),
                    tooltip=["label", "contribution"],
                )
                .properties(height=36 * len(chart_df))
            )
            st.altair_chart(chart)
            st.caption(
                "Red bars push the score up, blue bars push it down, compared with an average "
                "transaction. They describe what the model used, not what caused fraud."
            )
            with st.expander("How the score is built"):
                total = r["base_value"] + sum(f["contribution"] for f in r["top_factors"]) \
                    + r["other_contribution"]
                st.write(f"Starting point for an average transaction (log-odds): **{r['base_value']:.2f}**")
                st.write(f"Effect of this transaction's inputs: **{total - r['base_value']:+.2f}**")
                st.write(f"Model score (log-odds): **{total:.2f}**, which is a risk score of "
                         f"**{fmt_pct(r['risk_score'])}**")
            st.caption(r["note"])

# ---------------------------------------------------------------- Model performance
with tab_perf:
    bench = load_json("test_results.json")
    hours = load_json("chronological_results.json")
    channels = load_json("channel_breakdown.json")

    def at(budgets, k):
        return next(r for r in budgets if r["alerts"] == k)

    st.markdown("### Held-out test results")
    st.caption(
        "Each test set was scored once, after the model settings were chosen on validation data. "
        "Intervals are 95% bootstrap intervals over rows. Rows from the same account are not independent, "
        "so the real uncertainty is probably wider."
    )

    top500 = at(bench["budget_table"], 500)
    full = at(bench["budget_table"], 56918)
    random_rate = bench["test_fraud"] / bench["test_rows"]

    m1, m2, m3 = st.columns(3)
    lo, hi = bench["bootstrap_95"]["pr_auc"]
    m1.metric("PR-AUC (random split)", f"{bench['pr_auc']:.3f}",
              help=f"95% interval {lo:.3f} to {hi:.3f}. A random scorer would get {random_rate:.4f}.")
    lo, hi = bench["bootstrap_95"]["precision@500"]
    m2.metric("Precision, top 500 alerts", f"{top500['precision']:.1%}",
              help=f"95% interval {lo:.1%} to {hi:.1%}.")
    gain = (full["recall"] - bench["rule"]["recall"]) * 100
    m3.metric("Fraud caught, top 56,918 alerts", f"{full['recall']:.1%}",
              delta=f"{gain:+.1f} pts vs simple rule",
              help=f"The simple rule (ACH + new receiver) flagged {bench['rule']['flagged']:,} alerts.")

    st.markdown("##### Precision vs recall at different alert budgets (random-split test set)")
    pts = pd.concat([
        pd.DataFrame(bench["budget_table"]).assign(series="LightGBM"),
        pd.DataFrame([{"alerts": bench["rule"]["flagged"],
                       "precision": bench["rule"]["precision"],
                       "recall": bench["rule"]["recall"]}]).assign(series="Simple rule"),
    ])
    curve = (
        alt.Chart(pts)
        .mark_line(point=alt.OverlayMarkDef(size=90))
        .encode(
            x=alt.X("recall:Q", title="Recall: share of fraud caught", axis=alt.Axis(format="%")),
            y=alt.Y("precision:Q", title="Precision (log scale)",
                    scale=alt.Scale(type="log"),
                    axis=alt.Axis(format="%", values=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["LightGBM", "Simple rule"],
                                            range=["#8FA8FF", "#FFB454"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["series", "alerts",
                     alt.Tooltip("precision:Q", format=".2%"),
                     alt.Tooltip("recall:Q", format=".1%")],
        )
        .properties(height=320)
    )
    st.altair_chart(curve)
    st.caption(
        "Each LightGBM point is a number of alerts (500 up to 56,918) taken from the top of the model's ranking. "
        f"The simple rule is 'ACH and new receiver'; it produces one fixed set of {bench['rule']['flagged']:,} alerts."
    )

    st.markdown("##### Same model, different ways of splitting the data")

    def summary(name, d, rows, fraud):
        t500 = at(d["budget_table"], 500)
        ci = d["bootstrap_95"]
        return {
            "Evaluation": name,
            "Fraud cases": f"{fraud:,}",
            "Fraud rate": f"{fraud / rows:.3%}",
            "PR-AUC": f"{d['pr_auc']:.3f} ({ci['pr_auc'][0]:.3f} to {ci['pr_auc'][1]:.3f})",
            "Precision, top 500": f"{t500['precision']:.1%} "
                                  f"({ci['precision@500'][0]:.1%} to {ci['precision@500'][1]:.1%})",
            "Simple-rule recall": f"{d['rule']['recall']:.1%}",
        }

    comparison = pd.DataFrame([
        summary("Random split, test", bench, bench["test_rows"], bench["test_fraud"]),
        summary("Hour-of-day split, validation (hours 16-20)", hours["valid"],
                hours["valid"]["rows"], hours["valid"]["fraud"]),
        summary("Hour-of-day split, test (hours 20-23)", hours["test"],
                hours["test"]["rows"], hours["test"]["fraud"]),
    ])
    st.dataframe(comparison, hide_index=True)
    st.caption(
        "The hour-of-day split trains on hours 0-16 and uses 21 inputs (no hour features). "
        "Fraud rates differ between the splits, so PR-AUC and precision are not directly comparable across rows. "
        f"The simple rule's recall also fell on the last hours ({hours['valid']['rule']['recall']:.1%} to "
        f"{hours['test']['rule']['recall']:.1%}), so part of the drop comes from a harder period. "
        "We have not established the full cause."
    )

    st.markdown("##### Where the model finds fraud: ACH vs other channels")
    group_rows = []
    for part, label in [("valid", "Validation (hours 16-20)"), ("test", "Test (hours 20-23)")]:
        for group in ["ACH", "non_ACH"]:
            g = channels[part][group]
            group_rows.append({
                "Split": label,
                "Group": group.replace("_", "-"),
                "Rows": f"{g['rows']:,}",
                "Fraud": f"{g['fraud']:,}",
                "Fraud rate": f"{g['base_rate']:.3%}",
                "PR-AUC": f"{g['pr_auc']:.3f}",
                "PR-AUC vs random": f"{g['lift_vs_base']:.0f}x",
            })
    st.dataframe(pd.DataFrame(group_rows), hide_index=True)
    top = channels["test"]["caught_in_top_alerts"]
    st.caption(
        f"Fraud found in the top 500 alerts of the hour-split test: {top['500']['ACH']} in ACH, "
        f"{top['500']['non_ACH']} outside ACH. Among the top 5,000: {top['5000']['ACH']} in ACH, "
        f"{top['5000']['non_ACH']} outside ACH. The model mostly detects fraud in ACH transfers. "
        "This breakdown was run after seeing the test results, so it is a diagnosis, not part of model selection."
    )

    st.markdown("##### What the model relies on")
    st.image(str(REPORTS / "feature_importance.png"))
    st.caption(
        "Average size of each input's effect on the score, over 200,000 random validation rows (random split). "
        "velocity_score has the largest effect. The dataset has no account identifier, so we cannot rule out "
        "that it partly acts as an account fingerprint shared between training and evaluation rows."
    )

    with st.expander("Hyperparameter search"):
        tuning = pd.read_csv(REPORTS / "tuning_results.csv").sort_values("PR_AUC", ascending=False)
        st.dataframe(tuning.round(4), hide_index=True)
        st.caption(
            f"Validation PR-AUC ranged from {tuning['PR_AUC'].min():.3f} to {tuning['PR_AUC'].max():.3f}. "
            "The search used a learning rate of 0.1 to run faster; the final model uses 0.05 with 1,500 trees. "
            "Differences this small are probably within sampling noise (the validation set has 1,311 fraud "
            "cases), so the original settings were kept."
        )

# ---------------------------------------------------------------- About & limits
with tab_about:
    perf_bench = load_json("test_results.json")
    perf_hours = load_json("chronological_results.json")
    perf_channels = load_json("channel_breakdown.json")

    st.markdown("### Model card")
    st.markdown(
        "**Intended use.** A portfolio demonstration of an explainable fraud-scoring pipeline: data audit, "
        "modelling, evaluation, an API and this dashboard. It is **not** a production AML system and must "
        "not be used to make decisions about real people or transactions."
    )

    st.markdown("##### Data")
    st.markdown(
        "- **Dataset:** [IBM AML Stateful Behavioral Fraud Dataset]"
        "(https://www.kaggle.com/datasets/vai136/ibm-aml-stateful-behavioral-fraud-dataset) on Kaggle, "
        "built from the HI-Small and LI-Small files of the IBM Transactions for Anti Money Laundering (AML) "
        "dataset by Erik Altman. Licence: CDLA-Sharing 1.0. Cite both datasets when reusing this work.\n"
        "- **Size:** 12,002,394 transactions, 8,742 with the positive label (`is_fraud`, which the dataset "
        "specification describes as laundering). The dataset's own statistics file says 8,746; the files "
        "contain 8,742.\n"
        f"- **Training:** {meta['n_train_rows']:,} rows, {meta['n_train_fraud']:,} positive."
    )

    st.markdown("##### What the data audit found")
    st.markdown(
        "1. `month` is 9 in every row, so it was dropped.\n"
        "2. The dataset's \"chronological\" protocol is a split by **hour of day** (train hours 0-16, "
        "validation 16-20, test 20-23), not by date. There is no full timestamp.\n"
        "3. There is **no account identifier**, so we cannot group rows by account or rule out the same "
        "account appearing in training and evaluation data.\n"
        "4. Hour 0 holds about 1.5 million rows, roughly three times any other hour. "
        "We could not explain this.\n"
        "5. `is_new_bank` is 1 for 93.2% of legitimate training rows. We could not explain this either.\n"
        "6. `spending_deviation_score` has extreme values (training range about -7.1e7 to 2.0e8) and is 0 "
        "for senders with 0 or 1 previous transactions, so it carries no information for new senders.\n"
        "7. Wire and Reinvestment transactions have no positive cases in any split."
    )

    st.markdown("##### Model")
    st.markdown(
        f"LightGBM classifier on {len(meta['feature_columns'])} inputs, trained on the random (benchmark) "
        "split. Behavioural inputs such as the sender's number of previous transactions come ready-made "
        "with the dataset, so the API expects them as inputs; it does not compute account history."
    )
    with st.expander("Settings and input list"):
        st.json(meta["params"])
        st.write(", ".join(meta["feature_columns"]))

    st.markdown("##### Limitations")
    top5000 = perf_channels["test"]["caught_in_top_alerts"]["5000"]
    st.markdown(
        "- **Synthetic data.** Results come from a simulator and should not be read as performance on a real bank.\n"
        f"- **Mostly ACH.** In the top 5,000 alerts of the hour-split test, {top5000['ACH']} confirmed cases "
        f"were ACH and {top5000['non_ACH']} were not.\n"
        "- **Possible account overlap.** The random split can put the same account in training and "
        "evaluation data, which may inflate those results. The hour split does not separate accounts either.\n"
        f"- **Harder late hours.** PR-AUC was {perf_hours['valid']['pr_auc']:.3f} on hours 16-20 and "
        f"{perf_hours['test']['pr_auc']:.3f} on hours 20-23. Part of this is a lower fraud rate and a harder period; "
        "the full cause is not established.\n"
        f"- **Few positive cases.** The test sets hold {perf_bench['test_fraud']:,} and "
        f"{perf_hours['test']['fraud']:,} positive cases, so results move by a few points between samples.\n"
        "- **Scores, not probabilities.** Score ranges were checked against outcomes on the random-split "
        "validation set only. Use the score to rank transactions and choose how many alerts your team can review.\n"
        "- **Optimistic validation numbers.** Model settings were chosen using validation data. The test sets "
        "were scored once, but the ACH breakdown was run after seeing test results, as a diagnosis."
    )
    st.caption("Built by Maria. Code, tests and notes live in the project repository.")