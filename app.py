"""
Diabetes 30-day readmission: project demo (AI & Advanced Machine Learning group project).
Student demo, NOT a medical tool.
Run locally:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st

TEAM = "Group project: AI and Advanced Machine Learning"   # <- put your team's names here

st.set_page_config(page_title="Diabetes readmission project", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------------------------------- data
@st.cache_data
def load_json(name):
    with open(f"artifacts/{name}.json") as f:
        return json.load(f)

@st.cache_resource
def load_hgb():
    """Gradient Boosting is stored with joblib; if the library version differs, the app still works without it."""
    try:
        import joblib
        return joblib.load("artifacts/model_hgb.joblib")
    except Exception:
        return None

R, D, L, DIST, C = (load_json(n) for n in ["results", "defaults", "logreg", "dist", "cluster"])
HGB = load_hgb()

PAGES = ["Home", "Data and preparation", "Clustering", "Classification models", "Try a patient"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(TEAM)
st.sidebar.warning("Student demo. Not medical advice. Do not use it for real patients.")


# ---------------------------------------------------------------------------------------- helpers
def models_table():
    rows = []
    for m in R["models"]:
        rows.append({
            "Model": m["model"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
            "Recall": m["recall"], "Precision": m["precision"], "F1": m["f1"],
            "Patients flagged (%)": m["flagged_pct"],
        })
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def build_row(inp):
    """Turn the form answers into the 71-column table the models were trained on."""
    row = {f: 0.0 for f in D["feature_names"]}
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]:
        row[c] = float(inp[c])
    row["age_numeric"] = float(inp["age_numeric"])
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        row[c + "_log"] = float(np.log1p(inp[c]))
    row["service_utilization_log"] = float(np.log1p(inp["number_outpatient"] + inp["number_emergency"] + inp["number_inpatient"]))
    for col, info in D["cat"].items():            # categories not on the form use the most common value
        label = inp.get(col, info["mode"])
        target = info["columns"].get(label)
        if target:
            row[target] = 1.0
    return row


def lr_parts(row):
    """Logistic Regression by hand: returns the score and the contribution of each feature (scaled)."""
    z = L["intercept"]
    xs = {}
    for f, coef in L["coef"].items():
        x = row[f]
        if f in L["mean"]:
            x = (x - L["mean"][f]) / L["scale"][f]
        xs[f] = x
        z += coef * x
    return 1 / (1 + np.exp(-z)), xs


def friendly(name):
    nice = {"time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab tests", "num_procedures": "Procedures",
            "num_medications": "Medications", "number_diagnoses": "Number of diagnoses", "age_numeric": "Age",
            "number_outpatient_log": "Prior outpatient visits", "number_emergency_log": "Prior emergency visits",
            "number_inpatient_log": "Prior inpatient stays", "service_utilization_log": "Total prior visits"}
    if name in nice:
        return nice[name]
    for col in sorted(D["cat"], key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{col.replace('_', ' ')}: {name[len(col) + 1:]}"
    return name


# ---------------------------------------------------------------------------------------- pages
if page == "Home":
    st.title("Predicting 30-day hospital readmission of diabetic patients")
    st.caption(TEAM)
    st.write(
        "Hospitals want to know which diabetic patients will come back within 30 days of leaving. "
        "In this project we **cleaned a real hospital dataset**, **grouped similar patients** (clustering), "
        "and **trained six models** to predict early readmission (classification)."
    )
    a, b, c, d = st.columns(4)
    best = models_table().iloc[0]
    a.metric("Patients after cleaning", f"{R['n_patients']:,}")
    b.metric("Readmitted within 30 days", f"{R['pos_rate'] * 100:.1f}%")
    c.metric("Best ROC-AUC", f"{best['ROC-AUC']:.3f}", help="0.5 = random guessing, 1 = perfect")
    d.metric("Best PR-AUC", f"{best['PR-AUC']:.3f}", help=f"A random model scores about {R['pos_rate']:.2f}")
    st.subheader("How the project is organised")
    st.markdown(
        "1. **Data and preparation:** explore, clean and encode the data.\n"
        "2. **Clustering:** find two types of patients with KMeans.\n"
        "3. **Classification models:** compare six models fairly on patients they never saw.\n"
        "4. **Try a patient:** enter a patient profile and see the model's risk score."
    )
    st.info(
        f"**Honest summary:** all models are only modestly better than random guessing "
        f"(ROC-AUC about 0.60 to 0.64, where 0.5 is random). Early readmission is hard to predict from these data. "
        f"The models can help prioritise patients for follow-up, but they cannot decide about one patient."
    )
    st.caption("Use the menu on the left to explore each part.")

elif page == "Data and preparation":
    st.title("Data and preparation")
    st.write(
        "The dataset has **101,766 hospital encounters** of diabetic patients (US hospitals, 1999 to 2008), with 50 columns "
        "about the patient, the stay, the medications and the outcome."
    )
    st.markdown(
        "**Target:** `readmitted_30` is 1 if the patient came back within 30 days, otherwise 0. "
        f"Only **{R['pos_rate'] * 100:.1f}%** are 1, so the data is strongly imbalanced."
    )
    st.subheader("What we did to the data")
    steps = [
        ("Fix the gaps", "Dropped two almost-empty columns (`weight`, `payer_code`); missing race and specialty became their own category."),
        ("Drop useless medicines", "15 of 23 drug columns have the same value for more than 99% of patients, so they were removed."),
        ("Remove patients who died or went to hospice", "They cannot be readmitted, so they would distort the target."),
        ("One visit per patient", f"Kept each patient's first encounter only: 101,766 rows became {R['n_patients']:,} patients."),
        ("Create new columns", "The 0/1 target, nine diagnosis groups, total prior visits, age as a number, grouped admission and discharge codes."),
        ("Tame huge counts", "`log(1 + x)` on visit counts so a few frequent visitors do not dominate."),
        ("Turn words into numbers", f"One-hot encoding gives a table of {R['n_features']} feature columns plus the target."),
        ("Scale (inside the models)", "Distance-based models scale the numeric columns inside a pipeline, learned from training data only, so the test set stays truly unseen."),
    ]
    for i, (t, txt) in enumerate(steps, 1):
        st.markdown(f"**{i}. {t}.** {txt}")
    st.subheader("How the models are tested")
    st.write(
        f"The cleaned data is split **80% / 20%** (stratified). The models learn from {R['n_train']:,} patients and are "
        f"graded once on the hidden {R['n_test']:,} patients, of whom {R['n_test_pos']:,} were readmitted early."
    )

elif page == "Clustering":
    st.title("Clustering: two types of patients")
    st.write(
        "KMeans groups patients by similarity **without ever seeing the target**. It uses seven columns: days in hospital, "
        "lab tests, procedures, medications, number of diagnoses, age and prior visits. Afterwards we check whether the groups "
        "differ in readmission."
    )
    st.subheader("How many groups?")
    sil = pd.DataFrame(C["silhouette_by_k"]).set_index("k")
    st.line_chart(sil, y="silhouette")
    st.caption(
        f"The silhouette score is highest at k = {C['k_chosen']} ({sil.loc[C['k_chosen'], 'silhouette']:.3f}). "
        "Values around 0.2 mean the groups overlap a lot: patients form a continuum, not separate islands."
    )
    st.subheader("What are the two groups like?")
    prof = pd.DataFrame(C["profile"]).rename(index={
        "time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab tests", "num_procedures": "Procedures",
        "num_medications": "Medications", "number_diagnoses": "Diagnoses", "age_numeric": "Age (years)",
        "service_utilization_log": "Prior visits (log scale)"})
    st.dataframe(prof)
    st.subheader("Do they differ in 30-day readmission?")
    rates = pd.DataFrame(C["readmission"]).T.rename(columns={"rate": "Readmitted within 30 days (%)", "n": "Patients"})
    left, right = st.columns([1, 1])
    left.bar_chart(rates[["Readmitted within 30 days (%)"]])
    right.dataframe(rates.round({"Readmitted within 30 days (%)": 1}))
    st.caption(
        f"Overall rate: {C['overall_rate']:.1f}%. The long-stay, higher-intensity group comes back more often, "
        "and a chi-square test (in the notebook) shows the gap is not luck, but the effect is small to moderate."
    )
    st.subheader("A picture of the groups (PCA)")
    pts = pd.DataFrame(C["pca_sample"])
    st.scatter_chart(pts, x="PC1", y="PC2", color="Cluster")
    st.caption(
        f"The two axes capture {sum(C['pca_var']) * 100:.0f}% of the variation, so the picture hides part of the structure. "
        "A random sample of 3,000 patients is shown."
    )

elif page == "Classification models":
    st.title("Classification: six models, one fair test")
    tbl = models_table()
    st.write(
        "Every model was tuned with cross-validation on the training data, the decision cutoff was chosen on training data "
        "only, and everything was graded once on the same hidden test patients."
    )
    show = tbl.copy()
    for c in ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]:
        show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    show["Patients flagged (%)"] = show["Patients flagged (%)"].map(lambda v: f"{v:.1f}")
    st.dataframe(show, hide_index=True)
    st.caption(f"Recall, precision and F1 are at each model's tuned cutoff. A random model has PR-AUC of about {R['pos_rate']:.3f}.")
    st.bar_chart(tbl.set_index("Model")[["PR-AUC"]])

    st.subheader("The top 3")
    top = tbl[~tbl["Model"].str.startswith("Baseline")].head(3)
    texts = {
        "Gradient Boosting": "Builds small trees one after another, each correcting the previous ones. The most selective model: it flags about 1 in 5 patients, and about 16% of those are really readmitted (the overall rate is 9%).",
        "Random Forest": "200 decision trees voting together. A very stable model and a close second.",
        "Logistic Regression": "A simple linear model, fast and easy to explain. It catches more readmissions but gives more false alarms. That a simple model comes this close suggests the signal in these data is mostly linear and modest.",
    }
    cols = st.columns(3)
    for col, (_, r) in zip(cols, top.iterrows()):
        with col:
            st.markdown(f"#### {r['Model']}")
            st.metric("PR-AUC", f"{r['PR-AUC']:.3f}")
            st.write(f"ROC-AUC **{r['ROC-AUC']:.3f}**, recall **{r['Recall']:.2f}**, precision **{r['Precision']:.2f}**")
            st.write(texts.get(r["Model"], ""))
    st.warning(
        "The top three are very close (about 0.01 in PR-AUC) and were compared on a single split without a significance test. "
        "We call the best one *slightly* better, not clearly better."
    )

    st.subheader("What the confusion matrix says")
    cm_rows = [m for m in R["models"] if "tp" in m]
    st.dataframe(pd.DataFrame([{
        "Model": m["model"], "Caught (TP)": m["tp"], "Missed (FN)": m["fn"],
        "False alarms (FP)": m["fp"], "Correct 'no' (TN)": m["tn"]} for m in cm_rows]), hide_index=True)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy is not used:** 91% of patients are not readmitted, so a model that always says 'no' is 91% accurate and useless.\n"
            "- **Recall:** of the patients who really came back, the share we caught.\n"
            "- **Precision:** of the patients we flagged, the share who really came back.\n"
            "- **F1:** one number that balances recall and precision.\n"
            "- **ROC-AUC:** the chance that a patient who came back gets a higher score than one who did not (0.5 = random).\n"
            "- **PR-AUC:** summarises precision and recall; more informative when the positive class is rare.\n"
            "- **Cutoff:** the score above which a patient is flagged. 0.5 is a poor choice here, so we tuned it."
        )

else:  # Try a patient
    st.title("Try a patient")
    st.warning("Student demo only. The score comes from a model that is only modestly better than random. Never use it for real decisions.")

    model_choice = "Logistic Regression"
    if HGB is not None:
        model_choice = st.radio("Model", ["Gradient Boosting (best score)", "Logistic Regression (explains why)"], horizontal=True)
        model_choice = model_choice.split(" (")[0]
    else:
        st.info("Gradient Boosting could not be loaded in this environment, so Logistic Regression is used.")

    bands = [f"[{i * 10}-{i * 10 + 10})" for i in range(10)]
    cat = D["cat"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**The stay**")
        age_band = st.select_slider("Age band", options=bands, value="[60-70)")
        t_hosp = st.slider("Days in hospital", 1, 14, 3)
        n_lab = st.slider("Lab tests", 1, 120, 44)
        n_proc = st.slider("Procedures", 0, 6, 1)
        n_med = st.slider("Medications", 1, 80, 14)
        n_diag = st.slider("Number of diagnoses", 1, 16, 8)
    with c2:
        st.markdown("**Prior hospital use (last year)**")
        n_out = st.number_input("Outpatient visits", 0, 40, 0)
        n_emg = st.number_input("Emergency visits", 0, 40, 0)
        n_inp = st.number_input("Inpatient stays", 0, 20, 0)
        st.markdown("**Arrival and discharge**")
        discharge = st.selectbox("Discharge", cat["discharge_group"]["categories"], index=cat["discharge_group"]["categories"].index("Home"))
        adm_type = st.selectbox("Admission type", cat["admission_type_group"]["categories"], index=cat["admission_type_group"]["categories"].index("Emergency/Urgent"))
        adm_src = st.selectbox("Admission source", cat["admission_source_group"]["categories"], index=cat["admission_source_group"]["categories"].index("Emergency"))
    with c3:
        st.markdown("**Diagnosis and treatment**")
        diag = st.selectbox("Primary diagnosis group", cat["diag_1_group"]["categories"], index=cat["diag_1_group"]["categories"].index("Circulatory"))
        gender = st.selectbox("Gender", cat["gender"]["categories"])
        insulin = st.selectbox("Insulin", cat["insulin"]["categories"], index=cat["insulin"]["categories"].index("No"))
        dmed = st.selectbox("Diabetes medication", cat["diabetesMed"]["categories"], index=cat["diabetesMed"]["categories"].index("Yes"))
        a1c = st.selectbox("HbA1c test result", cat["A1Cresult"]["categories"], index=cat["A1Cresult"]["categories"].index("None"))
        change = st.selectbox("Medication changed during stay", cat["change"]["categories"], index=cat["change"]["categories"].index("No"))
    st.caption("Everything not on this form (race, specialty, other drugs and so on) is set to the most common value.")

    inp = {"time_in_hospital": t_hosp, "num_lab_procedures": n_lab, "num_procedures": n_proc, "num_medications": n_med,
           "number_diagnoses": n_diag, "age_numeric": bands.index(age_band) * 10 + 5,
           "number_outpatient": n_out, "number_emergency": n_emg, "number_inpatient": n_inp,
           "discharge_group": discharge, "admission_type_group": adm_type, "admission_source_group": adm_src,
           "diag_1_group": diag, "gender": gender, "insulin": insulin, "diabetesMed": dmed, "A1Cresult": a1c, "change": change}
    row = build_row(inp)

    if model_choice == "Gradient Boosting":
        score = float(HGB["model"].predict_proba(pd.DataFrame([row])[HGB["feature_names"]])[:, 1][0])
        thr = HGB["threshold"]
    else:
        score, _ = lr_parts(row)
        thr = L["threshold"]
    dist = DIST[model_choice]
    pct = float(np.interp(score, dist["quantiles"], np.linspace(0, 100, 101)))
    dec = int(np.searchsorted(dist["edges"], score, side="right"))
    dec_rate = dist["deciles"][dec]["rate"] * 100

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Risk score", f"{score:.2f}", help="A ranking score from 0 to 1. It is NOT a probability.")
    b.metric("Higher than", f"{pct:.0f}% of test patients")
    c.metric("Alert cutoff", f"{thr:.2f}")
    if score >= thr:
        st.error("**Flagged as higher risk:** the score is at or above the model's alert cutoff.")
    else:
        st.success("**Not flagged:** the score is below the model's alert cutoff.")
    st.write(
        f"In our hidden test set, patients with a score in this range (band {dec + 1} of 10) were readmitted within 30 days "
        f"**{dec_rate:.1f}%** of the time, compared with **{R['pos_rate'] * 100:.1f}%** overall."
    )
    st.caption(
        "Why is the score so high for an ordinary patient? The models were trained to give extra weight to the rare "
        "'readmitted' class, which pushes all scores upward. The score is only useful for ranking patients; "
        "the real readmission rate for the score band is the percentage shown above."
    )
    bars = pd.DataFrame({"Readmitted within 30 days (%)": [x["rate"] * 100 for x in dist["deciles"]]},
                        index=[f"{x['decile']:02d}" for x in dist["deciles"]])
    bars.index.name = "Score band (1 = lowest scores, 10 = highest)"
    st.bar_chart(bars)

    if model_choice == "Logistic Regression":
        st.subheader("Why this score? (compared with a typical patient)")
        typical = build_row({"time_in_hospital": D["raw_counts"]["time_in_hospital"], "num_lab_procedures": D["raw_counts"]["num_lab_procedures"],
                             "num_procedures": D["raw_counts"]["num_procedures"], "num_medications": D["raw_counts"]["num_medications"],
                             "number_diagnoses": D["raw_counts"]["number_diagnoses"], "age_numeric": 65,
                             "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0})
        _, xs_t = lr_parts(typical)
        _, xs_p = lr_parts(row)
        contrib = {f: L["coef"][f] * (xs_p[f] - xs_t[f]) for f in L["coef"]}
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        top = [(k, v) for k, v in top if abs(v) > 1e-6]
        if top:
            cdf = pd.DataFrame({"Effect on risk (log-odds)": [v for _, v in top]}, index=[friendly(k) for k, _ in top])
            st.bar_chart(cdf)
            st.caption("Positive bars push the score up, negative bars push it down, relative to a typical patient "
                       "(65 years, no prior visits, most common categories).")
        else:
            st.caption("This patient looks like the typical patient, so there is nothing to explain.")
