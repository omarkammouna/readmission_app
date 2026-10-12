"""
Builds the small files used by app.py.
Run once on your computer (or in Colab) if you want to regenerate them:
    python train_models.py path/to/diabetic_data.csv
It repeats the notebook's data preparation, Section 6 clustering (k = 2), and trains
the Logistic Regression and Gradient Boosting models with the same settings and seeds.
"""
import sys, json, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix, silhouette_score)
warnings.filterwarnings('ignore')
RS = 42
OUT = 'artifacts/'
path = sys.argv[1] if len(sys.argv) > 1 else 'diabetic_data.csv'

# ------------------------------------------------------------------ 1. data preparation (Section 5)
df = pd.read_csv(path, na_values=['?'], keep_default_na=False, low_memory=False)
d = df.drop(columns=['weight', 'payer_code']).copy()
d['medical_specialty'] = d['medical_specialty'].fillna('Missing')
d['race'] = d['race'].fillna('Unknown')
d = d[d['diag_1'].notnull()]
d = d[d['gender'] != 'Unknown/Invalid']
med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide',
            'glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
            'tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
            'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']
dominant = d[med_cols].apply(lambda s: s.value_counts(normalize=True).iloc[0])
informative = dominant[dominant <= 0.99].index.tolist()
d = d.drop(columns=dominant[dominant > 0.99].index.tolist())
d = d[~d['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
d = d.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')
d['readmitted_30'] = (d['readmitted'] == '<30').astype(int)

def group_diag(code):
    code = str(code)
    if code[0] in 'VE': return 'Other'
    try: v = float(code)
    except ValueError: return 'Other'
    if 390 <= v <= 459 or v == 785: return 'Circulatory'
    if 460 <= v <= 519 or v == 786: return 'Respiratory'
    if 520 <= v <= 579 or v == 787: return 'Digestive'
    if 250 <= v < 251: return 'Diabetes'
    if 800 <= v <= 999: return 'Injury'
    if 710 <= v <= 739: return 'Musculoskeletal'
    if 580 <= v <= 629 or v == 788: return 'Genitourinary'
    if 140 <= v <= 239: return 'Neoplasms'
    return 'Other'
d['diag_1_group'] = d['diag_1'].apply(group_diag)
d['service_utilization'] = d[['number_outpatient','number_emergency','number_inpatient']].sum(axis=1)
d['age_numeric'] = d['age'].map({f'[{i*10}-{i*10+10})': i*10+5 for i in range(10)})
d['discharge_group'] = d['discharge_disposition_id'].apply(
    lambda c: 'Home' if c in [1,6,8] else ('Transferred' if c in [2,3,4,5,9,10,12,15,16,17,22,23,24,27,28,29,30] else 'Other/Unknown'))
d['admission_source_group'] = d['admission_source_id'].apply(
    lambda c: 'Referral' if c in [1,2,3] else ('Transfer' if c in [4,5,6,10,18,19,22,25,26] else ('Emergency' if c == 7 else 'Other/Unknown')))
d['admission_type_group'] = d['admission_type_id'].apply(
    lambda c: 'Emergency/Urgent' if c in [1,2,7] else ('Elective' if c == 3 else 'Other/Unknown'))
top = d['medical_specialty'].value_counts().nlargest(8).index.tolist()
d['medical_specialty_group'] = d['medical_specialty'].apply(lambda x: x if x in top else ('Missing' if x == 'Missing' else 'Other'))
for c in ['number_outpatient', 'number_emergency', 'number_inpatient', 'service_utilization']:
    d[c + '_log'] = np.log1p(d[c])

nominal = ['race','gender','diag_1_group','discharge_group','admission_source_group','admission_type_group',
           'medical_specialty_group','max_glu_serum','A1Cresult','change','diabetesMed'] + informative
redundant = ['encounter_id','patient_nbr','readmitted','age','diag_1','diag_2','diag_3','discharge_disposition_id',
             'admission_source_id','admission_type_id','medical_specialty','number_outpatient','number_emergency',
             'number_inpatient','service_utilization']
enc = pd.get_dummies(d, columns=nominal, drop_first=True).drop(columns=redundant)
print('data_encoded', enc.shape)

numeric_features = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_diagnoses',
                    'age_numeric','number_outpatient_log','number_emergency_log','number_inpatient_log',
                    'service_utilization_log']
y = enc['readmitted_30'].astype(int)
X = enc.drop(columns=['readmitted_30']).astype(float)
feature_names = X.columns.tolist()

# ------------------------------------------------------------------ 2. defaults for the "try a patient" page
cat_info = {}
for col in nominal:
    cats = sorted(d[col].astype(str).unique())
    mode = d[col].astype(str).value_counts().index[0]
    cat_info[col] = {'categories': cats, 'mode': mode,
                     'columns': {c: (f'{col}_{c}' if f'{col}_{c}' in feature_names else None) for c in cats}}
defaults = {'numeric_median': {c: float(X[c].median()) for c in numeric_features},
            'numeric_range': {c: [float(X[c].min()), float(X[c].max())] for c in numeric_features},
            'raw_counts': {c: float(d[c].median()) for c in ['time_in_hospital','num_lab_procedures','num_procedures',
                           'num_medications','number_diagnoses','number_outpatient','number_emergency','number_inpatient']},
            'cat': cat_info, 'feature_names': feature_names, 'numeric_features': numeric_features}

# ------------------------------------------------------------------ 3. classification models (Section 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)
cv = StratifiedKFold(3, shuffle=True, random_state=RS)
cv_oof = StratifiedKFold(5, shuffle=True, random_state=RS)

def scaled(est):
    pre = ColumnTransformer([('num', StandardScaler(), numeric_features)], remainder='passthrough')
    return Pipeline([('prep', pre), ('model', est)])

def scores_of(m, X_):
    return m.predict_proba(X_)[:, 1]

def tune_eval(name, pipe, params, n_iter):
    t0 = time.time()
    s = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring='average_precision', cv=cv,
                           random_state=RS, n_jobs=-1, refit=True).fit(X_train, y_train)
    m = s.best_estimator_
    oof = cross_val_predict(m, X_train, y_train, cv=cv_oof, method='predict_proba', n_jobs=-1)[:, 1]
    p, r, t = precision_recall_curve(y_train, oof)
    thr = float(t[np.argmax(2*p[:-1]*r[:-1] / (p[:-1]+r[:-1] + 1e-12))])
    sc = scores_of(m, X_test)
    pred = (sc >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(name, s.best_params_, f'PR-AUC {average_precision_score(y_test, sc):.4f}', f'({time.time()-t0:.0f}s)')
    # score -> observed readmission rate by decile of the TEST scores (honest absolute-risk lookup)
    edges = np.quantile(sc, np.linspace(0, 1, 11))[1:-1]
    bins = np.searchsorted(edges, sc, side='right')
    dec = [{'decile': int(b) + 1, 'n': int((bins == b).sum()), 'rate': float(y_test.values[bins == b].mean())} for b in range(10)]
    row = {'model': name, 'roc_auc': float(roc_auc_score(y_test, sc)), 'pr_auc': float(average_precision_score(y_test, sc)),
           'recall': float(recall_score(y_test, pred)), 'precision': float(precision_score(y_test, pred)),
           'f1': float(f1_score(y_test, pred)), 'threshold': thr, 'flagged_pct': float((tp + fp) / len(y_test) * 100),
           'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'best_params': {k: str(v) for k, v in s.best_params_.items()}}
    dist = {'edges': edges.tolist(), 'deciles': dec, 'quantiles': np.quantile(sc, np.linspace(0, 1, 101)).tolist()}
    return m, row, dist

logreg, lr_row, lr_dist = tune_eval('Logistic Regression',
    scaled(LogisticRegression(class_weight='balanced', max_iter=2000)),
    {'model__C': [0.001, 0.01, 0.1, 1, 10]}, 5)
hgb, hgb_row, hgb_dist = tune_eval('Gradient Boosting',
    HistGradientBoostingClassifier(class_weight='balanced', early_stopping=True, validation_fraction=0.15,
                                   n_iter_no_change=15, max_iter=400, random_state=RS),
    {'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [3, 4, 6, None], 'max_leaf_nodes': [15, 31],
     'min_samples_leaf': [20, 50, 100], 'l2_regularization': [0.0, 1.0, 10.0]}, 8)

# Logistic Regression exported as plain numbers (no pickle: works with any library version)
pre = logreg.named_steps['prep']; lrm = logreg.named_steps['model']
names = [n.split('__', 1)[1] for n in pre.get_feature_names_out()]
sc_ = pre.named_transformers_['num']
lr_json = {'intercept': float(lrm.intercept_[0]), 'coef': dict(zip(names, map(float, lrm.coef_[0]))),
           'mean': dict(zip(numeric_features, map(float, sc_.mean_))), 'scale': dict(zip(numeric_features, map(float, sc_.scale_))),
           'threshold': lr_row['threshold']}
joblib.dump({'model': hgb, 'feature_names': feature_names, 'threshold': hgb_row['threshold']}, OUT + 'model_hgb.joblib', compress=3)

# Results of the other four models come from the notebook run (same data, same split, same seed)
prev = [
 ('Random Forest',  0.6357, 0.1593, 0.3914, 0.1559, 0.2230, 0.4866, 262),
 ('SVM',            0.6255, 0.1496, 0.4121, 0.1517, 0.2218, 0.1642, 183),
 ('Decision Tree',  0.6166, 0.1470, 0.4121, 0.1500, 0.2200, 0.5664, 10),
 ('KNN',            0.5990, 0.1379, 0.5068, 0.1206, 0.1949, 0.0990, 510)]
P, N = int(y_test.sum()), len(y_test)
rows = [lr_row, hgb_row]
for n_, roc, pr, rec, prec, f1, thr, secs in prev:
    tp = rec * P
    rows.append({'model': n_, 'roc_auc': roc, 'pr_auc': pr, 'recall': rec, 'precision': prec, 'f1': f1,
                 'threshold': thr, 'flagged_pct': tp / prec / N * 100})
rows.append({'model': 'Baseline (always "no")', 'roc_auc': 0.5, 'pr_auc': float(y_test.mean()), 'recall': 0.0,
             'precision': None, 'f1': None, 'threshold': None, 'flagged_pct': 0.0})
results = {'models': rows, 'n_test': N, 'n_test_pos': P, 'pos_rate': float(y_test.mean()),
           'n_patients': int(len(enc)), 'n_features': len(feature_names), 'n_train': int(len(X_train))}

# ------------------------------------------------------------------ 4. clustering (Section 6, KMeans)
cluster_features = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_diagnoses',
                    'age_numeric','service_utilization_log']
Xc = pd.DataFrame(StandardScaler().fit_transform(d[cluster_features]), columns=cluster_features, index=d.index)
sil = []
for k in range(2, 11):
    lab = KMeans(k, random_state=RS, n_init=10).fit_predict(Xc)
    sil.append({'k': k, 'silhouette': float(silhouette_score(Xc, lab, sample_size=10000, random_state=RS))})
    print('k', k, round(sil[-1]['silhouette'], 4))
km = KMeans(2, random_state=RS, n_init=10).fit(Xc)
d['cluster'] = km.labels_
hi = d.groupby('cluster')['time_in_hospital'].mean().idxmax()
name = {c: ('Long-stay / higher-intensity' if c == hi else 'Short-stay / lower-intensity') for c in (0, 1)}
prof = d.groupby('cluster')[cluster_features].mean()
prof.index = [name[c] for c in prof.index]
rate = d.groupby('cluster')['readmitted_30'].agg(['mean', 'count'])
pca = PCA(2, random_state=RS); co = pca.fit_transform(Xc)
sample = np.random.RandomState(RS).choice(len(d), 3000, replace=False)
cluster_json = {'silhouette_by_k': sil, 'k_chosen': 2, 'profile': prof.round(2).to_dict('index'),
                'readmission': {name[c]: {'rate': float(r['mean'] * 100), 'n': int(r['count'])} for c, r in rate.iterrows()},
                'overall_rate': float(d['readmitted_30'].mean() * 100),
                'pca_var': [float(v) for v in pca.explained_variance_ratio_],
                'pca_sample': [{'PC1': float(co[i, 0]), 'PC2': float(co[i, 1]), 'Cluster': name[int(d['cluster'].iloc[i])]} for i in sample]}

# ------------------------------------------------------------------ 5. save
for fname, obj in [('defaults.json', defaults), ('logreg.json', lr_json), ('results.json', results),
                   ('dist.json', {'Logistic Regression': lr_dist, 'Gradient Boosting': hgb_dist}), ('cluster.json', cluster_json)]:
    json.dump(obj, open(OUT + fname, 'w'))
print('saved to', OUT)
