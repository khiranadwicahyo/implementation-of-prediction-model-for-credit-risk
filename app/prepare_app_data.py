"""Membuat file data ringkas untuk dashboard dari dataset mentah dan model v2.

Jalankan dari root repo setelah notebook v2 selesai dijalankan:
    python app/prepare_app_data.py data/loan_data_2007_2014.csv

Output: app/data/test_scores.csv dan app/data/app_meta.json (dataset mentah tidak dibutuhkan dashboard).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, CATEGORICAL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'app' / 'data'
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'data' / 'loan_data_2007_2014.csv', low_memory=False)
df = raw.copy()
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y')
SNAPSHOT = pd.Timestamp('2016-01-01')
cohort = df[(df['term'].str.strip() == '36 months')
            & (df['issue_d'] + pd.DateOffset(months=36) < SNAPSHOT)
            & (~df['loan_status'].str.startswith('Does not meet'))].copy()
BAD = ['Charged Off', 'Default', 'Late (31-120 days)']
cohort = cohort[cohort['loan_status'].isin(BAD + ['Fully Paid'])].copy()
cohort['target'] = cohort['loan_status'].isin(BAD).astype(int)

X_all = build_features(cohort)
is_train = cohort['issue_d'] < pd.Timestamp('2012-07-01')
X_train, X_test = X_all[is_train], X_all[~is_train]
y_test = cohort.loc[~is_train, 'target']

bundle = joblib.load(ROOT / 'outputs' / 'v2' / 'credit_risk_model_v2.joblib')
model = bundle['pipeline']
p_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, p_test)
assert abs(auc - 0.6562) < 5e-4, f'AUC tidak cocok dengan notebook: {auc:.4f}'

cash_test = cohort.loc[~is_train, 'total_pymnt'] - cohort.loc[~is_train, 'funded_amnt']
pd.DataFrame({'pd': p_test.round(5), 'target': y_test.values, 'cash': cash_test.round(2).values}) \
  .to_csv(OUT / 'test_scores.csv', index=False)

# --- statistik input (dari data latih) untuk default dan batas form ---
raw_train = cohort[is_train]
num_inputs = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
              'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mths_since_last_delinq', 'mths_since_last_record']
input_stats = {c: {'p01': float(raw_train[c].quantile(.01)), 'median': float(raw_train[c].median()),
                   'p99': float(raw_train[c].quantile(.99)), 'max': float(raw_train[c].max())} for c in num_inputs}
input_stats['credit_hist_years'] = {'median': float((X_train['credit_hist_months'] / 12).median()),
                                    'p99': float((X_train['credit_hist_months'] / 12).quantile(.99))}
# rata-rata suku bunga per sub-grade (membantu pengguna mengisi suku bunga yang realistis)
rate_by_subgrade = raw_train.groupby('sub_grade')['int_rate'].median().round(2).to_dict()
categories = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CATEGORICAL}

# --- ringkasan global ---
prep = model.named_steps['prep']
booster = model.named_steps['model'].get_booster()
Xt = prep.transform(X_test)
names = list(prep.get_feature_names_out())
contribs = booster.predict(xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)[:, :-1]
shap_df = pd.DataFrame(np.abs(contribs), columns=names)
grouped = {}
for c in names:
    base = next((cat for cat in CATEGORICAL if c.startswith(cat + '_')), c)
    grouped[base] = grouped.get(base, 0) + shap_df[c]
shap_global = pd.Series({k: v.mean() for k, v in grouped.items()}).sort_values(ascending=False)

dec = pd.DataFrame({'p': p_test, 'y': y_test.values})
dec['decile'] = pd.qcut(dec['p'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
dec_tbl = dec.groupby('decile').agg(n=('y', 'size'), avg_pd=('p', 'mean'), bad_rate=('y', 'mean')).reset_index()

fpr, tpr, _ = roc_curve(y_test, p_test)
idx = np.linspace(0, len(fpr) - 1, 200).astype(int)

metrics = json.loads((ROOT / 'outputs' / 'v2' / 'metrics_v2.json').read_text())
meta = {
    'thresholds': {'profit': float(bundle['threshold_profit']), 'reject_10pct': float(bundle['threshold_reject_10pct']),
                   'breakeven_pd': float(bundle['breakeven_pd'])},
    'score_quantiles': np.quantile(p_test, np.linspace(0, 1, 101)).round(5).tolist(),
    'base_bad_rate_test': float(y_test.mean()),
    'input_stats': input_stats,
    'rate_by_subgrade': rate_by_subgrade,
    'categories': categories,
    'shap_global': shap_global.round(4).to_dict(),
    'deciles': dec_tbl.round(4).to_dict(orient='records'),
    'roc': {'fpr': fpr[idx].round(4).tolist(), 'tpr': tpr[idx].round(4).tolist()},
    'metrics': metrics,
}
(OUT / 'app_meta.json').write_text(json.dumps(meta, indent=1, default=float))
print(f'AUC {auc:.4f} | test_scores.csv {len(p_test):,} baris | app_meta.json tersimpan')
