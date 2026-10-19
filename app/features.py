"""Transformasi fitur untuk model credit risk v2.

`build_features` identik dengan fungsi di notebook `src code/credit_risk_model_v2.ipynb`,
sehingga input dari dashboard diolah persis sama seperti data latih.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ['home_ownership', 'verification_status', 'purpose']

SUB_GRADES = [f'{g}{n}' for g in 'ABCDEFG' for n in range(1, 6)]

EMP_LENGTH_OPTIONS = ['< 1 year', '1 year'] + [f'{i} years' for i in range(2, 10)] + ['10+ years']


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    out['loan_amnt'] = d['loan_amnt']
    out['int_rate'] = d['int_rate']
    out['installment'] = d['installment']
    out['sub_grade_num'] = d['sub_grade'].map({f'{g}{n}': i * 5 + n for i, g in enumerate('ABCDEFG') for n in range(1, 6)})

    emp = d['emp_length'].str.extract(r'(\d+)')[0].astype(float)
    emp[d['emp_length'].str.strip().eq('< 1 year').fillna(False).astype(bool)] = 0
    out['emp_length_yrs'] = emp

    out['log_annual_inc'] = np.log1p(d['annual_inc'])
    out['loan_to_income'] = d['loan_amnt'] / d['annual_inc'].replace(0, np.nan)
    out['installment_to_income'] = d['installment'] * 12 / d['annual_inc'].replace(0, np.nan)
    out['dti'] = d['dti']

    for c in ['delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_util', 'total_acc',
              'collections_12_mths_ex_med', 'acc_now_delinq']:
        out[c] = d[c]
    out['log_revol_bal'] = np.log1p(d['revol_bal'])

    for c, flag in [('mths_since_last_delinq', 'ever_delinq'), ('mths_since_last_record', 'ever_public_record')]:
        out[flag] = d[c].notna().astype(int)
        out[c] = d[c].clip(upper=120).fillna(120)

    ecl = pd.to_datetime(d['earliest_cr_line'], format='%b-%y')
    ecl = ecl.where(ecl <= d['issue_d'], ecl - pd.DateOffset(years=100))
    out['credit_hist_months'] = (d['issue_d'] - ecl).dt.days / 30.44

    out['home_ownership'] = d['home_ownership'].replace({'NONE': 'OTHER', 'ANY': 'OTHER'})
    out['verification_status'] = d['verification_status']
    out['purpose'] = d['purpose']
    return out


def installment_36m(loan_amnt: float, int_rate_pct: float) -> float:
    """Cicilan bulanan pinjaman 36 bulan (anuitas)."""
    r = int_rate_pct / 100 / 12
    if r == 0:
        return loan_amnt / 36
    return loan_amnt * r / (1 - (1 + r) ** -36)


def raw_row_from_inputs(inp: dict) -> pd.DataFrame:
    """Ubah input form dashboard menjadi satu baris dengan format kolom asli Lending Club."""
    issue_d = pd.Timestamp('2012-12-01')
    ecl = issue_d - pd.DateOffset(months=int(round(inp['credit_hist_years'] * 12)))
    row = {
        'loan_amnt': inp['loan_amnt'],
        'int_rate': inp['int_rate'],
        'installment': installment_36m(inp['loan_amnt'], inp['int_rate']),
        'sub_grade': inp['sub_grade'],
        'emp_length': inp['emp_length'],
        'annual_inc': inp['annual_inc'],
        'dti': inp['dti'],
        'delinq_2yrs': inp['delinq_2yrs'],
        'inq_last_6mths': inp['inq_last_6mths'],
        'open_acc': inp['open_acc'],
        'pub_rec': inp['pub_rec'],
        'revol_bal': inp['revol_bal'],
        'revol_util': inp['revol_util'],
        'total_acc': inp['total_acc'],
        'collections_12_mths_ex_med': 0.0,
        'acc_now_delinq': 0.0,
        'mths_since_last_delinq': inp['mths_since_last_delinq'] if inp['ever_delinq'] else np.nan,
        'mths_since_last_record': inp['mths_since_last_record'] if inp['ever_public_record'] else np.nan,
        'earliest_cr_line': ecl.strftime('%b-%y'),
        'issue_d': issue_d,
        'home_ownership': inp['home_ownership'],
        'verification_status': inp['verification_status'],
        'purpose': inp['purpose'],
    }
    d = pd.DataFrame([row])
    d['emp_length'] = d['emp_length'].astype(object)
    return d
