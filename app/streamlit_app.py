"""Dashboard Credit Risk Scoring (model v2) — Streamlit."""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
from features import (CATEGORICAL, EMP_LENGTH_OPTIONS, SUB_GRADES, build_features,  # noqa: E402
                      installment_36m, raw_row_from_inputs)

st.set_page_config(page_title='Credit Risk Scoring', page_icon='📊', layout='wide')

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8984'
REPO_URL = 'https://github.com/khiranadwicahyo/implementation-of-prediction-model-for-credit-risk'

# ---------------------------------------------------------------- label & deskripsi
PURPOSE_ID = {
    'debt_consolidation': 'Konsolidasi utang', 'credit_card': 'Pelunasan kartu kredit',
    'home_improvement': 'Renovasi rumah', 'major_purchase': 'Pembelian besar', 'small_business': 'Usaha kecil',
    'car': 'Kendaraan', 'wedding': 'Pernikahan', 'medical': 'Biaya medis', 'moving': 'Pindahan',
    'house': 'Pembelian rumah', 'vacation': 'Liburan', 'educational': 'Pendidikan',
    'renewable_energy': 'Energi terbarukan', 'other': 'Lainnya',
}
HOME_ID = {'MORTGAGE': 'KPR (mortgage)', 'RENT': 'Sewa', 'OWN': 'Milik sendiri', 'OTHER': 'Lainnya'}
VERIF_ID = {'Verified': 'Terverifikasi', 'Source Verified': 'Sumber pendapatan terverifikasi',
            'Not Verified': 'Tidak terverifikasi'}
EMP_ID = {'< 1 year': '< 1 tahun', '1 year': '1 tahun', '10+ years': '10 tahun atau lebih',
          **{f'{i} years': f'{i} tahun' for i in range(2, 10)}}

# (fitur model, kolom asal, label singkat, deskripsi, pengolahan)
FEATURE_DOCS = [
    ('loan_amnt', 'loan_amnt', 'Jumlah pinjaman', 'Jumlah pinjaman yang diajukan peminjam (USD).', 'Dipakai langsung.'),
    ('int_rate', 'int_rate', 'Suku bunga', 'Suku bunga tahunan pinjaman (%) yang ditetapkan Lending Club berdasarkan penilaian risiko awal.', 'Dipakai langsung.'),
    ('installment', 'installment', 'Cicilan bulanan', 'Cicilan bulanan yang harus dibayar jika pinjaman dicairkan (USD).', 'Dipakai langsung. Di kalkulator dihitung otomatis dari jumlah pinjaman dan suku bunga (tenor 36 bulan).'),
    ('sub_grade_num', 'sub_grade', 'Sub-grade', 'Peringkat risiko dari Lending Club, dari A1 (paling aman) sampai G5 (paling berisiko).', 'Diubah menjadi angka urut 1–35.'),
    ('emp_length_yrs', 'emp_length', 'Lama bekerja', 'Lama bekerja peminjam dalam tahun.', '"< 1 year" menjadi 0, "10+ years" menjadi 10.'),
    ('log_annual_inc', 'annual_inc', 'Pendapatan tahunan', 'Pendapatan tahunan yang dilaporkan peminjam saat mendaftar (USD).', 'Transformasi log(1 + x) untuk meredam nilai ekstrem.'),
    ('loan_to_income', 'loan_amnt, annual_inc', 'Rasio pinjaman/pendapatan', 'Besar pinjaman dibanding pendapatan setahun.', 'Fitur turunan: jumlah pinjaman ÷ pendapatan tahunan.'),
    ('installment_to_income', 'installment, annual_inc', 'Rasio cicilan/pendapatan', 'Porsi pendapatan tahunan yang terpakai untuk cicilan pinjaman ini.', 'Fitur turunan: cicilan × 12 ÷ pendapatan tahunan.'),
    ('dti', 'dti', 'Debt-to-income (DTI)', 'Total cicilan utang bulanan (di luar KPR dan pinjaman yang diajukan) dibagi pendapatan bulanan, dalam %.', 'Dipakai langsung.'),
    ('delinq_2yrs', 'delinq_2yrs', 'Tunggakan 2 tahun terakhir', 'Jumlah kejadian tunggakan lebih dari 30 hari dalam 2 tahun terakhir di catatan kredit.', 'Dipakai langsung.'),
    ('inq_last_6mths', 'inq_last_6mths', 'Inquiry kredit 6 bulan', 'Jumlah pengecekan kredit (inquiry) dalam 6 bulan terakhir, di luar kredit kendaraan dan KPR. Banyak inquiry menandakan peminjam sedang aktif mencari pinjaman.', 'Dipakai langsung.'),
    ('open_acc', 'open_acc', 'Fasilitas kredit aktif', 'Jumlah rekening atau fasilitas kredit yang masih aktif.', 'Dipakai langsung.'),
    ('pub_rec', 'pub_rec', 'Catatan publik negatif', 'Jumlah catatan publik negatif, misalnya kebangkrutan atau putusan pengadilan.', 'Dipakai langsung.'),
    ('revol_util', 'revol_util', 'Utilisasi kredit bergulir', 'Persentase limit kredit bergulir (misalnya kartu kredit) yang sedang terpakai.', 'Dipakai langsung; data kosong diisi median data latih.'),
    ('total_acc', 'total_acc', 'Total fasilitas kredit', 'Jumlah seluruh fasilitas kredit yang pernah tercatat.', 'Dipakai langsung.'),
    ('collections_12_mths_ex_med', 'collections_12_mths_ex_med', 'Penagihan 12 bulan', 'Jumlah penagihan dalam 12 bulan terakhir, di luar tagihan medis.', 'Dipakai langsung. Hampir selalu 0, sehingga di kalkulator diset 0.'),
    ('acc_now_delinq', 'acc_now_delinq', 'Rekening sedang menunggak', 'Jumlah rekening yang sedang menunggak saat pengajuan.', 'Dipakai langsung. Hampir selalu 0, sehingga di kalkulator diset 0.'),
    ('log_revol_bal', 'revol_bal', 'Saldo kredit bergulir', 'Total saldo kredit bergulir (USD).', 'Transformasi log(1 + x).'),
    ('ever_delinq', 'mths_since_last_delinq', 'Pernah menunggak', 'Penanda apakah peminjam pernah menunggak (1) atau tidak (0).', 'Dibuat dari kolom asal: terisi = pernah, kosong = tidak pernah.'),
    ('mths_since_last_delinq', 'mths_since_last_delinq', 'Bulan sejak tunggakan terakhir', 'Jumlah bulan sejak tunggakan terakhir.', 'Kosong ("tidak pernah") diisi 120 dan nilai dibatasi maksimal 120, bukan diisi 0 seperti di versi 1.'),
    ('ever_public_record', 'mths_since_last_record', 'Pernah punya catatan publik', 'Penanda apakah peminjam pernah memiliki catatan publik negatif.', 'Dibuat dari kolom asal: terisi = pernah, kosong = tidak pernah.'),
    ('mths_since_last_record', 'mths_since_last_record', 'Bulan sejak catatan publik terakhir', 'Jumlah bulan sejak catatan publik terakhir.', 'Kosong diisi 120 dan nilai dibatasi maksimal 120.'),
    ('credit_hist_months', 'earliest_cr_line, issue_d', 'Lama riwayat kredit', 'Lama riwayat kredit: jarak antara pembukaan fasilitas kredit pertama dan tanggal pencairan pinjaman.', 'Fitur turunan dalam bulan. Tahun dua digit (misal "65") dikoreksi agar tidak terbaca 2065.'),
    ('home_ownership', 'home_ownership', 'Kepemilikan rumah', 'Status kepemilikan tempat tinggal: KPR (mortgage), sewa, milik sendiri, atau lainnya.', 'One-hot encoding; NONE dan ANY digabung ke OTHER.'),
    ('verification_status', 'verification_status', 'Verifikasi pendapatan', 'Apakah pendapatan peminjam diverifikasi Lending Club: terverifikasi, sumber pendapatan terverifikasi, atau tidak terverifikasi.', 'One-hot encoding.'),
    ('purpose', 'purpose', 'Tujuan pinjaman', 'Tujuan pinjaman yang dipilih peminjam, misalnya konsolidasi utang atau pelunasan kartu kredit.', 'One-hot encoding; kategori dengan kurang dari 50 data digabung.'),
]
FEATURE_LABEL = {f: lab for f, _, lab, _, _ in FEATURE_DOCS}

EXCLUDED = [
    ('Data pembayaran & penagihan', 'total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee, out_prncp, out_prncp_inv, last_pymnt_amnt, last_pymnt_d, next_pymnt_d, last_credit_pull_d',
     'Baru tercatat setelah pinjaman berjalan. Ini sumber utama data leakage di versi 1 (misalnya recoveries hanya terisi pada pinjaman macet).'),
    ('Identitas', 'Unnamed: 0, id, member_id, url', 'Nomor urut dan ID tidak menggambarkan perilaku peminjam.'),
    ('Kosong sebelum 2012', 'tot_cur_bal, tot_coll_amt, total_rev_hi_lim, mths_since_last_major_derog', 'Hampir seluruhnya kosong pada periode data latih; pola kosongnya berubah menurut waktu.'),
    ('Konstan / berubah definisi', 'policy_code, application_type, initial_list_status, term, pymnt_plan', 'Nilainya sama untuk semua pinjaman, atau baru muncul di tengah periode (initial_list_status = w mulai 2012). Term konstan karena model khusus tenor 36 bulan.'),
    ('Kolom kosong total', 'annual_inc_joint, dti_joint, verification_status_joint, open_acc_6m, il_util, all_util, dll.', '100% kosong pada dataset ini.'),
    ('Teks bebas & wilayah', 'desc, title, emp_title, zip_code, addr_state', 'Di luar cakupan versi ini (butuh pengolahan teks).'),
    ('Redundan', 'grade, funded_amnt, funded_amnt_inv', 'Sudah terwakili oleh sub_grade dan loan_amnt.'),
]


# ---------------------------------------------------------------- data & model
@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'outputs' / 'v2' / 'credit_risk_model_v2.joblib')


@st.cache_data
def load_meta():
    return json.loads((APP_DIR / 'data' / 'app_meta.json').read_text())


@st.cache_data
def load_scores():
    return pd.read_csv(APP_DIR / 'data' / 'test_scores.csv')


bundle, meta, scores = load_model(), load_meta(), load_scores()
model = bundle['pipeline']
TH = meta['thresholds']
BASE_BAD = meta['base_bad_rate_test']


def pct(x, d=1):
    return f'{x * 100:.{d}f}%'.replace('.', ',')


def num(x, d=2):
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def usd(x):
    return 'USD ' + f'{x:,.0f}'.replace(',', '.')


def risk_band(p):
    if p >= TH['breakeven_pd']:
        return 'Sangat tinggi', 'red', ':material/error:', 'Di atas break-even PD: secara rata-rata, pinjaman seperti ini merugi.'
    if p >= TH['reject_10pct']:
        return 'Tinggi', 'orange', ':material/warning:', 'Masuk kelompok 10% skor tertinggi (cutoff dari data latih).'
    if p >= 0.10:
        return 'Sedang', 'blue', ':material/info:', 'Di sekitar rata-rata portofolio.'
    return 'Rendah', 'green', ':material/check_circle:', 'Di bawah rata-rata portofolio.'


def explain(X_row: pd.DataFrame) -> pd.DataFrame:
    """Kontribusi SHAP per fitur asal (one-hot dijumlahkan kembali)."""
    prep = model.named_steps['prep']
    booster = model.named_steps['model'].get_booster()
    names = list(prep.get_feature_names_out())
    contrib = booster.predict(xgb.DMatrix(prep.transform(X_row), feature_names=names), pred_contribs=True)[0, :-1]
    agg = {}
    for n, v in zip(names, contrib):
        base = next((c for c in CATEGORICAL if n.startswith(c + '_')), n)
        agg[base] = agg.get(base, 0.0) + float(v)
    return pd.DataFrame({'fitur': list(agg), 'kontribusi': list(agg.values())})


# ================================================================ halaman 1: kalkulator
PRESETS = {
    'Profil risiko rendah': dict(loan_amnt=8000, sub_grade='A3', emp_length='10+ years', annual_inc=110000, dti=7.0,
                                 delinq_2yrs=0, inq_last_6mths=0, open_acc=9, pub_rec=0, revol_bal=6000, revol_util=18.0,
                                 total_acc=30, ever_delinq=False, ever_public_record=False, credit_hist_years=22.0,
                                 home_ownership='MORTGAGE', verification_status='Not Verified', purpose='credit_card'),
    'Profil rata-rata': None,  # diisi median data latih
    'Profil risiko tinggi': dict(loan_amnt=25000, sub_grade='F2', emp_length='< 1 year', annual_inc=32000, dti=27.0,
                                 delinq_2yrs=1, inq_last_6mths=4, open_acc=7, pub_rec=1, revol_bal=14000, revol_util=96.0,
                                 total_acc=12, ever_delinq=True, mths_since_last_delinq=10, ever_public_record=True,
                                 mths_since_last_record=30, credit_hist_years=5.0, home_ownership='RENT',
                                 verification_status='Verified', purpose='small_business'),
}


def median_profile():
    s = meta['input_stats']
    return dict(loan_amnt=int(round(s['loan_amnt']['median'], -2)), sub_grade='B3', emp_length='5 years',
                annual_inc=int(round(s['annual_inc']['median'], -3)), dti=float(round(s['dti']['median'], 1)),
                delinq_2yrs=0, inq_last_6mths=int(s['inq_last_6mths']['median']), open_acc=int(s['open_acc']['median']),
                pub_rec=0, revol_bal=int(round(s['revol_bal']['median'], -2)), revol_util=float(round(s['revol_util']['median'], 1)),
                total_acc=int(s['total_acc']['median']), ever_delinq=False, ever_public_record=False,
                credit_hist_years=float(round(s['credit_hist_years']['median'], 0)), home_ownership='MORTGAGE',
                verification_status='Not Verified', purpose='debt_consolidation')


def apply_profile(profile: dict):
    defaults = dict(mths_since_last_delinq=24, mths_since_last_record=60)
    for k, v in {**defaults, **profile}.items():
        st.session_state[f'in_{k}'] = v
    st.session_state['in_int_rate'] = meta['rate_by_subgrade'].get(profile['sub_grade'], 12.0)


def on_subgrade_change():
    st.session_state['in_int_rate'] = meta['rate_by_subgrade'].get(st.session_state['in_sub_grade'], 12.0)


def on_preset_change():
    name = st.session_state.get('preset')
    if name:
        apply_profile(PRESETS[name] or median_profile())


def page_calculator():
    if 'in_loan_amnt' not in st.session_state:
        apply_profile(median_profile())

    st.title('Kalkulator Risiko Kredit')
    st.caption('Isi data pengajuan pinjaman (tenor 36 bulan). Model memperkirakan **probabilitas gagal bayar (PD)** '
               'dan menunjukkan faktor yang paling memengaruhinya.')
    st.pills('Isi cepat dengan contoh:', list(PRESETS), key='preset', on_change=on_preset_change)

    s = meta['input_stats']
    left, right = st.columns([1.05, 1], gap='large')
    with left:
        with st.container(border=True):
            st.markdown('**Pinjaman**')
            c1, c2, c3 = st.columns(3)
            c1.number_input('Jumlah pinjaman (USD)', 500, 35000, step=500, key='in_loan_amnt')
            c2.selectbox('Sub-grade Lending Club', SUB_GRADES, key='in_sub_grade', on_change=on_subgrade_change,
                         help='A1 = paling aman, G5 = paling berisiko. Mengubah sub-grade otomatis mengisi suku bunga tipikalnya.')
            c3.number_input('Suku bunga (%/tahun)', 5.0, 26.0, step=0.1, format='%.2f', key='in_int_rate')
            c1, c2 = st.columns(2)
            c1.selectbox('Tujuan pinjaman', list(PURPOSE_ID), format_func=PURPOSE_ID.get, key='in_purpose')
            inst = installment_36m(st.session_state['in_loan_amnt'], st.session_state['in_int_rate'])
            c2.text_input('Cicilan bulanan (otomatis, 36 bulan)', value=f'USD {num(inst)}', disabled=True)

        with st.container(border=True):
            st.markdown('**Profil peminjam**')
            c1, c2, c3 = st.columns(3)
            c1.number_input('Pendapatan tahunan (USD)', 4000, 1_000_000, step=1000, key='in_annual_inc')
            c2.selectbox('Lama bekerja', EMP_LENGTH_OPTIONS, format_func=EMP_ID.get, key='in_emp_length')
            c3.selectbox('Kepemilikan rumah', list(HOME_ID), format_func=HOME_ID.get, key='in_home_ownership')
            c1, c2 = st.columns(2)
            c1.number_input('Debt-to-income / DTI (%)', 0.0, 35.0, step=0.5, key='in_dti',
                            help='Cicilan utang bulanan lain (di luar KPR dan pinjaman ini) ÷ pendapatan bulanan.')
            c2.selectbox('Verifikasi pendapatan', list(VERIF_ID), format_func=VERIF_ID.get, key='in_verification_status')

        with st.container(border=True):
            st.markdown('**Riwayat kredit**')
            c1, c2, c3 = st.columns(3)
            c1.number_input('Lama riwayat kredit (tahun)', 3.0, 60.0, step=1.0, key='in_credit_hist_years')
            c2.number_input('Inquiry kredit 6 bln terakhir', 0, 10, key='in_inq_last_6mths')
            c3.number_input('Utilisasi kredit bergulir (%)', 0.0, 120.0, step=1.0, key='in_revol_util')
            c1, c2, c3 = st.columns(3)
            c1.number_input('Saldo kredit bergulir (USD)', 0, 500_000, step=500, key='in_revol_bal')
            c2.number_input('Fasilitas kredit aktif', 1, 50, key='in_open_acc')
            c3.number_input('Total fasilitas kredit', 2, 90, key='in_total_acc')
            c1, c2, c3 = st.columns(3)
            c1.number_input('Tunggakan >30 hari (2 thn)', 0, 15, key='in_delinq_2yrs')
            c2.number_input('Catatan publik negatif', 0, 10, key='in_pub_rec')
            c1, c2 = st.columns(2)
            c1.toggle('Pernah menunggak', key='in_ever_delinq',
                      on_change=lambda: st.session_state.setdefault('in_mths_since_last_delinq', 24))
            if st.session_state['in_ever_delinq']:
                c1.number_input('Bulan sejak tunggakan terakhir', 0, 120, key='in_mths_since_last_delinq')
            c2.toggle('Pernah punya catatan publik', key='in_ever_public_record',
                      on_change=lambda: st.session_state.setdefault('in_mths_since_last_record', 60))
            if st.session_state['in_ever_public_record']:
                c2.number_input('Bulan sejak catatan publik terakhir', 0, 120, key='in_mths_since_last_record')

    inp = {k[3:]: v for k, v in st.session_state.items() if k.startswith('in_')}
    X_row = build_features(raw_row_from_inputs(inp))
    p = float(model.predict_proba(X_row)[0, 1])
    band, color, icon, band_text = risk_band(p)
    pctile = float(np.searchsorted(meta['score_quantiles'], p) - 1)
    pctile = min(max(pctile, 0), 100)
    decile = min(int(pctile // 10) + 1, 10)
    dec_bad = meta['deciles'][decile - 1]['bad_rate']

    with right:
        with st.container(border=True):
            st.markdown('**Hasil prediksi**')
            c1, c2 = st.columns([1, 1.2])
            c1.metric('Probabilitas gagal bayar (PD)', pct(p), border=False)
            with c2:
                st.badge(f'Risiko {band}', icon=icon, color=color)
                st.caption(band_text)
            st.progress(min(p / 0.5, 1.0), text=f'Lebih berisiko daripada {pctile:.0f}% pemohon di data uji')
            c1, c2, c3 = st.columns(3)
            c1.metric('Decile risiko', f'{decile} / 10', border=True)
            c2.metric('Bad rate aktual decile ini', pct(dec_bad), border=True)
            c3.metric('Rata-rata portofolio', pct(BASE_BAD), border=True)
            decision = 'Tolak' if p >= TH['profit'] else 'Setujui'
            st.caption(f'Keputusan pada threshold profit-optimal ({num(TH["profit"])}): **{decision}**. '
                       f'Break-even PD: {num(TH["breakeven_pd"])}. Atur threshold sendiri di halaman *Simulasi Threshold*.')

        with st.container(border=True):
            st.markdown('**Faktor yang paling memengaruhi prediksi**')
            ex = explain(X_row)
            ex['label'] = ex['fitur'].map(FEATURE_LABEL)
            ex['arah'] = np.where(ex['kontribusi'] > 0, 'Menaikkan risiko', 'Menurunkan risiko')
            top = ex.reindex(ex['kontribusi'].abs().sort_values(ascending=False).index).head(8)
            chart = alt.Chart(top).mark_bar(cornerRadius=3, height=16).encode(
                x=alt.X('kontribusi:Q', title='Kontribusi terhadap log-odds gagal bayar'),
                y=alt.Y('label:N', sort=alt.EncodingSortField('kontribusi', op='sum', order='descending'), title=None,
                        axis=alt.Axis(labelLimit=240, labelOverlap=False)),
                color=alt.Color('arah:N', scale=alt.Scale(domain=['Menaikkan risiko', 'Menurunkan risiko'],
                                                          range=[ORANGE, BLUE]),
                                legend=alt.Legend(orient='bottom', title=None)),
                tooltip=[alt.Tooltip('label:N', title='Fitur'), alt.Tooltip('kontribusi:Q', title='Kontribusi', format='+.3f'),
                         alt.Tooltip('arah:N', title='Arah')],
            ).properties(height=300)
            st.altair_chart(chart, width='stretch')
            st.caption('Dihitung dengan SHAP (TreeSHAP bawaan XGBoost), relatif terhadap rata-rata peminjam di data latih.')

    st.caption('Model dilatih dengan data Lending Club 2007–2012 untuk keperluan portofolio dan edukasi, bukan untuk keputusan kredit sungguhan.')


# ================================================================ halaman 2: simulasi
@st.cache_data
def policy_curve(scores: pd.DataFrame):
    grid = np.round(np.arange(0.05, 0.605, 0.01), 2)
    base_cash = scores['cash'].sum()
    rows = []
    for t in grid:
        a = scores['pd'] < t
        rows.append({'threshold': t, 'approval': a.mean(), 'bad_rate': scores.loc[a, 'target'].mean(),
                     'recall': 1 - scores.loc[a, 'target'].sum() / scores['target'].sum(),
                     'cash_delta': scores.loc[a, 'cash'].sum() / base_cash - 1})
    return pd.DataFrame(rows)


PRESET_T = {'Profit-optimal': 'profit', 'Break-even PD': 'breakeven_pd', 'Cutoff 10%': 'reject_10pct'}


def on_tpreset():
    k = st.session_state.get('tpreset')
    if k:
        st.session_state['threshold'] = round(TH[PRESET_T[k]], 2)


def page_simulation():
    st.title('Simulasi Threshold')
    st.caption('Model menghasilkan PD. Keputusan setuju/tolak bergantung pada **threshold**: pengajuan dengan PD ≥ threshold ditolak. '
               f'Simulasi memakai {len(scores):,} pinjaman data uji (dicairkan Jul–Des 2012) yang hasil akhirnya sudah diketahui.'.replace(',', '.'))
    if 'threshold' not in st.session_state:
        st.session_state['threshold'] = round(TH['profit'], 2)

    c1, c2 = st.columns([2, 1.3], vertical_alignment='bottom')
    c1.slider('Threshold PD', 0.05, 0.60, step=0.01, key='threshold', format='%.2f')
    c2.pills('Titik referensi', list(PRESET_T), key='tpreset', on_change=on_tpreset)
    t = st.session_state['threshold']

    a = scores['pd'] < t
    approval = a.mean()
    bad = scores.loc[a, 'target'].mean() if a.any() else 0.0
    recall = 1 - scores.loc[a, 'target'].sum() / scores['target'].sum()
    cash = scores.loc[a, 'cash'].sum()
    base_cash = scores['cash'].sum()

    m = st.columns(4)
    m[0].metric('Approval rate', pct(approval), f'{pct(approval - 1)} vs setujui semua', delta_color='off', border=True)
    m[1].metric('Bad rate pinjaman disetujui', pct(bad), f'{(bad - BASE_BAD) * 100:+.1f} poin'.replace('.', ','),
                delta_color='inverse', border=True)
    m[2].metric('Gagal bayar yang tertolak', pct(recall), help='Recall kelas berisiko: porsi pinjaman macet yang berhasil ditolak.',
                border=True)
    m[3].metric('Arus kas bersih', f'USD {num(cash / 1e6, 1)} jt', f'{(cash / base_cash - 1) * 100:+.1f}% vs setujui semua'.replace('.', ','),
                border=True)

    curve = policy_curve(scores)
    ref = pd.DataFrame([{'threshold': TH[v], 'nama': k} for k, v in PRESET_T.items()])
    cur = curve.iloc[[int(np.argmin(np.abs(curve['threshold'] - t)))]]

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.markdown('**Arus kas bersih vs threshold**')
        line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X('threshold:Q', title='Threshold PD'),
            y=alt.Y('cash_delta:Q', title='Selisih vs setujui semua', axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('threshold:Q', format='.2f', title='Threshold'),
                     alt.Tooltip('cash_delta:Q', format='+.1%', title='Arus kas vs setujui semua'),
                     alt.Tooltip('approval:Q', format='.1%', title='Approval rate')])
        rules = alt.Chart(ref).mark_rule(color=GREY, strokeDash=[4, 3]).encode(
            x='threshold:Q', tooltip=[alt.Tooltip('nama:N', title='Referensi'), alt.Tooltip('threshold:Q', format='.2f')])
        point = alt.Chart(cur).mark_point(color=ORANGE, filled=True, size=110, stroke='white', strokeWidth=2).encode(
            x='threshold:Q', y='cash_delta:Q')
        st.altair_chart((rules + line + point).properties(height=300), width='stretch')
        st.caption('Garis putus-putus: break-even PD, cutoff 10%, dan threshold profit-optimal.')
    with c2:
        st.markdown('**Trade-off approval rate vs bad rate**')
        base = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X('approval:Q', title='Approval rate', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
            y=alt.Y('bad_rate:Q', title='Bad rate pinjaman disetujui', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
            order='threshold:Q',
            tooltip=[alt.Tooltip('threshold:Q', format='.2f', title='Threshold'),
                     alt.Tooltip('approval:Q', format='.1%', title='Approval rate'),
                     alt.Tooltip('bad_rate:Q', format='.2%', title='Bad rate')])
        point = alt.Chart(cur).mark_point(color=ORANGE, filled=True, size=110, stroke='white', strokeWidth=2).encode(
            x='approval:Q', y='bad_rate:Q')
        st.altair_chart((base + point).properties(height=300), width='stretch')
        st.caption('Menolak lebih banyak pengajuan menurunkan bad rate, tetapi mengurangi volume dan pendapatan bunga.')

    c1, c2 = st.columns([1, 1.6], gap='large')
    with c1:
        st.markdown('**Confusion matrix pada threshold ini**')
        y, rej = scores['target'], ~a
        cm = pd.DataFrame({'Disetujui': [int(((y == 0) & ~rej).sum()), int(((y == 1) & ~rej).sum())],
                           'Ditolak': [int(((y == 0) & rej).sum()), int(((y == 1) & rej).sum())]},
                          index=['Lancar (aktual)', 'Gagal bayar (aktual)'])
        st.dataframe(cm, width='stretch')
    with c2:
        st.markdown('**Cara membaca**')
        st.markdown(
            f'- Rata-rata pinjaman lunas memberi untung, pinjaman macet memberi rugi lebih besar. '
            f'Titik impasnya ada di **PD {num(TH["breakeven_pd"])}**: di atas itu, pinjaman rata-rata merugi.\n'
            f'- Bunga pinjaman di data ini sudah disesuaikan dengan risiko, jadi hanya sedikit pengajuan yang PD-nya di atas titik impas. '
            f'Akibatnya threshold profit-optimal ({num(TH["profit"])}) hampir sama dengan menyetujui semua.\n'
            f'- Threshold yang lebih ketat cocok jika perusahaan ingin menekan bad rate (risk appetite), dengan konsekuensi arus kas turun.')


# ================================================================ halaman 3: ringkasan
def page_summary():
    M = meta['metrics']
    test = {r['Model']: r for r in M['test_out_of_time']}
    final = test['XGBoost (final)']
    st.title('Ringkasan Model')
    st.caption('Model XGBoost untuk memprediksi gagal bayar pinjaman 36 bulan Lending Club, dievaluasi pada data out-of-time. '
               f'[Kode & notebook di GitHub]({REPO_URL})')

    m = st.columns(4)
    m[0].metric('ROC-AUC (data uji)', num(final['ROC-AUC'], 3), border=True)
    m[1].metric('Gini', num(final['Gini'], 3), border=True)
    m[2].metric('KS', num(final['KS'], 3), border=True)
    d = meta['deciles']
    m[3].metric('Bad rate decile 10 vs 1', f'{pct(d[-1]["bad_rate"])} vs {pct(d[0]["bad_rate"])}', border=True)

    st.subheader('Kenapa bukan 99%?')
    c1, c2 = st.columns([1.2, 1], gap='large')
    ab = pd.DataFrame(M['ablation'])
    ab = pd.concat([ab[['Skenario', 'ROC-AUC']],
                    pd.DataFrame([{'Skenario': 'D. Model final v2 (out-of-time)', 'ROC-AUC': final['ROC-AUC']}])])
    ab['jenis'] = np.where(ab['Skenario'].str[0].isin(['A', 'B']), 'Masih bocor', 'Tanpa kebocoran')
    with c1:
        chart = alt.Chart(ab).mark_bar(cornerRadius=3, height=22).encode(
            x=alt.X('ROC-AUC:Q', scale=alt.Scale(domain=[0.5, 1.0], clamp=True), title='ROC-AUC (0,5 = tebakan acak)'),
            y=alt.Y('Skenario:N', sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
            color=alt.Color('jenis:N', scale=alt.Scale(domain=['Masih bocor', 'Tanpa kebocoran'], range=[GREY, BLUE]),
                            legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Skenario:N', alt.Tooltip('ROC-AUC:Q', format='.3f')])
        text = chart.mark_text(align='left', dx=4, color='currentColor').encode(text=alt.Text('ROC-AUC:Q', format='.3f'),
                                                                                color=alt.value('#52514e'))
        st.altair_chart((chart + text).properties(height=210), width='stretch')
    with c2:
        st.markdown(
            '- **A.** Versi 1 apa adanya: SMOTE sebelum split, data uji berisi sampel sintetis.\n'
            '- **B.** SMOTE dipindah ke setelah split: AUC hampir tidak berubah.\n'
            '- **C.** Fitur pasca-pencairan (`recoveries`, `total_rec_prncp`, dll.) dibuang: AUC turun ke 0,69. '
            'Inilah sumber kebocoran terbesar.\n'
            '- **D.** Model final: target hanya pinjaman yang sudah jatuh tempo, diuji pada periode berikutnya.')

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Bad rate per decile skor')
        dd = pd.DataFrame(d)
        bars = alt.Chart(dd).mark_bar(color=ORANGE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, width=22).encode(
            x=alt.X('decile:O', title='Decile risiko (1 = terendah)', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('bad_rate:Q', title='Bad rate aktual', axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('decile:O', title='Decile'), alt.Tooltip('n:Q', title='Jumlah', format=','),
                     alt.Tooltip('avg_pd:Q', title='Rata-rata PD', format='.1%'),
                     alt.Tooltip('bad_rate:Q', title='Bad rate aktual', format='.1%')])
        avg = alt.Chart(pd.DataFrame({'y': [BASE_BAD]})).mark_rule(color=GREY, strokeDash=[4, 3]).encode(y='y:Q')
        st.altair_chart((bars + avg).properties(height=280), width='stretch')
        st.caption(f'Garis putus-putus: rata-rata bad rate data uji ({pct(BASE_BAD)}).')
    with c2:
        st.subheader('Fitur terpenting (SHAP)')
        sh = pd.Series(meta['shap_global']).head(10).rename_axis('fitur').reset_index(name='nilai')
        sh['label'] = sh['fitur'].map(FEATURE_LABEL)
        chart = alt.Chart(sh).mark_bar(color=BLUE, cornerRadius=3, height=16).encode(
            x=alt.X('nilai:Q', title='Rata-rata |SHAP|'),
            y=alt.Y('label:N', sort='-x', title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
            tooltip=[alt.Tooltip('label:N', title='Fitur'), alt.Tooltip('nilai:Q', format='.3f', title='Rata-rata |SHAP|')])
        st.altair_chart(chart.properties(height=280), width='stretch')

    st.subheader('Perbandingan model (cross-validation data latih)')
    cv = pd.DataFrame(M['cv_train']).set_index('Model')
    st.dataframe(cv.style.format('{:.3f}'), width='stretch')
    st.caption('SMOTE dan class weight tidak menaikkan ROC-AUC dan membuat probabilitas tidak terkalibrasi (Brier score memburuk), '
               'sehingga model final memakai XGBoost tanpa balancing.')

    with st.expander('Metodologi & keterbatasan'):
        st.markdown(
            '**Metodologi**\n'
            '- Target: pinjaman 36 bulan yang dicairkan Jun-2007 s/d Des-2012 dan sudah jatuh tempo per Jan-2016. '
            'Gagal bayar = Charged Off / Default / Late 31–120 hari; lancar = Fully Paid.\n'
            '- Fitur: hanya yang tersedia saat pengajuan (lihat halaman *Kamus Fitur*).\n'
            '- Split out-of-time: latih sebelum Jul-2012 (44.063 pinjaman), uji Jul–Des 2012 (28.458 pinjaman).\n'
            '- Preprocessing di dalam pipeline scikit-learn; pemilihan model dengan 5-fold cross-validation pada data latih.\n\n'
            '**Keterbatasan**\n'
            '- Data hanya berisi pinjaman yang disetujui (bias seleksi); perlu *reject inference* untuk seluruh pemohon.\n'
            '- Populasi bergeser di periode uji (bad rate 12,0% → 13,6%), sehingga model perlu dimonitor.\n'
            '- Perhitungan arus kas disederhanakan: tanpa diskonto, biaya dana, dan biaya operasional.')


# ================================================================ halaman 4: kamus fitur
def page_dictionary():
    st.title('Kamus Fitur')
    st.caption('Deskripsi fitur yang dipakai model, berdasarkan data dictionary Lending Club, beserta cara pengolahannya.')

    with st.container(border=True):
        st.markdown('**Variabel target**')
        st.markdown('`target` = **1 (gagal bayar)** jika `loan_status` Charged Off, Default, atau Late (31–120 hari); '
                    '**0 (lancar)** jika Fully Paid. Hanya pinjaman 36 bulan yang sudah jatuh tempo yang dipakai, '
                    'sehingga hasil akhir setiap pinjaman sudah diketahui.')

    st.subheader(f'Fitur yang dipakai model ({len(FEATURE_DOCS)})')
    q = st.text_input('Cari fitur', placeholder='misalnya: kredit, pendapatan, tunggakan', label_visibility='collapsed')
    docs = pd.DataFrame(FEATURE_DOCS, columns=['Fitur model', 'Kolom asal', 'Nama', 'Deskripsi', 'Pengolahan'])
    imp = pd.Series(meta['shap_global'])
    docs['Peringkat pengaruh'] = docs['Fitur model'].map(imp.rank(ascending=False)).astype(int)
    docs = docs.sort_values('Peringkat pengaruh')
    if q:
        mask = docs.apply(lambda r: q.lower() in ' '.join(r.astype(str)).lower(), axis=1)
        docs = docs[mask]
    st.caption('Diurutkan dari fitur yang paling berpengaruh (rata-rata |SHAP| di data uji).')
    st.table(docs.rename(columns={'Peringkat pengaruh': '#'}).set_index('#')[
        ['Nama', 'Fitur model', 'Kolom asal', 'Deskripsi', 'Pengolahan']])

    st.subheader('Kolom yang tidak dipakai')
    st.table(pd.DataFrame(EXCLUDED, columns=['Kelompok', 'Kolom', 'Alasan']).set_index('Kelompok'))


pg = st.navigation([
    st.Page(page_calculator, title='Kalkulator Risiko', icon=':material/calculate:', default=True),
    st.Page(page_simulation, title='Simulasi Threshold', icon=':material/tune:', url_path='simulasi'),
    st.Page(page_summary, title='Ringkasan Model', icon=':material/insights:', url_path='ringkasan'),
    st.Page(page_dictionary, title='Kamus Fitur', icon=':material/menu_book:', url_path='kamus-fitur'),
], position='top')
pg.run()
