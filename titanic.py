# titan_full.py — TITAN merged full app
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Connection
from datetime import datetime, timedelta, date
import io
import base64
import os
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Configuration
# ----------------------------
DB_FILE = 'titan_leads.db'
MODEL_FILE = 'titan_lead_scoring.joblib'
PIPELINE_STAGES = ['New', 'Contacted', 'Qualified', 'Estimate Sent', 'Won', 'Lost']
SLA_HOURS = 72  # SLA threshold for "time left" calculation (72 hours default)

st.set_page_config(page_title="TITAN - Lead Pipeline", layout='wide')

APP_CSS = """
<style>
body {background-color: #071129; color: #e6eef8; font-family: 'Comfortaa', sans-serif;}
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
.header {display:flex; align-items:center; gap:12px; margin-bottom:6px;}
.metric-card {background: linear-gradient(135deg,#0ea5a0,#06b6d4); padding:14px; border-radius:10px; color:white}
.small {font-size:12px; opacity:0.95}
.kpi {font-size:26px; font-weight:700}
.card-row {display:flex; gap:12px; flex-wrap:wrap}
.table-container {background:#0b1220; padding:12px; border-radius:8px}
.lead-card {background:#071a2a; border-radius:8px; padding:10px; margin-bottom:8px}
.alert-badge {background:#ef4444;color:#fff;padding:8px 10px;border-radius:8px;cursor:pointer}
.kpi-black {background:#000;padding:14px;border-radius:10px;color:#fff}
.small-muted {color:#a7b2c7;font-size:13px}
.topbar {display:flex; gap:12px; align-items:center; justify-content:space-between}
.badge-small {background:#111;color:#fff;padding:6px 8px;border-radius:8px}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------------
# Database helpers
# ----------------------------

def get_conn() -> Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # leads table
    c.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT UNIQUE,
            created_at TEXT,
            source TEXT,
            stage TEXT,
            estimated_value REAL,
            ad_cost REAL,
            converted INTEGER,
            notes TEXT,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ensure score column exists (backward-compat safety)
def ensure_score_column():
    conn = get_conn()
    c = conn.cursor()
    c.execute("PRAGMA table_info(leads)")
    cols = [r['name'] for r in c.fetchall()]
    if 'score' not in cols:
        c.execute("ALTER TABLE leads ADD COLUMN score REAL DEFAULT NULL")
    conn.commit()
    conn.close()

ensure_score_column()

# ----------------------------
# CRUD helpers
# ----------------------------

def insert_leads(df: pd.DataFrame):
    conn = get_conn()
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at']).astype(str)
    rows = df[['lead_id','created_at','source','stage','estimated_value','ad_cost','converted','notes']].fillna('').values.tolist()
    c = conn.cursor()
    for r in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO leads (lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes)
                         VALUES (?,?,?,?,?,?,?,?)''', r)
        except Exception as e:
            print('Insert error', e)
    conn.commit()
    conn.close()

def upsert_lead(row: dict):
    conn = get_conn()
    c = conn.cursor()
    # use INSERT OR REPLACE to simplify (preserves unique lead_id)
    # But prefer UPSERT syntax
    c.execute('''
        INSERT INTO leads (lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes,score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lead_id) DO UPDATE SET
            created_at=excluded.created_at,
            source=excluded.source,
            stage=excluded.stage,
            estimated_value=excluded.estimated_value,
            ad_cost=excluded.ad_cost,
            converted=excluded.converted,
            notes=excluded.notes,
            score=excluded.score
    ''', (
        row.get('lead_id'),
        row.get('created_at'),
        row.get('source'),
        row.get('stage'),
        float(row.get('estimated_value') or 0.0),
        float(row.get('ad_cost') or 0.0),
        int(row.get('converted') or 0),
        row.get('notes') or '',
        float(row.get('score')) if row.get('score') is not None else None
    ))
    conn.commit()
    conn.close()

def delete_lead(lead_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM leads WHERE lead_id=?', (lead_id,))
    conn.commit()
    conn.close()

def load_all_leads_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY datetime(created_at) DESC", conn)
    conn.close()
    # parse dates
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df

# ----------------------------
# Mock data seeding
# ----------------------------

def generate_mock_leads_df(n=300):
    rng = np.random.default_rng(42)
    created = [ (datetime.now() - timedelta(days=int(x))).isoformat() for x in rng.integers(0,120,size=n) ]
    sources = rng.choice(['Google Ads','Organic','Referral','Facebook Ads','Direct','Partner'], size=n)
    stages = rng.choice(PIPELINE_STAGES, size=n, p=[0.18,0.25,0.2,0.15,0.12,0.1])
    est_value = np.round(rng.normal(2500,1800,size=n).clip(200,25000),2)
    cost = np.round(rng.normal(55,35,size=n).clip(0,800),2)
    lead_id = [f"L{200000+i}" for i in range(n)]
    converted = np.where(stages=='Won',1,0)
    df = pd.DataFrame({
        'lead_id':lead_id,
        'created_at':created,
        'source':sources,
        'stage':stages,
        'estimated_value':est_value,
        'ad_cost':cost,
        'converted':converted,
        'notes':''
    })
    return df

# Seed DB with mock data if empty
try:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) AS cnt FROM leads')
    cnt = cur.fetchone()['cnt']
    conn.close()
except Exception:
    cnt = 0

if cnt == 0:
    seed = generate_mock_leads_df(300)
    insert_leads(seed)

# ----------------------------
# Utility helpers
# ----------------------------

def to_dataframe(sql: str, params=()):
    conn = get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    if not df.empty and 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df

def download_link(df: pd.DataFrame, filename: str):
    towrite = io.BytesIO()
    # write excel
    try:
        df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
        return f'<a href="{href}" download="{filename}">Download {filename}</a>'
    except Exception as e:
        # fallback to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'data:text/csv;base64,{b64}'
        return f'<a href="{href}" download="{filename.replace(".xlsx",".csv")}">Download {filename.replace(".xlsx",".csv")}</a>'

# ----------------------------
# Model helpers (train / load / score)
# ----------------------------

def train_model_from_db():
    df = load_all_leads_df()
    if df.empty:
        return None, "No data"
    # prepare features
    df2 = df.copy()
    df2['age_days'] = (datetime.now() - df2['created_at']).dt.days
    # One-hot for source and stage
    X = pd.get_dummies(df2[['source','stage']].astype(str), drop_first=False)
    X['ad_cost'] = df2['ad_cost']
    X['estimated_value'] = df2['estimated_value']
    X['age_days'] = df2['age_days']
    y = df2['converted']
    if len(y.unique()) == 1:
        return None, "Only one class present; cannot train"
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump({'model': model, 'columns': X.columns.tolist()}, MODEL_FILE)
    return model, acc

def load_model():
    if os.path.exists(MODEL_FILE):
        obj = joblib.load(MODEL_FILE)
        return obj.get('model'), obj.get('columns')
    return None, None

def score_leads_in_memory(df: pd.DataFrame, model, model_cols):
    if model is None or df.empty:
        df['score'] = np.nan
        return df
    df2 = df.copy()
    df2['age_days'] = (datetime.now() - df2['created_at']).dt.days
    X = pd.get_dummies(df2[['source','stage']].astype(str), drop_first=False)
    X['ad_cost'] = df2['ad_cost']
    X['estimated_value'] = df2['estimated_value']
    X['age_days'] = df2['age_days']
    # align columns
    for c in model_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[model_cols].fillna(0)
    # predict_proba might exist
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = model.predict(X)
    df['score'] = proba
    return df

# ----------------------------
# SLA / Priority helpers
# ----------------------------
def compute_time_left_hours(created_at: pd.Timestamp):
    if pd.isna(created_at):
        return SLA_HOURS
    delta = datetime.utcnow() - pd.to_datetime(created_at)
    hours_passed = delta.total_seconds() / 3600.0
    time_left = max(0.0, SLA_HOURS - hours_passed)
    return time_left

def top_priority(df: pd.DataFrame, top_n=5):
    if df.empty:
        return df
    # priority heuristic: weighted score if present else estimated_value and time left
    if 'score' not in df.columns:
        df['score'] = 0.0
    df['time_left_h'] = df['created_at'].apply(compute_time_left_hours)
    # priority = 0.6 * normalized score + 0.4 * normalized value + overdue boost
    vals = df['estimated_value'].fillna(0.0)
    if vals.max() > 0:
        v_norm = vals / vals.max()
    else:
        v_norm = 0
    s_norm = df['score'].fillna(0.0)
    priority = 0.6 * s_norm + 0.4 * v_norm
    # boost overdue items
    priority = priority + np.where(df['time_left_h'] <= 0, 0.5, 0.0)
    df['priority'] = priority
    return df.sort_values('priority', ascending=False).head(top_n)

# ----------------------------
# Layout & Pages
# ----------------------------

# top bar with small controls: date range + alert bell
st.markdown("<div class='topbar'><div><h2 style='margin:0'>📊 TOTAL LEAD PIPELINE KPI</h2><div class='small-muted'>Snapshot of leads and conversion across pipeline stages</div></div></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([7,3])
with col_right:
    date_quick = st.selectbox("Quick range", ["Today", "Last 7 days", "Last 30 days", "All", "Custom"], index=3)
    if date_quick == "Today":
        start_date = date.today(); end_date = date.today()
    elif date_quick == "Last 7 days":
        start_date = date.today() - timedelta(days=6); end_date = date.today()
    elif date_quick == "Last 30 days":
        start_date = date.today() - timedelta(days=29); end_date = date.today()
    elif date_quick == "All":
        start_date = None; end_date = None
    else:
        start_date, end_date = st.date_input("Start - End", [date.today() - timedelta(days=6), date.today()])

# Alert bell UI-only
# compute overdue count
df_all = load_all_leads_df()
overdue_count = 0
if not df_all.empty:
    df_all['time_left_h'] = df_all['created_at'].apply(compute_time_left_hours)
    overdue_count = int((df_all['time_left_h'] <= 0).sum())

if 'alerts_open' not in st.session_state:
    st.session_state.alerts_open = False

if overdue_count > 0:
    if st.button(f"🔔 {overdue_count} SLA alerts"):
        st.session_state.alerts_open = not st.session_state.alerts_open
else:
    st.markdown("<span class='badge-small'>No SLA alerts</span>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title('TITAN Control Panel')
page = st.sidebar.selectbox('Choose page', ['Dashboard','Leads','CPA','ML Lead Scoring','Imports/Exports','Settings','Reports'], index=0)

# ----------------------------
# Page: Dashboard
# ----------------------------
def page_dashboard(filtered_df):
    st.markdown("<div class='card-row'>", unsafe_allow_html=True)

    total_leads = len(filtered_df)
    new_leads = int((filtered_df['stage']=='New').sum()) if not filtered_df.empty else 0
    contacted = int((filtered_df['stage']=='Contacted').sum()) if not filtered_df.empty else 0
    conv_rate = filtered_df['converted'].mean() if total_leads>0 else 0.0

    # pipeline job value and estimate sent, inspection booked derived via stages
    estimate_sent = int((filtered_df['stage']=='Estimate Sent').sum()) if not filtered_df.empty else 0
    won = int((filtered_df['stage']=='Won').sum()) if not filtered_df.empty else 0
    lost = int((filtered_df['stage']=='Lost').sum()) if not filtered_df.empty else 0
    pipeline_value = float(filtered_df['estimated_value'].sum()) if not filtered_df.empty else 0.0

    # first row (4)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='kpi'>{total_leads}</div><div class='small'>Total leads</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='kpi'>{new_leads}</div><div class='small'>New leads</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='kpi'>{contacted}</div><div class='small'>Contacted</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='kpi'>{conv_rate*100:.1f}%</div><div class='small'>Conversion rate</div></div>", unsafe_allow_html=True)

    # second row (3)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"<div class='metric-card'><div class='kpi'>{estimate_sent}</div><div class='small'>Estimates sent</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='metric-card'><div class='kpi'>{won}</div><div class='small'>Won</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='metric-card'><div class='kpi'>${pipeline_value:,.0f}</div><div class='small'>Pipeline job value</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Pipeline stages bar (ordered)
    stage_counts = filtered_df['stage'].value_counts().reindex(PIPELINE_STAGES, fill_value=0).reset_index()
    stage_counts.columns = ['stage','count']
    st.subheader('Pipeline stages')
    st.plotly_chart(px.bar(stage_counts, x='stage', y='count', color='stage', color_discrete_sequence=px.colors.qualitative.Dark24), use_container_width=True)

    st.markdown("---")
    st.subheader('Top 5 Priority Leads')
    st.markdown('*Computed from internal ML score (if available), estimate value and SLA urgency.*')

    # load model and score
    model, model_cols = load_model()
    scored = score_leads_in_memory(filtered_df.copy(), model, model_cols) if not filtered_df.empty else filtered_df
    top5 = top_priority(scored, top_n=5)
    if top5.empty:
        st.info("No priority leads to display")
    else:
        # show cards horizontally
        cols = st.columns(5)
        for col, (_, r) in zip(cols, top5.iterrows()):
            time_left_h = r.get('time_left_h', compute_time_left_hours(r['created_at']))
            time_html = f"<div style='color:#ff6b6b;font-weight:700'>{int(time_left_h)}h left</div>" if time_left_h > 0 else "<div style='color:#ef4444;font-weight:900'>❗ OVERDUE</div>"
            money_html = f"<div style='color:#22c55e;font-weight:900'>${float(r.get('estimated_value') or 0):,.0f}</div>"
            col.markdown(f"<div class='lead-card'><b>#{r['lead_id']} — {r['source']}</b><br><div class='small-muted'>{r.get('notes') or ''}</div><br>{money_html}{time_html}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader('Recent Leads')
    st.dataframe(filtered_df.sort_values('created_at', ascending=False).head(50))

# ----------------------------
# Page: Leads (capture & manage)
# ----------------------------
def page_leads():
    st.header('Lead Capture & Management')
    df = load_all_leads_df()

    with st.form('lead_form'):
        lead_id = st.text_input('Lead ID (unique)', value=f"L{int(datetime.utcnow().timestamp())}" )
        source = st.selectbox('Source', ['Google Ads','Organic','Referral','Facebook Ads','Direct','Partner','Other'])
        stage = st.selectbox('Pipeline Stage', PIPELINE_STAGES)
        est_val = st.number_input('Estimated Value', value=0.0, min_value=0.0, step=10.0)
        ad_cost = st.number_input('Ad Cost', value=0.0, min_value=0.0, step=1.0)
        converted = st.checkbox('Converted', value=False)
        notes = st.text_area('Notes')
        submitted = st.form_submit_button('Save Lead')
        if submitted:
            if not lead_id:
                st.error('Lead ID required')
            else:
                row = {
                    'lead_id': lead_id.strip(),
                    'created_at': datetime.now().isoformat(),
                    'source': source,
                    'stage': stage,
                    'estimated_value': float(est_val),
                    'ad_cost': float(ad_cost),
                    'converted': int(bool(converted)),
                    'notes': notes or '',
                    'score': None
                }
                try:
                    upsert_lead(row)
                    st.success('Lead captured / updated')
                except Exception as e:
                    st.error(f"Failed to save lead: {e}")
    st.markdown('---')
    st.subheader('Filter & Edit Leads')
    df = load_all_leads_df()
    if df.empty:
        st.info("No leads in DB")
        return

    # filters
    src_filter = st.multiselect('Filter by Source', options=sorted(df['source'].dropna().unique()), default=sorted(df['source'].dropna().unique()))
    stg_filter = st.multiselect('Filter by Stage', options=PIPELINE_STAGES, default=PIPELINE_STAGES)
    start = st.date_input("Start date", value=(date.today() - timedelta(days=30)))
    end = st.date_input("End date", value=date.today())
    mask = (df['created_at'] >= pd.to_datetime(start)) & (df['created_at'] <= pd.to_datetime(end) + pd.Timedelta(days=1))
    filtered = df[mask & df['source'].isin(src_filter) & df['stage'].isin(stg_filter)]

    st.write(f"Showing {len(filtered)} leads")
    st.dataframe(filtered.sort_values('created_at', ascending=False).head(200))

    # quick edit: select a lead to edit
    sel = st.selectbox("Select lead to edit", options=[''] + filtered['lead_id'].tolist())
    if sel:
        rec = filtered[filtered['lead_id'] == sel].iloc[0].to_dict()
        st.markdown('### Edit lead')
        new_stage = st.selectbox("Stage", PIPELINE_STAGES, index=PIPELINE_STAGES.index(rec['stage']) if rec['stage'] in PIPELINE_STAGES else 0)
        new_est = st.number_input("Estimated value", value=float(rec['estimated_value'] or 0.0))
        new_cost = st.number_input("Ad cost", value=float(rec['ad_cost'] or 0.0))
        new_notes = st.text_area("Notes", value=rec.get('notes') or '')
        if st.button("Update lead"):
            try:
                upsert_lead({
                    'lead_id': rec['lead_id'],
                    'created_at': rec['created_at'].isoformat() if isinstance(rec['created_at'], pd.Timestamp) else rec['created_at'],
                    'source': rec['source'],
                    'stage': new_stage,
                    'estimated_value': new_est,
                    'ad_cost': new_cost,
                    'converted': 1 if new_stage == 'Won' else 0,
                    'notes': new_notes,
                    'score': rec.get('score')
                })
                st.success("Lead updated")
            except Exception as e:
                st.error("Failed to update: " + str(e))

    # delete lead
    st.markdown("---")
    del_id = st.text_input("Lead ID to delete (type exact lead_id)")
    if st.button("Delete lead"):
        if del_id:
            try:
                delete_lead(del_id.strip())
                st.success("Deleted (if existed)")
            except Exception as e:
                st.error("Failed to delete: " + str(e))

# ----------------------------
# Page: CPA & ROI
# ----------------------------
def page_cpa():
    st.header("💰 CPA & ROI")
    df = load_all_leads_df()
    if df.empty:
        st.info("No leads")
        return
    # apply date filter from top controls if set
    # compute totals
    total_spend = float(df['ad_cost'].sum())
    conversions = int(df[df['stage'] == 'Won'].shape[0])
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(df[df['stage']=='Won']['estimated_value'].sum())
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
    col2.metric("Conversions (Won)", f"{conversions}")
    col3.metric("CPA", f"${cpa:,.2f}")
    col4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")

    st.markdown("---")
    st.subheader("Marketing Spend vs Conversions (by source)")
    agg = df.groupby('source').agg(total_spend=('ad_cost','sum'), conversions=('lead_id', lambda s: df.loc[s.index,'stage'].eq('Won').sum())).reset_index()
    if not agg.empty:
        fig = px.bar(agg, x='source', y=['total_spend','conversions'], barmode='group', labels={'value':'Amount','source':'Source'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to plot")

# ----------------------------
# Page: ML Lead Scoring
# ----------------------------
def page_ml():
    st.header("🤖 ML Lead Scoring (internal only)")
    st.markdown("Train a RandomForest model to predict whether a lead converts. Model stored locally (`titan_lead_scoring.joblib`).")
    df = load_all_leads_df()
    st.write(f"Leads in DB: {len(df)}")
    if st.button("Train model from DB"):
        with st.spinner("Training model..."):
            m, acc = train_model_from_db()
            if m is None:
                st.error(f"Training failed or insufficient classes to train: {acc}")
            else:
                st.success(f"Model trained (approx accuracy {acc:.3f}). Saved to `{MODEL_FILE}`.")
    model, model_cols = load_model()
    if model is None:
        st.info("No pre-trained model found. Train using the button above.")
    else:
        st.success("Model available (internal).")
        # Score leads and optionally persist scores to DB
        if st.button("Score & persist lead scores"):
            df2 = load_all_leads_df()
            scored = score_leads_in_memory(df2.copy(), model, model_cols)
            # persist back to DB
            conn = get_conn()
            cur = conn.cursor()
            for _, r in scored.iterrows():
                cur.execute("UPDATE leads SET score=? WHERE lead_id=?", (float(r['score']) if not np.isnan(r['score']) else None, r['lead_id']))
            conn.commit()
            conn.close()
            st.success("Scores persisted to DB (score column).")

        st.subheader("Sample predictions (top probable leads)")
        df2 = load_all_leads_df()
        scored_preview = score_leads_in_memory(df2.copy(), model, model_cols).sort_values('score', ascending=False).head(20)
        st.dataframe(scored_preview[['lead_id','source','stage','estimated_value','ad_cost','score']].head(20))

# ----------------------------
# Page: Imports / Exports
# ----------------------------
def page_imports_exports():
    st.header("Imports & Exports")
    st.subheader("Import leads from Excel/CSV (columns: lead_id, created_at, source, stage, estimated_value, ad_cost, converted, notes)")
    uploaded = st.file_uploader("Upload file", type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
            # basic column checks
            required = {'lead_id','created_at','source','stage'}
            if not required.issubset(set(df_in.columns)):
                st.warning(f"File missing required columns: {required - set(df_in.columns)}")
            else:
                # coerce columns and insert
                df_in['created_at'] = pd.to_datetime(df_in['created_at']).astype(str)
                insert_leads(df_in)
                st.success("Imported (existing lead_id are ignored).")
        except Exception as e:
            st.error("Import failed: " + str(e))

    st.markdown("---")
    st.subheader("Export full leads or filtered selection")
    df = load_all_leads_df()
    if df.empty:
        st.info("No leads in DB")
    else:
        if st.button("Export all leads (Excel)"):
            html = download_link(df, 'all_leads.xlsx')
            st.markdown(html, unsafe_allow_html=True)
        st.markdown("Or export filtered leads from the Leads page (use filters).")

# ----------------------------
# Page: Settings
# ----------------------------
def page_settings():
    st.header("Settings")
    st.markdown("Configure app behavior and pipeline stages (static for now).")
    st.markdown("SLA hours (used for 'time left'):")
    global SLA_HOURS
    SLA_HOURS = st.number_input("SLA hours", value=SLA_HOURS, min_value=1, max_value=168, step=1)

# ----------------------------
# Page: Reports
# ----------------------------
def page_reports():
    st.header("Reports")
    df = load_all_leads_df()
    if df.empty:
        st.info("No leads")
        return
    st.subheader("SLA / Overdue Trend (last 30 days)")
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    counts = []
    for d in days:
        start = datetime.combine(d, datetime.min.time())
        end = datetime.combine(d, datetime.max.time())
        subset = df[(df['created_at'] >= start) & (df['created_at'] <= end)]
        overdue_count = int(((subset['created_at'].apply(lambda x: compute_time_left_hours(x)) <= 0)).sum())
        counts.append(overdue_count)
    trend = pd.DataFrame({'date': days, 'overdue': counts})
    st.line_chart(trend.set_index('date'))

    st.markdown('---')
    st.subheader('Stage counts (table)')
    st.table(df['stage'].value_counts().reindex(PIPELINE_STAGES, fill_value=0))

# ----------------------------
# Render pages
# ----------------------------
# apply date filters to data functions that accept date range
if start_date is None and end_date is None:
    df_filtered = load_all_leads_df()
else:
    df_tmp = load_all_leads_df()
    if df_tmp.empty:
        df_filtered = df_tmp
    else:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df_filtered = df_tmp[(df_tmp['created_at'] >= start_dt) & (df_tmp['created_at'] <= end_dt)]

# show alerts dropdown if toggled
if st.session_state.get('alerts_open', False):
    st.subheader(f"SLA Alerts ({overdue_count})")
    overdue_df = df_all[df_all['time_left_h'] <= 0] if not df_all.empty else pd.DataFrame()
    if overdue_df.empty:
        st.info("No overdue leads")
    else:
        for _, r in overdue_df.sort_values('created_at').iterrows():
            st.markdown(f"<div class='lead-card'><b>{r['lead_id']} • {r['source']}</b> — <span style='color:#22c55e'>${float(r['estimated_value'] or 0):,.0f}</span> <span style='float:right;color:#ef4444'>OVERDUE</span><br><div class='small-muted'>{r.get('notes') or ''}</div></div>", unsafe_allow_html=True)

if page == 'Dashboard':
    page_dashboard(df_filtered)
elif page == 'Leads':
    page_leads()
elif page == 'CPA':
    page_cpa()
elif page == 'ML Lead Scoring':
    page_ml()
elif page == 'Imports/Exports':
    page_imports_exports()
elif page == 'Settings':
    page_settings()
elif page == 'Reports':
    page_reports()
else:
    st.info("Page not found")

# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>TITAN — single-file Streamlit lead pipeline · SQLite persistence · ML scoring (internal)</div>", unsafe_allow_html=True)
