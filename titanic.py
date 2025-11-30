import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
import joblib
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Titan Restoration CRM", layout="wide")

# ---------- GLOBAL FONT & STYLE ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Comfortaa', sans-serif;
    background: white;
}

.metric-card {
    background:#000;
    color:white;
    padding:16px;
    border-radius:12px;
    margin-bottom:12px;
}

.metric-title {
    font-size:15px;
    font-weight:bold;
    color:white;
}

.metric-value {
    font-size:24px;
    font-weight:bold;
    margin-top:6px;
}

.progress-bar {
  height:7px;
  border-radius:4px;
  margin-top:10px;
}

.priority-money {
  font-size:17px;
  font-weight:bold;
  color:green;
}

.priority-time {
  font-size:13px;
  font-weight:bold;
  color:red;
}

.alert-panel {
  background:#000;
  padding:12px 16px;
  border-radius:8px;
  color:white;
  font-weight:bold;
  display:flex;
  justify-content:space-between;
  align-items:center;
}

.submit-btn > button {
    width:100%;
    background:red;
    border:none;
    padding:14px;
    border-radius:8px;
    font-size:16px;
    font-weight:600;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------- CLOUD SAFE DB DIRECTORY ----------
CLOUD_DB_PATH = "/home/adminuser/data"
os.makedirs(CLOUD_DB_PATH, exist_ok=True)

# ---------- DATABASE ENGINE ----------
DATABASE_URL = f"sqlite:///{CLOUD_DB_PATH}/restoration.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------- INITIAL TABLES ----------
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        full_name TEXT,
        role TEXT,
        alerts_enabled INTEGER,
        created_at TEXT
    );
    """))

    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        email TEXT,
        address TEXT,
        source TEXT,
        cost_to_acquire REAL,
        status TEXT,
        owner TEXT,
        estimate_value REAL,
        inspection_date TEXT,
        created_at TEXT,
        converted INTEGER,
        score INTEGER,
        time_left INTEGER
    );
    """))

    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS lead_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER,
        updated_by TEXT,
        old_status TEXT,
        new_status TEXT,
        note TEXT,
        updated_at TEXT
    );
    """))

# ---------- LOGIN ----------
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.page = "Pipeline"

if not st.session_state.user:
    u = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        with engine.connect() as conn:
            res = conn.execute(text("SELECT * FROM users WHERE username = :u"), {"u":u}).fetchone()
            if not res:
                conn.execute(text("""
                INSERT INTO users (username, full_name, role, alerts_enabled, created_at)
                VALUES (:u, :u, 'Viewer', 1, :d)
                """), {"u":u, "d":datetime.utcnow().isoformat()})

        st.session_state.user = u
        with engine.connect() as conn:
            role = conn.execute(text("SELECT role FROM users WHERE username=:u"), {"u":u}).scalar()
            st.session_state.role = role
        st.rerun()

    else:
        st.info("Please login to load your workspace.")
        st.stop()

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ---------- PAGE TITLES ----------
page = st.session_state.get("page", "Pipeline")

st.markdown(f"<h1 style='color:#000'>{page.upper()}</h1>", unsafe_allow_html=True)

# ---------- DATE FILTER ON PAGE ----------
st.markdown("### 📅 Filter Leads By Date")
col1, col2 = st.columns([1,1])
start_date = col1.date_input("Start Date", date.today())
end_date = col2.date_input("End Date", date.today())

# ---------- FUNCTIONS ----------
@st.cache_data(ttl=5)
def load_leads(start_date, end_date):
    with engine.connect() as conn:
        leads = conn.execute(text("""
        SELECT * FROM leads 
        WHERE created_at BETWEEN :start AND :end
        """), {"start":start_date.isoformat(), "end":end_date.isoformat()}).fetchall()
    return leads

def add_lead(name, phone, email, address, source, cost):
    with engine.connect() as conn:
        conn.execute(text("""
        INSERT INTO leads (name, phone, email, address, source, cost_to_acquire, status, owner, created_at, converted, score, time_left)
        VALUES (:name, :phone, :email, :address, :source, :cost, 'CAPTURED', 'UNASSIGNED', :d, 0, :score, :tl)
        """), {"name":name, "phone":phone, "email":email, "address":address, "source":source, "cost":cost or 0, "d":datetime.utcnow().isoformat(), "score":random.randint(40,100), "tl":48})

def update_lead(id, status, owner, spend, value, tl):
    with engine.connect() as conn:
        old = conn.execute(text("SELECT status FROM leads WHERE id=:id"),{"id":id}).scalar()

        conn.execute(text("""
        UPDATE leads 
        SET status=:status, owner=:owner, cost_to_acquire=:spend, estimate_value=:value, converted=:conv, time_left=:tl
        WHERE id=:id
        """), {"id":id, "status":status, "owner":owner, "spend":spend or 0, "value":value or 0, "conv":1 if status=="AWARDED" else 0, "tl":tl or 48})

        conn.execute(text("""
        INSERT INTO lead_history (lead_id, updated_by, old_status, new_status, updated_at)
        VALUES (:id, :user, :old, :new, :d)
        """), {"id":id, "user":st.session_state.user, "old":old, "new":status, "d":datetime.utcnow().isoformat()})

def get_stage_counts():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT status, COUNT(*) AS count FROM leads GROUP BY status", conn)
    return df

# ---------- LEAD CAPTURE UI ----------
if page == "Lead Capture":
    st.text_input("Lead Name", "", key="lname")
    st.text_input("Phone", "", key="lphone")
    st.text_input("Email", "", key="lemail")
    st.text_area("Address", "", key="laddr")
    st.selectbox("Lead Source", ["Website","Referral","Facebook","Instagram","Hotline","Walk-In","TikTok","Google Ads"], key="lsrc")
    st.number_input("Cost to Acquire ($0 default)", 0.0, key="lcost")

    if st.button("Submit Lead", key="submit_lead"):
        add_lead(st.session_state.lname, st.session_state.lphone, st.session_state.lemail, st.session_state.laddr, st.session_state.lsrc, st.session_state.lcost)
        st.success("✅ Lead submitted and saved!")

# ---------- PIPELINE DASHBOARD ----------
elif page == "Pipeline":
    df_stage = get_stage_counts()
    if not df_stage.empty:
        fig = px.pie(df_stage, names="status", values="count", hole=0.5)
        st.plotly_chart(fig, use_container_width=False)

    leads = load_leads(start_date, end_date)

    df_top = pd.DataFrame(leads, columns=["id","name","estimate_value","time_left"]).sort_values("estimate_value", ascending=False).head(5)

    st.markdown("### ⭐ TOP 5 PRIORITY LEADS")
    pcols = st.columns(5)
    for col,(_,row) in zip(pcols, df_top.iterrows()):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title' style='color:{random.choice(["red","blue","orange","green","purple"])}'>{row['name']}</div>
            <div class='priority-money'>${row['estimate_value']:,.2f}</div>
            <div class='priority-time'>{row['time_left']} hrs left</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 ALL LEADS")
    for l in leads:
        with st.expander(f"Lead #{l[0]} — {l[1]}"):
            st.selectbox("Assign Owner", ["Admin","Estimator","Adjuster","Tech","Viewer","UNASSIGNED"], key=f"own_{l[0]}")
            st.selectbox("Change Stage", ["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"], key=f"stg_{l[0]}")
            st.number_input("Spend", l[6], key=f"sp_{l[0]}")
            st.number_input("Estimate Value", l[8], key=f"val_{l[0]}")
            st.slider("Time left", 1, 96, l[10], key=f"tl_{l[0]}")

            if st.button("Save Lead", key=f"save_{l[0]}"):
                update_lead(l[0], st.session_state[f"stg_{l[0]}"], st.session_state[f"own_{l[0]}"], st.session_state[f"sp_{l[0]}"], st.session_state[f"val_{l[0]}"], st.session_state[f"tl_{l[0]}"])

                st.success("✅ Lead updated & audit saved!")

# ---------- ANALYTICS DASHBOARD ----------
elif page == "Analytics":
    df_stage = get_stage_counts()
    if not df_stage.empty:
        line = px.line(df_stage, x="status", y="count")
        st.plotly_chart(line, use_container_width=True)

# ---------- CPA/ROI ----------
elif page == "CPA/ROI":
    leads = load_leads(date.today(),date.today())
    spend = 0
    won = 0
    revenue = 0
    for l in leads:
        spend += l[6] or 0
        revenue += l[8] if l[7]=="AWARDED" else 0
        won += 1 if l[9]==1 else 0

    cpa = spend/won if won else 0
    roi = revenue-spend
    roi_pct = (roi/spend*100) if spend else 0

    st.markdown("### KPIs")
    c1,c2,c3 = st.columns(3)
    c1.markdown("<div class='metric-card'><div style='color:red'>Total Spend</div>${:,.2f}</div>".format(spend),unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><div style='color:blue'>Conversions (Won)</div>{}</div>".format(won),unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><div style='color:green'>ROI</div>${:,.2f} ({:,.1f}%)</div>".format(roi,roi_pct),unsafe_allow_html=True)

    if not df.empty:
        fig = plt.figure(figsize=(4,2))
        plt.plot(["Spend","Won"], [spend,won])
        st.pyplot(fig, use_container_width=False)
