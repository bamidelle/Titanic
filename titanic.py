import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import io, base64, os, subprocess, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import plotly.express as px

# Ensure openpyxl is installed for exports
try:
    import openpyxl
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"], stdout=subprocess.DEVNULL)
    st.rerun()

# ================= CONFIG ===================
DB_FILE = "titan_leads.db"
MODEL_FILE = "titan_lead_model.joblib"
PIPELINE_STAGES = ["New","Contacted","Qualified","Estimate Sent","Won","Lost"]
SLA_LIMIT_HOURS = 24

# ================ CSS =======================
APP_CSS = """
<style>
body {background:#071129;color:#e6eef8}
.metric-card {
    background:linear-gradient(135deg,#0ea5a0,#06b6d4);
    padding:16px;border-radius:14px;color:#fff;
    min-width:180px;text-align:center;
    box-shadow:0 3px 8px rgba(0,0,0,0.2)
}
.metric-title{font-size:13px;opacity:.9;margin-top:6px}
.kpi{font-size:28px;font-weight:800}
.priority-card{
    background:#000;padding:14px;border-radius:12px;
    min-width:260px;color:#fff;box-shadow:0 3px 8px rgba(0,0,0,0.3)
}
.priority-title{font-size:17px;font-weight:700}
.row-spacing{margin-bottom:28px}
.alert-bell{font-size:26px;cursor:pointer}
</style>
"""
st.set_page_config(page_title="TITAN Backend", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

# ================ DB SETUP ==================
def get_conn():
    return sqlite3.connect(DB_FILE,check_same_thread=False)

def init_db():
    conn=get_conn();c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS leads(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT UNIQUE,
        created_at TEXT,
        source TEXT,
        stage TEXT,
        estimated_value REAL,
        ad_cost REAL,
        converted INTEGER,
        user TEXT,
        notes TEXT,
        sla_hours INTEGER,
        updated_at TEXT,
        urgency TEXT
    )
    """)
    conn.commit();conn.close()

init_db()

# ================ USER LOGIN (internal) =====
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = "Viewer"  # default until admin sets

def do_login():
    u = st.session_state.login_name.strip()
    if u:
        st.session_state.user = u
        st.rerun()

def do_logout():
    st.session_state.user = None
    st.rerun()

# ================ DATA HELPERS ==============
@st.cache_data(ttl=10)
def load_leads_df():
    df=st.dataframe(pd.DataFrame())
    try:
        df=pd.read_sql("SELECT * FROM leads",get_conn())
    except:
        df=pd.DataFrame()
    if df.empty:
        df=pd.DataFrame(columns=["lead_id","created_at","source","stage","estimated_value",
                                 "ad_cost","converted","user","notes","sla_hours","urgency","urgency"])
    return df

def save_lead(row):
    conn=get_conn();c=conn.cursor()
    row=row.copy()
    row["created_at"]=row.get("created_at",datetime.now().isoformat())
    row["updated_at"]=datetime.now().isoformat()
    row["updated_at"]=datetime.now().isoformat()
    row["updated_at"]=datetime.now().isoformat()
    row["user"]=row.get("user",st.session_state.user)
    row["updated_at"]=datetime.now().isoformat()
    row["updated_at"]=datetime.now().isoformat()

    # Determine urgency color/level for priority cards
    hours_left = row["sla_hours"]
    row["urgency"] = "🔴 High Urgency" if hours_left < 6 else "🟡 Medium Urgency" if hours_left < 12 else "🟢 Normal"

    c.execute("""
    INSERT INTO leads(lead_id,created_at,source,stage,estimated_value,ad_cost,converted,user,notes,sla_hours,updated_at,urgency)
    VALUES(:lead_id,:created_at,:source,:stage,:estimated_value,:ad_cost,:converted,:user,:notes,:sla_hours,:updated_at,:urgency)
    ON CONFLICT(lead_id) DO UPDATE SET
    source=excluded.source,stage=excluded.stage,estimated_value=excluded.estimated_value,
    ad_cost=excluded.ad_cost,converted=excluded.converted,user=excluded.user,
    notes=excluded.notes,sla_hours=excluded.sla_hours,updated_at=excluded.updated_at,urgency=excluded.urgency
    """,row)
    conn.commit();conn.close()

def delete_lead(lead_id):
    conn=get_conn();c=conn.cursor();c.execute("DELETE FROM leads WHERE lead_id=?",(lead_id,))
    conn.commit();conn.close()

# ================ ML LEAD SCORING ===========
def train_lead_model(df):
    if df.empty or (df["converted"].nunique() < 2):
        return None, "Not enough conversion data for ML training"

    df2=df.copy()
    df2["created_at"]=pd.to_datetime(df2["created_at"])
    df2["age_days"]=(datetime.now()-df2["created_at"]).dt.days

    X=pd.get_dummies(df2[["source","stage"]].astype(str))
    X["estimated_value"]=df2["estimated_value"].fillna(0)
    X["ad_cost"]=df2["ad_cost"].fillna(0)
    X["age_days"]=df2["age_days"].fillna(0)
    y=df2["converted"]

    Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestClassifier(n_estimators=120,max_depth=8,random_state=42)
    m.fit(Xtr,ytr)
    joblib.dump({"model":m,"cols":X.columns.tolist()},MODEL_FILE)

    return round((yts == m.predict(Xts)).float().mean().item(),4), "Trained & saved"

def load_lead_model():
    if os.path.exists(MODEL_FILE):
        obj=joblib.load(MODEL_FILE)
        return obj["model"],obj["cols"]
    return None,None

# ================ ALERT BELL + NOTIFICATIONS
def get_overdue_count(df):
    if df.empty: return 0
    df2=df.copy()
    df2["created_at"]=pd.to_datetime(df2["created_at"],errors="coerce")
    deadlines=df2["created_at"] + pd.to_timedelta(df2["sla_hours"].fillna(1), unit="h")
    return int((deadlines < datetime.now()).sum())

def alert_bell(df):
    cnt=get_overdue_count(df)
    return f"🔔 <span style='color:red;font-weight:800'>{cnt}</span>" if cnt>0 else "🔔 0"

# ===================== UI PAGES ======================

def page_pipeline():
    st.markdown("<h1>📌 TOTAL LEAD PIPELINE KPI</h1>", unsafe_allow_html=True)
    st.markdown("*Your lead funnel and SLA performance in one view*", unsafe_allow_html=True)

    df=load_leads_df()
    total=len(df)
    won=(df["stage"]=="Won").sum()
    lost=(df["stage"]=="Lost").sum()
    conv=(df["converted"].mean()*100) if total>0 else 0
    overdue=get_overdue_count(df)

    # KPI cards in 2 rows
    r1=st.columns(3);r2=st.columns(3)

    for i,(val,title) in enumerate([
        (total,"Total Leads"),(won,"Won"),(lost,"Lost")
    ]):
        r1[i].markdown(f"<div class='metric-card'><div class='kpi'>{val}</div><div class='metric-title'>{title}</div></div>", unsafe_allow_html=True)

    for i,(val,title) in enumerate([
        (f"{conv:.1f}%","Conversion"),(overdue,"SLA Overdue Leads"),(overdue,"Alerts Triggered")
    ]):
        r2[i].markdown(f"<div class='metric-card'><div class='kpi'>{val}</div><div class='metric-title'>{title}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='row-spacing'></div>", unsafe_allow_html=True)

    st.subheader("Pipeline Stage Flow")
    frame= df["stage"].value_counts().reindex(PIPELINE_STAGES,fill_value=0)
    st.bar_chart(frame)

    # TOP 5 PRIORITY LEADS as black cards
    st.subheader("🔥 TOP 5 Priority Leads")
    dfp=df.copy()
    dfp["created_at"]=pd.to_datetime(dfp["created_at"],errors="coerce")
    dfp=dfp[dfp["stage"].isin(["New","Qualified","Estimate Sent"])]
    dfp["deadline"]=dfp["created_at"] + pd.to_timedelta(dfp["sla_hours"].fillna(1),unit="h")
    dfp["hours_left"]=np.round((dfp["deadline"]-datetime.now()).dt.total_seconds()/3600,2)
    dfp["urgency"]=dfp["urgency"].fillna("Normal")
    dfp=dfp.sort_values("hours_left").head(5)

    c=st.columns(min(5,len(dfp)))
    for i,(_,row) in enumerate(dfp.iterrows()):
        c[i].markdown(f"""
        <div class='priority-card'>
            <div class='priority-title'>{row["lead_id"]}</div>
            <div>{row["urgency"]}</div>
            <div>⏳ {row["hours_left"]} hours left</div>
            <div>Owner: {row["owner"] if "owner" in row else "Unassigned"}</div>
        </div>
        """,unsafe_allow_html=True)

def page_lead_capture():
    st.header("🚀 Capture New Lead")
    df=load_leads_df()
    with st.form("cap"):
        lid=st.text_input("Lead ID")
        src=st.selectbox("Source",["Google Ads","Organic","Referral","Facebook Ads","Direct","Partner","Other"])
        stg=st.selectbox("Stage",PIPELINE_STAGES)
        est=st.number_input("Estimated Value",value=0.0)
        cost=st.number_input("Ad Cost",value=0.0)
        sla=st.number_input("SLA Response Time (hours)", value=1.0, min_value=0.1, step=0.5,
                             help="SLA Response time must be greater than 0 hours")
        notes=st.text_area("Notes")

        ok=st.form_submit_button("Save Lead")
        if ok:
            if not lid: st.error("Lead ID required")
            elif sla<=0: st.error("SLA must be greater than 0 hours")
            else:
                save_lead({"lead_id":lid,"source":src,"stage":stg,"estimated_value":est,"ad_cost":cost,"converted":1 if stg=="Won" else 0,"sla_hours":sla,"notes":notes})
                st.success("✅ Lead stored!")

def page_analytics():
    st.header("📈 Analytics & SLA Metrics")
    df=load_leads_df()
    stg=df["stage"].value_counts().reindex(PIPELINE_STAGES,fill_value=0)

    if not df.empty:
        fig=px.bar(df,x="stage",y="stage",title="Cost vs Conversion by Stage",labels={"stage":"Stage"},height=400)
        fig=px.bar(df,x="stage",y="ad_cost", title="💰 Cost vs Stage",height=400)
        st.plotly_chart(fig)

    st.subheader("SLA Status")
    df2=df.copy()
    df2["created_at"]=pd.to_datetime(df2["created_at"],errors="coerce")
    df2["deadline"]=df2["created_at"] + pd.to_datetime(df2["sla_hours"].fillna(1),unit="h")
    df2["overdue"]=np.where(df2["deadline"]<datetime.now(),"Overdue","On Track")
    st.dataframe(df2[["lead_id","sla_hours","overdue","urgency"]])

def page_settings():
    st.header("⚙️ Admin Settings")
    if st.session_state.role!="Admin":
        pwd=st.text_input("Admin Access Code",type="password")
        if st.button("Unlock Admin"):
            if pwd=="titan2025":
                st.session_state.role="Admin"
                st.success("🔓 ADMIN MODE ACTIVATED");st.rerun()
            else: st.error("❌ Wrong admin code")
    else:
        st.success("You are Admin")
        st.subheader("User Roles Management")
        owner=st.selectbox("Assign role to user", ["AyoBami","Estimator","Manager","Auditor","Field Agent"])
        role=st.selectbox("Role", ["Admin","Estimator","Manager","Auditor","Viewer"])
        if st.button("Save Role Assignment"):
            st.info(f"Role {role} assigned to {owner}")

def page_import_export():
    st.header("🔁 Import/Export Leads")
    df=load_leads_df()

    up=st.file_uploader("Import Leads Excel",type=["xlsx"])
    if up:
        dfi=pd.read_excel(up);insert_leads(dfi);st.success("✅ Imported!")
    if st.button("Export All Leads"):
        st.markdown(download_link(df,"titan_leads_export.xlsx"), unsafe_allow_html=True)

def page_ml():
    st.header("🤖 ML Lead Scoring Engine")
    df=load_leads_df()
    m,cols=load_lead_model()
    if not m:
        if st.button("Train ML Now"):
            acc,msg=train_lead_model(df)
            st.success(f"📊 Accuracy: {acc}");st.info(msg)
    else:
        st.success("✅ ML Model Loaded")
        df2=df.copy();df2["created_at"]=pd.to_datetime(df2["created_at"],errors="coerce")
        stg ="stage"
        X=pd.get_dummies(df2[["source","stage"]].astype(str)).reindex(columns=cols,fill_value=0)
        X["estimated_value"]=df2["estimated_value"].fillna(0)
        X["ad_cost"]=df2["ad_cost"].fillna(0)
        df2["prediction"]=m.predict(X)
        st.dataframe(df2[["lead_id","stage","prediction","urgency","sla_hours"]])

# =========== NAVIGATION =====================
df=load_leads_df()
top = st.columns([0.85,0.05,0.05,0.05])
if not st.session_state.user:
    top[1].text_input("Login Username",key="login_name")
    if top[2].button("Login"): do_login()
else:
    top[1].markdown(f"Welcome, {st.session_state.user}")
    if top[2].button("Logout"): do_logout()

top[3].markdown(f"<div class='alert-bell'>{alert_bell(df)}</div>", unsafe_allow_html=True)

nav=st.sidebar.selectbox("Go to", ["Pipeline","Lead Capture","Analytics","Settings","Import/Export","ML","ML"])
if nav=="Pipeline": page_pipeline()
if nav=="Lead Capture":page_lead_capture()
if nav=="Analytics":page_analytics()
if nav=="Settings":page_settings()
if nav=="Import/Export":page_import_export()
if nav=="ML":page_ml()
