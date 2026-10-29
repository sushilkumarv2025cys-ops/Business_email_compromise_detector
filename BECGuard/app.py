import sys, json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"src"))
from predict import analyze, get_model

st.set_page_config(page_title="BECGuard",page_icon="🛡️",layout="wide")

@st.cache_resource(show_spinner=False)
def _warm():
    return get_model()

@st.cache_data(show_spinner=False)
def load_metrics(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None

_warm()

st.title("🛡️ BECGuard")
st.markdown("### Explainable Business Email Compromise Risk Detector")
st.caption("Academic ML cybersecurity prototype • Screens for BEC-like social-engineering indicators")

with st.sidebar:
    st.header("Email Metadata")
    sender=st.text_input("Sender","hr@example.com")
    subject=st.text_input("Subject","Team Meeting Reminder")
    reply_to=st.text_input("Reply-To","")
    external=st.checkbox("External sender",False)
    st.divider()
    st.write("**Score bands**")
    st.write("🟢 0–29  Low")
    st.write("🟡 30–69  Medium")
    st.write("🔴 70–100  High")
    st.caption("A score is a screening signal, not proof of malicious intent.")

email=st.text_area("Email content",height=250,placeholder="Paste email body here...")
if st.button("🔍 Analyze Email",type="primary",use_container_width=True):
    if not email.strip():
        st.error("Enter an email message first.")
        st.stop()

    r=analyze(email,sender,subject,reply_to,external)
    st.divider()

    c1,c2,c3=st.columns(3)
    c1.metric("BEC Risk Score",f"{r['score']}/100")
    c2.metric("Risk Level",r["level"])
    c3.metric("Indicators",len(r["indicators"])+len(r["metadata_indicators"]))

    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=r["score"],title={"text":"Risk Score"},
        gauge={"axis":{"range":[0,100]},"steps":[
            {"range":[0,30]},{"range":[30,70]},{"range":[70,100]}
        ]}
    ))
    fig.update_layout(height=280,margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)

    a,b=st.columns(2)
    with a:
        st.subheader("🚩 Detected Indicators")
        allflags=r["indicators"]+r["metadata_indicators"]
        if allflags:
            for x in allflags: st.write("✓",x)
        else: st.success("No predefined BEC indicators detected.")
    with b:
        st.subheader("🧠 Explanation")
        for x in r["reasons"]: st.write("•",x)
        if r["level"]=="HIGH":
            st.warning("Verify sensitive requests through an independent channel before acting.")
        elif r["level"]=="MEDIUM":
            st.info("Review the sender, context, and requested action before responding.")
        else:
            st.success("The message shows limited evidence of BEC-like behavior.")

    with st.expander("Model details"):
        st.write(f"Raw ML probability: {r['probability']:.3f}")
        st.write("The final score also considers explicit BEC indicators and selected metadata signals.")

st.divider()
st.subheader("📊 Model Evaluation")
m=load_metrics(str(BASE/"reports/metrics.json"))
if m:
    e1,e2,e3,e4=st.columns(4)
    e1.metric("Precision",f"{m['precision']:.3f}")
    e2.metric("Recall",f"{m['recall']:.3f}")
    e3.metric("F1",f"{m['f1']:.3f}")
    e4.metric("Test Emails",m["test_size"])
    st.caption(f"{m.get('evaluation','Hold-out test')} on the included synthetic starter dataset; replace it with a larger public/anonymized dataset for final evaluation.")

st.caption("BECGuard v2 • Use anonymized/public data only.")
