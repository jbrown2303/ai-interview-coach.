
import os, time, json, re, math, uuid, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import streamlit as st
import yaml
import textstat
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text

# ---------------- Utilities ----------------

FILLERS = {"um","uh","like","you know","sort of","kind of","basically","actually","literally","I guess"}
STAR_KEYS = {"s":["situation","background","context"],
             "t":["task","goal","objective","responsibility"],
             "a":["action","approach","what I did","how I did"],
             "r":["result","outcome","impact","metric","learned"]}
DEFAULT_QUESTION = "Tell me about a time you handled a difficult challenge."

@dataclass
class Attempt:
    id: str
    timestamp: float
    role: str
    qtype: str
    difficulty: str
    question: str
    answer: str
    duration_sec: int
    scores: Dict[str, float]
    feedback: str
    llm_feedback: str = ""

# ---------------- Load Questions ----------------

def load_questions(path="questions.yaml"):
    try:
        with open(path,"r") as f: return yaml.safe_load(f)
    except: return {}

QUESTIONS = load_questions()

# ---------------- Heuristic scoring ----------------

def token_count(text): return int(len(re.findall(r"\b\w+\b", text))*1.33)

def detect_fillers(text):
    t=text.lower(); return sum(len(re.findall(rf"\b{re.escape(f)}\b",t)) for f in FILLERS)

def star_coverage(text):
    t=text.lower(); return {k:1.0 if any(kw in t for kw in kws) else 0.0 for k,kws in STAR_KEYS.items()}

def relevance(q,a):
    qt=set(re.findall(r"\b\w+\b",q.lower())); at=set(re.findall(r"\b\w+\b",a.lower()))
    if not qt: return 0.0
    overlap=len(qt&at); return min(1.0, overlap/max(5,len(qt)*0.6))

def conciseness(a):
    words=len(re.findall(r"\b\w+\b",a))
    if words==0: return 0.0
    if 150<=words<=300: return 1.0
    if words<150: return max(0.0,words/150.0)
    return max(0.0,1.0-(words-300)/400.0)

def readability(a):
    try:
        g=textstat.flesch_kincaid_grade(a)
        if g<=0: return 0.5
        if 7<=g<=11: return 1.0
        if g<7: return max(0.6,g/7.0)
        return max(0.3,1.0-(g-11)/10.0)
    except: return 0.6

def structure_score(a):
    cov=star_coverage(a); return (0.2*cov["s"]+0.25*cov["t"]+0.3*cov["a"]+0.25*cov["r"])

def filler_penalty(a):
    words=len(re.findall(r"\b\w+\b",a)); 
    if words==0: return 0.0
    rate=detect_fillers(a)/words; return min(0.2,rate*20)

def overall_score(q,a):
    rel=relevance(q,a); stru=structure_score(a); conc=conciseness(a); read=readability(a); tok=token_count(a); pen=filler_penalty(a)
    base=(0.3*rel+0.3*stru+0.2*conc+0.2*read); base=max(0.0,min(1.0,base-pen))
    return {"relevance":round(rel*100,1),"structure":round(stru*100,1),"conciseness":round(conc*100,1),
            "readability":round(read*100,1),"tokens_est":tok,"final":round(base*100,1),"filler_penalty":round(pen*100,1)}

def suggest_outline(q,a):
    cov=star_coverage(a); miss=[k.upper() for k,v in cov.items() if v<1.0]; parts=" / ".join(miss) if miss else "All STAR elements present"
    return f"""**Suggested STAR outline**
- **Situation:** One-sentence context  
- **Task:** What you owned  
- **Action:** 3–5 bullets of what you did  
- **Result:** Quantify impact, and 1 lesson  
*Missing:* {parts}"""

# ---------------- LLM Feedback ----------------

def llm_feedback(api_base,api_key,q,a):
    if not api_base or not api_key: return ""
    try:
        import requests
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
        data={"model":"gpt-4o-mini","messages":[
            {"role":"system","content":"You are a strict interview coach. Give 3 bullets + 1-line summary feedback."},
            {"role":"user","content":f"Question: {q}\nAnswer: {a}"}
        ],"temperature":0.3}
        r=requests.post(f"{api_base.rstrip('/')}/chat/completions",headers=headers,json=data,timeout=20)
        if r.status_code==200: return r.json()["choices"][0]["message"]["content"]
        return f"(LLM error {r.status_code})"
    except Exception as e: return f"(LLM feedback error: {e})"

# ---------------- Job Spec Questions ----------------

def extract_text_from_url(url):
    try:
        r=requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200: return ""
        if "pdf" in r.headers.get("Content-Type","").lower() or url.lower().endswith(".pdf"):
            return pdf_extract_text(BytesIO(r.content))
        soup=BeautifulSoup(r.text,"html.parser")
        for t in soup(["script","style","noscript"]): t.extract()
        return "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())
    except: return ""

def derive_keywords(text):
    toks=re.findall(r"[A-Za-z]{3,}",text.lower()); stop=set("the and for with this that your you are can our etc".split())
    c={}
    for t in toks:
        if t in stop: continue
        c[t]=c.get(t,0)+1
    return [w for w,_ in sorted(c.items(), key=lambda x:x[1], reverse=True)[:6]]

def generate_spec_questions(raw):
    kws=derive_keywords(raw); out={"behavioral":[],"situational":[],"technical":[]}
    for kw in kws:
        out["behavioral"].append(f"Tell me about a time you worked with {kw}.")
        out["situational"].append(f"How would you handle challenges involving {kw}?")
        out["technical"].append(f"Explain your approach to {kw}.")
    return out

# ---------------- UI ----------------

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="wide")
st.title("🎤 AI Interview Coach")

with st.sidebar:
    st.subheader("Setup")
    roles=["general","software_engineer","product_manager","sales"]
    if "custom_pool" in st.session_state: roles.append("Custom (from job spec)")
    role=st.selectbox("Role",roles)
    qtype=st.selectbox("Type",["behavioral","situational","technical"])
    difficulty=st.selectbox("Difficulty",["easy","medium","hard"],index=1)

    st.markdown("---")
    st.subheader("LLM (optional)")
    use_llm=st.toggle("Use LLM feedback",value=False)
    api_base=st.text_input("API Base",disabled=not use_llm)
    api_key=st.text_input("API Key",type="password",disabled=not use_llm)

    st.markdown("---")
    st.subheader("Job Spec → Questions")
    spec_url=st.text_input("Job spec URL")
    spec_text=st.text_area("Or paste text",height=120)
    if st.button("Generate from spec"):
        raw=spec_text.strip() or extract_text_from_url(spec_url.strip())
        if raw:
            st.session_state["custom_pool"]=generate_spec_questions(raw)
            st.success("Custom questions ready! Choose 'Custom (from job spec)' role.")
        else:
            st.error("Could not extract job spec.")

    st.markdown("---")
    seconds_target=st.slider("Target answer time (sec)",60,300,120,30)

def get_question():
    if role=="Custom (from job spec)" and "custom_pool" in st.session_state:
        pool=st.session_state["custom_pool"].get(qtype,[])
        if pool: return random.choice(pool)
    try:
        return random.choice(QUESTIONS.get(role,{}).get(qtype,{}).get(difficulty,[]))
    except: return DEFAULT_QUESTION

if "history" not in st.session_state: st.session_state["history"]=[]

question=get_question()
st.subheader("Question")
st.write(question)

col1,col2=st.columns([2,1])
with col1: answer=st.text_area("Your answer",height=200)
with col2:
    st.write("⏱️ Timer")
    if st.button("Start timer"): st.session_state["timer_start"]=time.time()
    elapsed=int(time.time()-st.session_state.get("timer_start",time.time())) if "timer_start" in st.session_state else 0
    st.metric("Elapsed (s)",elapsed)

if st.button("Evaluate Answer",type="primary"):
    dur=int(time.time()-st.session_state.get("timer_start",time.time()))
    scores=overall_score(question,answer); outline=suggest_outline(question,answer)
    llm_fb=llm_feedback(api_base,api_key,question,answer) if use_llm else ""
    att=Attempt(str(uuid.uuid4()),time.time(),role,qtype,difficulty,question,answer,dur,scores,outline,llm_fb)
    st.session_state["history"].append(asdict(att))
    st.success("Evaluated!")

if st.session_state["history"]:
    st.subheader("Results")
    latest=st.session_state["history"][-1]; s=latest["scores"]
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Final",s["final"]); c2.metric("Rel",s["relevance"]); c3.metric("Struct",s["structure"])
    c4.metric("Concise",s["conciseness"]); c5.metric("Read",s["readability"])
    st.progress(min(1.0,s["final"]/100))
    st.markdown(latest["feedback"])
    if latest.get("llm_feedback"): st.info(latest["llm_feedback"])
    st.download_button("Download JSON",json.dumps(st.session_state["history"],indent=2),file_name="session.json")
