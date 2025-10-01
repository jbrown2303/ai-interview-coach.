
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
from collections import Counter

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
- **Result:** Quantify impact + 1 lesson  
*Missing:* {parts}"""

# ---------------- LLM Feedback ----------------

def llm_feedback(api_base,api_key,q,a):
    if not api_base or not api_key: return ""
    try:
        import requests
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
        data={"model":"gpt-4o-mini","messages":[
            {"role":"system","content":"You are a strict interview coach. Give 3 bullets + 1-line summary feedback, UK English."},
            {"role":"user","content":f"Question: {q}\nAnswer: {a}\nPlease score STAR, specificity, impact. Suggest 1 stronger RESULT sentence."}
        ],"temperature":0.3}
        r=requests.post(f"{api_base.rstrip('/')}/chat/completions",headers=headers,json=data,timeout=20)
        if r.status_code==200: return r.json()["choices"][0]["message"]["content"]
        return f"(LLM error {r.status_code})"
    except Exception as e: return f"(LLM feedback error: {e})"

# ---------------- Job Spec Extraction (clean keywords) ----------------

_STOP = set("""
a an the and or for with from this that these those to of in on by as is are be
will would could should can may might must your you you'll we we'll our they their
role roles responsibility responsibilities requirement requirements about if have has had
do did done make made get got work working team teams department company organisation organization
experience experiences strong excellent good ability etc using use used across within
""".split())

_WHITELIST = set("""
sql aws gcp azure sap crm seo ppc ga4 gdpr kpi okr ui ux qa api ml nlp etl bi
ci cd cicd devops saas b2b b2c b2g pm ba kpis okrs sme smes
""".split())

_PREFERENCE = (
    "stakeholder", "management", "roadmap", "discovery", "experimentation",
    "go-to-market", "customer", "retention", "acquisition", "churn", "pipeline",
    "forecast", "data", "quality", "governance", "privacy", "security", "gdpr",
    "analytics", "dashboard", "reporting", "product", "design", "research",
    "usability", "a/b", "experiments", "engineering", "architecture",
    "scalability", "reliability", "marketing", "campaign", "seo", "ppc",
    "paid", "creative", "salesforce", "hubspot", "tableau", "power", "bi",
    "excel", "python", "java", "react", "node", "sql", "aws"
)

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

def _clean_jd_text(t: str) -> str:
    t = re.sub(r"(?i)\byou\s+will\b", " ", t)
    t = re.sub(r"(?i)\bwe\s+will\b", " ", t)
    t = re.sub(r"(?i)\byou’ll\b", " ", t)
    t = re.sub(r"(?i)\byou'll\b", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t

def _tokenize_keep_acronyms(text: str):
    return re.findall(r"[A-Za-z][A-Za-z0-9\-\+/]{1,}", text.lower())

def _bigrams(tokens):
    return [" ".join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)]

def _score_candidates(unigrams, bigrams):
    cu = Counter(unigrams)
    cb = Counter(bigrams)
    for w in list(cu.keys()):
        if w in _STOP and w not in _WHITELIST:
            cu[w] = 0
    for p in list(cb.keys()):
        if any(pref in p for pref in _PREFERENCE):
            cb[p] *= 2
    for w in list(cu.keys()):
        if w in _WHITELIST:
            cu[w] *= 2
    return cu, cb

def derive_keywords(text: str, k: int = 8) -> List[str]:
    t = _clean_jd_text(text)
    toks = _tokenize_keep_acronyms(t)
    toks = [w for w in toks if len(w) >= 3 or w in _WHITELIST]
    bigs = _bigrams(toks)
    cu, cb = _score_candidates(toks, bigs)
    top_bigrams = [w for w, c in cb.most_common(20) if c > 1]
    used = set(sum((b.split() for b in top_bigrams), []))
    top_unigrams = [w for w, c in cu.most_common(30) if c > 1 and w not in _STOP and w not in used]
    tech_hits = [w for w in _WHITELIST if w in toks]
    candidates = top_bigrams + tech_hits + top_unigrams
    def _ok(term: str) -> bool:
        parts = term.split()
        return all((p not in _STOP) for p in parts)
    cleaned = []
    seen = set()
    for term in candidates:
        if not _ok(term): 
            continue
        if term in seen: 
            continue
        seen.add(term); cleaned.append(term)
    return cleaned[:k] if cleaned else ["key responsibilities"]

def generate_spec_questions(raw_text: str) -> Dict[str, List[str]]:
    kws = derive_keywords(raw_text, k=8)
    out = {"behavioral": [], "situational": [], "technical": []}
    behav_templates = [
        "Tell me about a time you delivered results related to {kw}. What was the impact?",
        "Describe a challenging situation involving {kw}. How did you handle it?",
        "Give an example of when you improved a process around {kw}. What changed?"
    ]
    situ_templates = [
        "You’ve joined a team where {kw} is underperforming. What do you do in the first 30 days?",
        "A key dependency for {kw} slips. How do you reprioritise and communicate risk?",
        "How would you measure success for initiatives involving {kw}?"
    ]
    tech_templates = [
        "Walk me through how you would design or implement a solution for {kw}.",
        "What trade-offs would you consider when scaling {kw}?",
        "How would you diagnose and fix recurring issues related to {kw}?"
    ]
    for i, kw in enumerate(kws):
        out["behavioral"].append(behav_templates[i % len(behav_templates)].format(kw=kw))
        out["situational"].append(situ_templates[i % len(situ_templates)].format(kw=kw))
        out["technical"].append(tech_templates[i % len(tech_templates)].format(kw=kw))
    for k in out:
        seen, uniq = set(), []
        for q in out[k]:
            if q not in seen:
                seen.add(q); uniq.append(q)
        out[k] = uniq[:12]
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
    api_base=st.text_input("API Base",disabled=not use_llm, placeholder="https://api.openai.com/v1")
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
with col1: answer=st.text_area("Your answer",height=200, placeholder="Use STAR → Situation, Task, Action, Result")
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
    st.success("Evaluated! Scroll down for results.")

if st.session_state["history"]:
    st.subheader("Results")
    latest=st.session_state["history"][-1]; s=latest["scores"]
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Final",s["final"]); c2.metric("Rel",s["relevance"]); c3.metric("Struct",s["structure"])
    c4.metric("Concise",s["conciseness"]); c5.metric("Read",s["readability"])
    st.progress(min(1.0,s["final"]/100))
    with st.expander("Detailed feedback", expanded=True):
        st.markdown(latest["feedback"])
        st.caption(f'Estimated tokens: {s["tokens_est"]} | Filler penalty: -{s["filler_penalty"]} pts')
        if latest.get("llm_feedback"): st.info(latest["llm_feedback"])
    st.download_button("Download session (JSON)",json.dumps(st.session_state["history"],indent=2),file_name="session.json")
