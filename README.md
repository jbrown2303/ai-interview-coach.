# AI Interview Coach (Full, with Job Spec → Questions)

This is a complete **AI Interview Coach** app built in Streamlit.

## ✨ Features
- Question bank by role, type, and difficulty
- **NEW:** Generate questions from job descriptions (URL or pasted text)
- Timed answers + automated evaluation (STAR, conciseness, relevance, readability, filler words)
- Actionable STAR outline feedback
- Export attempts as JSON or Markdown
- Optional LLM feedback (plug in any OpenAI-compatible API key)
- Works offline with heuristics, or online with AI feedback
- Deployable to Streamlit Cloud for easy access on iPhone

## 🚀 Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📱 iPhone Access
Run locally with `--server.address 0.0.0.0` and open from Safari using your computer's IP.  
Or deploy to Streamlit Cloud and add to your home screen as an app.

