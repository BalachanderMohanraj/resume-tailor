import streamlit as st
import pdfplumber
import docx
import re
import spacy
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_models():
    return spacy.load("en_core_web_sm"), SentenceTransformer("all-MiniLM-L6-v2")

nlp, sem_model = load_models()

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_email(text):
    m = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
    return m.group(0) if m else ""

def extract_phone(text):
    m = re.search(r'(\+?\d[\d\-\s]{8,}\d)', text)
    return m.group(0).strip() if m else ""

def extract_name(text):
    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""

def extract_location(text):
    for ent in nlp(text).ents:
        if ent.label_ == "GPE":
            return ent.text
    return ""

def extract_skills(text, candidates):
    found = [s for s in candidates if re.search(rf'\b{re.escape(s)}\b', text, re.IGNORECASE)]
    return list(set(found))

def semantic_skill_match(resume_skills, job_skills, threshold=0.7):
    if not resume_skills or not job_skills:
        return [], job_skills, 0.0
    res_emb = sem_model.encode(resume_skills, convert_to_tensor=True)
    job_emb = sem_model.encode(job_skills, convert_to_tensor=True)
    scores = util.cos_sim(res_emb, job_emb).cpu().numpy()
    matched = {job_skills[j] for j in range(len(job_skills))
               if any(scores[i][j] >= threshold for i in range(len(resume_skills)))}
    missing = [s for s in job_skills if s not in matched]
    coverage = len(matched) / len(job_skills)
    return list(matched), missing, coverage

def save_excel(records, fname="candidate_data.xlsx"):
    df = pd.DataFrame(records)
    return df.to_excel(fname, index=False) or fname

st.title("🚀 AI Resume Analyzer (Transformer‑based)")

jd_file = st.file_uploader("Upload Job Description (JSON)", type="json")
if not jd_file:
    st.stop()
job = json.load(jd_file)
job_skills = job.get("skills", [])
job_certs = job.get("certifications", [])

res_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
if not res_file:
    st.info("Upload resume to analyze.")
    st.stop()

if res_file.type == "application/pdf":
    text = extract_text_from_pdf(res_file)
else:
    text = extract_text_from_docx(res_file)

name = extract_name(text)
email = extract_email(text)
phone = extract_phone(text)
location = extract_location(text)
skills = extract_skills(text, job_skills)
certs = extract_skills(text, job_certs)

matched, missing, coverage = semantic_skill_match(skills, job_skills)

st.header("Candidate Profile")
st.write(f"**Name:** {name or '—'}")
st.write(f"**Email:** {email or '—'}")
st.write(f"**Phone:** {phone or '—'}")
st.write(f"**Location:** {location or '—'}")
st.write(f"**Skills Detected:** {', '.join(skills) or 'None'}")
st.write(f"**Certifications Detected:** {', '.join(certs) or 'None'}")

st.header("Job Fit Analysis")
st.metric("Skill Coverage", f"{coverage:.0%}")
st.write(f"Required Skills: {', '.join(job_skills)}")
st.write(f"Matched Skills: {', '.join(matched) or '—'}")
if missing:
    st.write(f"Missing: {', '.join(missing)}")

record = {
    "name": name, "email": email, "phone": phone, "location": location,
    "skills": ", ".join(skills), "certifications": ", ".join(certs),
    "skill_coverage": f"{coverage:.0%}"
}

save_excel([record], "candidate.xlsx")
with open("candidate.xlsx", "rb") as f:
    st.download_button("Download Candidate Data (Excel)", f, file_name="candidate.xlsx")

