import streamlit as st
import pandas as pd
from resume_parser import parse_resume 
st.set_page_config(page_title="Resume Tailor", layout="wide")
st.title("Resume Tailor - AI Resume Optimizer")
uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])
uploaded_jd = st.file_uploader("Upload Job Description", type=["txt", "docx"])

if uploaded_resume and uploaded_jd:
    resume_text = parse_resume(uploaded_resume)
    jd_text = uploaded_jd.read().decode("utf-8")
    st.subheader("Resume Extracted Text")
    st.write(resume_text[:500])  # preview
    st.subheader("Job Description")
    st.write(jd_text[:500])
    st.success("Resume parsed successfully! More features coming soon")
