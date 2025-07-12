# resume_optimizer_app.py

import streamlit as st
import requests
import fitz  # PyMuPDF
from fpdf import FPDF
import unicodedata
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import re

# ---------------- Constants ----------------
PAGE_START = "start"
PAGE_LOGIN = "login"
PAGE_SIGNUP = "signup"
PAGE_UPLOAD_JD = "upload_jd"
PAGE_FILL_RESUME = "fill_resume"
PAGE_SHOW_RESULT = "show_result"

# ---------------- Environment ----------------
load_dotenv()
EMAIL_ADDRESS = "bbhavya3007@gmail.com"
EMAIL_PASSWORD = "cfedbynwsvxvxsqy"
BASE_URL = "http://127.0.0.1:8000"

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Resume Optimizer", layout="centered")
if "page" not in st.session_state:
    st.session_state.page = PAGE_START
if "optimized" not in st.session_state:
    st.session_state.optimized = ""

# ---------------- Utilities ----------------
def send_email_with_resume(subject, body, to, pdf_bytes, cc=None, bcc=None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename="optimized_resume.pdf")

    all_recipients = to + (cc if cc else []) + (bcc if bcc else [])
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg, to_addrs=all_recipients)
        return True, "✅ Email sent successfully."
    except Exception as e:
        return False, f"❌ Failed to send email: {e}"

def extract_job_title(jd: str) -> str:
    patterns = [
        r'(?i)\b(Job\s*Title|Position|Role)\s*[:\-\u2013]\s*(.+)',
        r'(?i)\bWe are looking for a[n]?\s+(.+?)\s+who',
        r'(?i)\bOpening for\s+(.+)',
        r'(?i)\bHiring\s+(.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, jd)
        if match:
            return match.group(2 if len(match.groups()) > 1 else 1).strip().title()
    return "Software Engineer"

def generate_pdf(resume_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Optimized Resume", ln=True, align="C")
    pdf.ln(5)

    cleaned = unicodedata.normalize("NFKD", resume_text).encode("latin-1", "ignore").decode("latin-1")
    lines = cleaned.split("\n")
    section_headers = [
        "EDUCATION", "ABOUT ME", "COURSEWORK", "SKILLS", "COURSEWORK/SKILLS", "PROJECTS",
        "CERTIFICATIONS", "CO-CURRICULAR", "WORK EXPERIENCE", "ACHIEVEMENTS",
        "SUMMARY", "EXPERIENCE", "INTERNSHIPS", "OBJECTIVE", "TECHNICAL SKILLS"
    ]

    pdf.set_font("Arial", size=11)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper() in section_headers:
            pdf.ln(3)
            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(0, 0, 128)
            pdf.cell(0, 10, line, ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("- ") or any(line.startswith(prefix) for prefix in ["*", "~", "–", "•"]):
            pdf.multi_cell(0, 8, "- " + line[2:] if len(line) > 2 else "-")
        else:
            pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1")

# ---------------- Pages ----------------
def start_page():
    st.title("Welcome to Resume Optimizer")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state.page = PAGE_LOGIN
    with col2:
        if st.button("Signup"):
            st.session_state.page = PAGE_SIGNUP

def signup_page():
    st.title("Signup")
    with st.form("signup_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Signup")
    if submit:
        if name and email and password:
            r = requests.post(f"{BASE_URL}/signup", json={"name": name, "email": email, "password": password})
            if r.status_code == 200:
                st.success("Signup successful. Please login.")
                st.session_state.page = PAGE_LOGIN
            else:
                st.error(r.json().get("detail", "Signup failed."))
        else:
            st.warning("Please fill all fields.")
    if st.button("Back"):
        st.session_state.page = PAGE_START

def login_page():
    st.title("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        if email and password:
            r = requests.post(f"{BASE_URL}/login", json={"email": email, "password": password})
            if r.status_code == 200:
                st.session_state.name = r.json().get("name", "User")
                st.session_state.page = PAGE_UPLOAD_JD
            else:
                st.error("Invalid credentials")
        else:
            st.warning("Please fill both fields.")
    if st.button("Back"):
        st.session_state.page = PAGE_START

def upload_jd_page():
    st.title("Upload Job Description")
    with st.form("jd_form"):
        jd_text = st.text_area("Paste your job description here:")
        submit = st.form_submit_button("Continue")
    if submit and jd_text:
        st.session_state.jd = jd_text
        st.session_state.page = PAGE_FILL_RESUME

def fill_resume_page():
    st.title("Resume Input")
    uploaded_file = st.file_uploader("Upload an existing resume (PDF or TXT)", type=["pdf", "txt"])
    structured = {}
    parsed_resume = ""

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            parsed_resume = "\n".join([page.get_text() for page in doc])
        else:
            parsed_resume = uploaded_file.read().decode("utf-8")

        with st.spinner("Parsing resume using AI..."):
            res = requests.post(f"{BASE_URL}/extract_resume_data", json={"resume": parsed_resume})
            
            if res.status_code == 200:
                structured = res.json()
                st.success("Resume parsed and fields filled.")
            else:
                st.warning("Parsing failed. Please fill manually.")

    name = st.text_input("Full Name*", value=structured.get("name", ""))
    contact = st.text_input("Contact Number*", value=structured.get("contact", ""))
    email = st.text_input("Email*", value=structured.get("email", ""))
    linkedin = st.text_input("LinkedIn", value=structured.get("linkedin", ""))
    github = st.text_input("GitHub", value=structured.get("github", ""))
    about_me = st.text_area("About Me", value=structured.get("summary", ""), height=100)
    education_text = st.text_area("Education", value=structured.get("education", ""), height=150)
    projects_text = st.text_area("Projects", value=structured.get("projects", ""), height=150)
    skills = st.text_area("Skills* (comma separated)", value=", ".join(structured.get("skills", [])))
    certs = st.text_area("Certifications", value=structured.get("certifications", ""))
    work = st.text_area("Work Experience", value=structured.get("work_experience", ""))
    achv = st.text_area("Achievements", value=structured.get("achievements", ""))

    if st.button("Submit & Optimize Resume"):
        missing = [field for field, value in [("Full Name", name), ("Contact Number", contact), ("Email", email), ("Skills", skills), ("Education", education_text)] if not value.strip()]
        if missing:
            st.warning("Please fill in required fields:\n- " + "\n- ".join(missing))
        else:
            resume = f"""
Name: {name}
Contact: {contact}
Email: {email}
LinkedIn: {linkedin}
GitHub: {github}

EDUCATION
{education_text.strip()}

ABOUT ME
{about_me.strip()}

SKILLS
{skills.strip()}

PROJECTS
{projects_text.strip()}

CERTIFICATIONS
{certs.strip()}

WORK EXPERIENCE
{work.strip()}

ACHIEVEMENTS
{achv.strip()}
"""
            send_to_backend(st.session_state.jd, resume)

def send_to_backend(jd, resume):
    with st.spinner("Optimizing resume..."):
        r = requests.post(f"{BASE_URL}/optimize_resume", json={"jd": jd, "resume": resume})
        if r.status_code == 200:
            st.session_state.optimized = r.json()["optimized_resume"]
            st.session_state.page = PAGE_SHOW_RESULT
        else:
            st.error("Something went wrong while optimizing.")

def result_page():
    st.title("Optimized Resume")
    st.text_area("Here is your optimized resume:", st.session_state.optimized, height=500)
    pdf_bytes = generate_pdf(st.session_state.optimized)
    name = "Candidate"
    job_title = extract_job_title(st.session_state.jd)

    st.download_button("Download as PDF", data=pdf_bytes, file_name="optimized_resume.pdf", mime="application/pdf")

    st.subheader("📧 Send Resume via Email")
    with st.form("email_form"):
        recipient = st.text_input("Recipient Email")
        cc = st.text_input("CC (optional)", placeholder="comma-separated if multiple")
        subject = st.text_input("Subject", value=f"Application for {job_title} – {name}")
        body = st.text_area("Message", value=f"""Dear Hiring Team,

I am writing to express my interest in the {job_title} position. Please find my optimized resume attached.

Thank you for your consideration.
– {name}
""")
        send_btn = st.form_submit_button("Send Email")

    if send_btn:
        if not recipient or "@" not in recipient:
            st.error("Please enter a valid recipient email.")
        else:
            cc_list = [x.strip() for x in cc.split(",")] if cc else []
            success, msg = send_email_with_resume(
                subject=subject,
                body=body,
                to=[recipient],
                cc=cc_list,
                pdf_bytes=pdf_bytes
            )
            st.success(msg) if success else st.error(msg)

    if st.button("Start Over"):
        st.session_state.page = PAGE_START
        st.session_state.optimized = ""

# ---------------- Main Entry ----------------
def main():
    page = st.session_state.page
    if page == PAGE_START: start_page()
    elif page == PAGE_SIGNUP: signup_page()
    elif page == PAGE_LOGIN: login_page()
    elif page == PAGE_UPLOAD_JD: upload_jd_page()
    elif page == PAGE_FILL_RESUME: fill_resume_page()
    elif page == PAGE_SHOW_RESULT: result_page()

if __name__ == "__main__":
    main()
