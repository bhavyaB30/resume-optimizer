import streamlit as st
import os
import fitz  # PyMuPDF
import logging
import requests
import unicodedata
import smtplib
import re
import json
from dotenv import load_dotenv
from email.message import EmailMessage
from pydantic import BaseModel
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.llms import ChatMessage as LlamaChatMessage
from llama_index.core.output_parsers import PydanticOutputParser
from PIL import Image


# ---------------- Environment & Setup ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
EMAIL_ADDRESS = "bbhavya3007@gmail.com"
EMAIL_PASSWORD =  "cfedbynwsvxvxsqy"
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Resume Optimizer", layout="centered")
if "page" not in st.session_state:
    st.session_state.page = "start"
if "optimized" not in st.session_state:
    st.session_state.optimized = ""

# ---------------- Gemini LLM Setup ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    convert_system_message_to_human=True,
    google_api_key=""  # Set in .env
)
wrapped_llm = LangChainLLM(llm=llm)

class ResumeStructured(BaseModel):
    name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""
    about_me: Optional[str] = ""
    linkedin: Optional[str] = ""
    github: Optional[str] = ""
    education: Optional[List[str]] = []
    experience: Optional[List[str]] = []
    skills: Optional[List[str]] = []
    projects: Optional[List[str]] = []
    certifications: Optional[List[str]] = []
    co_curricular_activities: Optional[List[str]] = []

parser = PydanticOutputParser(output_cls=ResumeStructured)

def extract_skills_from_jd_with_llm(jd_text: str) -> List[str]:
    prompt = f"""
You are an expert resume analyzer.

Given the following job description, extract and return a JSON array of **distinct skills** mentioned in it.
Return only the JSON array with no explanation.

Job Description:
\"\"\"{jd_text}\"\"\"
"""
    messages = [LlamaChatMessage(role="user", content=prompt)]
    response = wrapped_llm.chat(messages)
    print(response)
    try:
        content = response.message.content.strip()

        # Remove Markdown formatting like ```json ... ```
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        # Optional: remove "assistant:" label if present
        content = content.replace("assistant:", "").strip()

        skills = json.loads(content)
        print(skills)
        if isinstance(skills, list):
            print(sorted(set(skill.strip() for skill in skills if isinstance(skill, str))))
            return sorted(set(skill.strip() for skill in skills if isinstance(skill, str)))
    except Exception as e:
        logging.warning(f"LLM skill extraction failed: {e}")
    
    return []

def extract_resume_fields(text: str) -> ResumeStructured:
    prompt = f"""
Extract the following information from the resume text and return it in JSON format:

name: str
email: str
phone: str
about_me: str (optional)
linkedin: str (optional)
github: str (optional)
education: list of strings
experience: list of strings
skills: list of strings
projects: list of strings
certifications: list of strings
co_curricular_activities: list of strings (optional)

Resume:
\"\"\"{text}\"\"\"
"""
    messages = [LlamaChatMessage(role="user", content=prompt)]
    response = wrapped_llm.chat(messages)
    return parser.parse(response.message.content)

def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

# ---------------- Utility Functions ----------------
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
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Optimized Resume", ln=True, align="C")
    pdf.ln(5)
    cleaned = unicodedata.normalize("NFKD", resume_text).encode("latin-1", "ignore").decode("latin-1")
    section_headers = [
        "EDUCATION", "ABOUT ME", "COURSEWORK", "SKILLS", "PROJECTS",
        "CERTIFICATIONS", "CO-CURRICULAR", "WORK EXPERIENCE", "ACHIEVEMENTS",
        "SUMMARY", "EXPERIENCE", "INTERNSHIPS", "OBJECTIVE", "TECHNICAL SKILLS"
    ]
    pdf.set_font("Arial", size=11)
    for line in cleaned.split("\n"):
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
        else:
            pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1")

def extract_name(text):
    match = re.findall(r"(?i)(?:name\s*[:\-]?\s*|my name is\s+|i am\s+|this is\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})", text)
    if match:
        return match[0].strip()
    return "Candidate"
def generate_application_email(name: str, job_title: str, matched_skills: list) -> str:
    prompt = f"""

you are a professional EMail Genrartor and you have to write a email for a job application 
based on the below details write a formal email to the hiring manager 

Candidate: {name}
Role: {job_title}
Skills: {', '.join(matched_skills)}
do not add subject line.
Start with a greeting, justify the application using the skills, and end formally.
Don't mention attachments or resume.

Output only the message body in a proper structured format without any additional text or explanation.

"""
    messages = [LlamaChatMessage(role="user", content=prompt)]
    response = wrapped_llm.chat(messages)
    return response.message.content.strip()


def display_logo():
    st.image("../assets/image.png", width=150)

    

# ---------------- Pages ----------------
def start_page():
    st.title("CVOpsify - Resume Optimizer")
    display_logo()
    col1, col2 = st.columns(2)
    if col1.button("Login"): st.session_state.page = "login"
    if col2.button("Signup"): st.session_state.page = "signup"

def signup_page():
    display_logo()
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
                st.session_state.page = "login"
            else:
                st.error(r.json().get("detail", "Signup failed."))
        else:
            st.warning("Please fill all fields.")
    if st.button("Back"): st.session_state.page = "start"

def login_page():
    display_logo()
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
                st.session_state.page = "upload_jd"
            else:
                st.error("Invalid credentials")
        else:
            st.warning("Please fill both fields.")
    if st.button("Back"): st.session_state.page = "start"

# def upload_jd_page():
#     st.session_state.page = "upload_jd"
#     st.title("🎯 Define Your Dream Role") 
#     display_logo()
#     with st.form("jd_form"):
#         jd_text = st.text_area("Paste your job description here:")
#         submit = st.form_submit_button("Continue")
#     if submit and jd_text:
#         st.session_state.jd = jd_text
#         with st.spinner("Extracting skills from JD using Gemini..."):
#             st.session_state.jd_skills = extract_skills_from_jd_with_llm(jd_text)
#         st.success("✅ Skills extracted from JD successfully!")
#         st.session_state.page = "fill_resume"
def upload_jd_page():
    display_logo()
    st.session_state.page = "upload_jd"
    st.markdown("""
        <h1 style="font-size: 2.8rem; margin-bottom: 0.2em;">🎯 Define Your Dream Role</h1>
        <p style="font-size: 1.1rem; color: #666;">
            Paste the job description below and we’ll tailor your resume to match it perfectly.
        </p>
    """, unsafe_allow_html=True)

    

    with st.form("jd_form"):
        jd_text = st.text_area("Paste your job description here:")
        submit = st.form_submit_button("Continue")

    if submit and jd_text:
        st.session_state.jd = jd_text
        with st.spinner("Extracting skills from JD using Gemini..."):
            st.session_state.jd_skills = extract_skills_from_jd_with_llm(jd_text)
        st.success("✅ Skills extracted from JD successfully!")
        st.session_state.page = "fill_resume"


def fill_resume_page():
    display_logo()
    st.title("📄 Upload or Create Resume")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
    structured = {}

    if uploaded_file:
        parsed_resume = extract_text_from_pdf(uploaded_file)
        with st.spinner("Parsing resume using Gemini..."):
            try:
                result = extract_resume_fields(parsed_resume)
                structured = result.model_dump()
                st.success("✅ Resume parsed successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        structured = {}

    def safe_get(key):
        val = structured.get(key, "")
        return ", ".join(val) if isinstance(val, list) else val or ""

    def safe_strip(val):
        return (val or "").strip()

    name = st.text_input("Full Name*", value=safe_get("name"))
    st.session_state.name_for_email = name
    contact = st.text_input("Contact Number*", value=safe_get("phone"))
    email = st.text_input("Email*", value=safe_get("email"))
    linkedin = st.text_input("LinkedIn", value=safe_get("linkedin"))
    github = st.text_input("GitHub", value=safe_get("github"))

    about_me = st.text_area("About Me", value=safe_get("about_me"), height=100)
    education_text = st.text_area("Education", value="\n".join(structured.get("education", [])), height=100)
    experience_text = st.text_area("Experience", value="\n".join(structured.get("experience", [])), height=100)
    projects_text = st.text_area("Projects", value="\n".join(structured.get("projects", [])), height=100)
    skills = st.text_area("Skills*", value=", ".join(structured.get("skills", [])))
    certs = st.text_area("Certifications", value="\n".join(structured.get("certifications", [])))
    cca = st.text_area("Co-Curricular Activities", value="\n".join(structured.get("co_curricular_activities", [])))

    if st.button("Submit & Optimize Resume"):
        missing = [f for f, v in [("Full Name", name), ("Contact Number", contact), ("Email", email), ("Skills", skills)] if not safe_strip(v)]

        if missing:
            st.warning("Please fill in required fields:\n- " + "\n- ".join(missing))
        else:
            resume = f"""
Name: {safe_strip(name)}
Contact: {safe_strip(contact)}
Email: {safe_strip(email)}
LinkedIn: {safe_strip(linkedin)}
GitHub: {safe_strip(github)}

ABOUT ME
{safe_strip(about_me)}

EDUCATION
{safe_strip(education_text)}

EXPERIENCE
{safe_strip(experience_text)}

PROJECTS
{safe_strip(projects_text)}

SKILLS
{safe_strip(skills)}

CERTIFICATIONS
{safe_strip(certs)}

CO-CURRICULAR ACTIVITIES
{safe_strip(cca)}
"""
            
            send_to_backend(st.session_state.jd, resume)


def send_to_backend(jd, resume):
    with st.spinner("Optimizing resume..."):
        try:
            r = requests.post(f"{BASE_URL}/optimize_resume", json={"jd": jd, "resume": resume})
            if r.status_code == 200:
                res_data = r.json()

                # ✅ Extract optimized resume and missing skills
                optimized_resume = res_data.get("optimized_resume")
                missing_skills = res_data.get("missing_skills", [])

                if not isinstance(optimized_resume, dict):
                    st.error("❌ Unexpected backend response format.")
                    return

                st.session_state.optimized_json = optimized_resume
                st.session_state.missing_skills = missing_skills

                # ✅ Prepare formatted resume for PDF/email
                resume_text = f"""
Name: {optimized_resume.get("name", "")}
Contact: {optimized_resume.get("contact", "")}
Email: {optimized_resume.get("email", "")}
LinkedIn: {optimized_resume.get("linkedin", "")}
GitHub: {optimized_resume.get("github", "")}

SUMMARY
{optimized_resume.get("summary", "")}

EDUCATION
{optimized_resume.get("education", "")}

SKILLS
{", ".join(optimized_resume.get("skills", []))}

PROJECTS
{optimized_resume.get("projects", "")}

CERTIFICATIONS
{optimized_resume.get("certifications", "")}

WORK EXPERIENCE
{optimized_resume.get("work_experience", "")}

ACHIEVEMENTS
{optimized_resume.get("achievements", "")}
""".strip()

                st.session_state.optimized = resume_text
                st.session_state.page = "show_result"

            else:
                st.error(f"❌ Backend error {r.status_code}: {r.text}")

        except Exception as e:
            st.error(f"❌ Exception: {e}")
def result_page():
    display_logo()
    st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>Resume Insights Dashboard </h1>",
    
    unsafe_allow_html=True
)

    # st.title("📄 Optimized Resume")
    
    

    # ✅ Optimized Resume JSON Display
    st.subheader("🗂️ Resume Content Breakdown")
    st.json(st.session_state.optimized_json)

    # ✅ Initialize sets
    if "added_skills" not in st.session_state:
        st.session_state.added_skills = set()

    jd_skills_raw = st.session_state.get("jd_skills", [])
    jd_skills = set(skill.lower() for skill in jd_skills_raw)
    resume_skills = set(skill.lower() for skill in st.session_state.optimized_json.get("skills", []))

    # ✅ Recalculate matched skills
    matched_skills = sorted([skill for skill in jd_skills_raw if skill.lower() in resume_skills])

    # ✅ Styled Missing Skills Section
    missing_skills = st.session_state.get("missing_skills", [])
    st.markdown(
        """
             <div style="
        background-color: #2e2e2e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #666;
        margin-bottom: 10px;
    ">
        <h4 style="color: #f5f5f5; margin: 0;">🔍 Not Yet in Your Resume</h4>
    </div>
        """,
        unsafe_allow_html=True
    )

    for skill in missing_skills:
        col1, col2 = st.columns([0.85, 0.15])
        col1.markdown(f"<li style='font-size: 16px; color: #000000;'>{skill}</li>", unsafe_allow_html=True)

        if skill.lower() in resume_skills:
            if col2.button("➖", key=f"remove_{skill}"):
                st.session_state.optimized_json["skills"].remove(skill)
                st.session_state.added_skills.discard(skill)
        else:
            if col2.button("➕", key=f"add_{skill}"):
                st.session_state.optimized_json["skills"].append(skill)
                st.session_state.added_skills.add(skill)

    # ✅ Matched Skills Section
    st.markdown(
        """
     <div style="
        background-color: #2e2e2e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #666;
        margin: 20px 0 10px 0;
    ">
        <h4 style="color: #f5f5f5; margin: 0;">🎯 Already Covered JD Skills</h4>
    </div>
    
        """,
        unsafe_allow_html=True
    )

    if matched_skills:
        for skill in matched_skills:
            st.markdown(f"<li style='font-size: 16px; color: #000000;'>{skill}</li>", unsafe_allow_html=True)
    else:
        st.info("No matched skills found — all JD skills are currently missing in the resume.")

    # ✅ Download Missing Skills JSON
    # missing_skills_json_bytes = json.dumps(missing_skills, indent=2).encode("utf-8")
    # st.download_button(
    #     label="📥 Download Missing Skills (JSON)",
    #     data=missing_skills_json_bytes,
    #     file_name="missing_skills.json",
    #     mime="application/json"
    # )

    # # ✅ Download Optimized Resume JSON
    # optimized_json_bytes = json.dumps(st.session_state.optimized_json, indent=2).encode("utf-8")
    # st.download_button(
    #     label="📥 Download Optimized Resume (JSON)",
    #     data=optimized_json_bytes,
    #     file_name="optimized_resume.json",
    #     mime="application/json"
    # )

    # ✅ Generate Optimized Resume Text for PDF
    optimized_resume = st.session_state.optimized_json
    resume_text = f"""
Name: {optimized_resume.get("name", "")}
Contact: {optimized_resume.get("contact", "")}
Email: {optimized_resume.get("email", "")}
LinkedIn: {optimized_resume.get("linkedin", "")}
GitHub: {optimized_resume.get("github", "")}

SUMMARY
{optimized_resume.get("summary", "")}

EDUCATION
{optimized_resume.get("education", "")}

SKILLS
{", ".join(optimized_resume.get("skills", []))}

PROJECTS
{optimized_resume.get("projects", "")}

CERTIFICATIONS
{optimized_resume.get("certifications", "")}

WORK EXPERIENCE
{optimized_resume.get("work_experience", "")}

ACHIEVEMENTS
{optimized_resume.get("achievements", "")}
""".strip()

    pdf_bytes = generate_pdf(resume_text)
    name = st.session_state.get("name_for_email") or extract_name(resume_text)
    job_title = extract_job_title(st.session_state.jd)
    st.markdown(
    """
    <style>
    .button-style {
        display: inline-flex;
        align-items: center;
        font-weight: 500;
        padding: 10px 18px;
        border-radius: 8px;
        border: 1px solid #ccc;
        background-color: white;
        color: black;
        margin-bottom: 10px;
        font-size: 15px;
    }
    .button-style img {
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.download_button(
        label="📌 Grab Optimized Resume",
        data=pdf_bytes,
        file_name="optimized_resume.pdf",
        mime="application/pdf",
        key="download_button"
)

    # ✅ EMAIL GENERATION BLOCK
    if "generated_email" not in st.session_state:
        st.session_state.generated_email = ""

    if st.button("🤖 Craft Email with AI"):
        with st.spinner("Generating formal email..."):
            st.session_state.generated_email = generate_application_email(name, job_title, matched_skills)
            st.success("✅ Email generated!")

    default_email = f"""Dear Hiring Team,

I am writing to express my interest in the {job_title} position. My skills in {', '.join(matched_skills)} make me a strong candidate.

Thank you for considering my application.

Regards,
{name}
"""
    final_body = st.session_state.generated_email or default_email

    # ✅ Email Form
    st.subheader("📤 Email Your Application")
    with st.form("email_form"):
        recipient = st.text_input("Recipient Email")
        cc = st.text_input("CC (optional)", placeholder="comma-separated")
        subject = st.text_input("Subject", value=f"Application for {job_title} – {name}")
        body = st.text_area("Message", value=final_body, height=200)
        send_btn = st.form_submit_button("Send Email")

    if send_btn:
        if not recipient or "@" not in recipient:
            st.error("❌ Please enter a valid email.")
        else:
            cc_list = [x.strip() for x in cc.split(",")] if cc else []
            success, msg = send_email_with_resume(subject, body, [recipient], pdf_bytes, cc=cc_list)
            st.success(msg) if success else st.error(msg)

    # ✅ Navigation Buttons
    if st.button("🛠️ Edit My Resume"):
        st.session_state.page = "fill_resume"
    if st.button("📄 Review Job Description"):
        st.session_state.page = "upload_jd"
    if st.button("♻️ Start From Scratch"):
        st.session_state.page = "start"
        st.session_state.optimized = ""
        st.session_state.optimized_json = {}
        st.session_state.missing_skills = []

def main():
    page = st.session_state.page
    if page == "start": start_page()
    elif page == "signup": signup_page()
    elif page == "login": login_page()
    elif page == "upload_jd": upload_jd_page()
    elif page == "fill_resume": fill_resume_page()
    elif page == "show_result": result_page()

if __name__ == "__main__":
    main()



# def result_page():
#     st.title("📄 Optimized Resume")

#     # ✅ Optimized Resume JSON Display
#     st.subheader("🧾 Optimized Resume (JSON View)")
#     st.json(st.session_state.optimized_json)

#     # ✅ Styled Missing Skills Section
#     missing_skills = st.session_state.get("missing_skills", [])

#     if "added_skills" not in st.session_state:
#         st.session_state.added_skills = set()

#     st.markdown(
#         """
#         <div style="background-color:#0a58ca; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
#             <h4 style="color: white; margin: 0;">🛑 Missing Skills from JD</h4>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     for skill in missing_skills:
#         col1, col2 = st.columns([0.85, 0.15])
#         col1.markdown(f"<li style='font-size: 16px; color: #000000;'>{skill}</li>", unsafe_allow_html=True)

#         if skill not in st.session_state.optimized_json.get("skills", []):
#             if col2.button("➕", key=f"add_{skill}"):
#                 st.session_state.optimized_json["skills"].append(skill)
#                 st.session_state.added_skills.add(skill)
#         else:
#             if col2.button("➖", key=f"remove_{skill}"):
#                 st.session_state.optimized_json["skills"].remove(skill)
#                 st.session_state.added_skills.discard(skill)

#     else:
#         st.success("✅ No missing skills found. Your resume matches the JD well!")

#     # ✅ Download Missing Skills JSON
#     missing_skills_json_bytes = json.dumps(missing_skills, indent=2).encode("utf-8")
#     st.download_button(
#         label="📥 Download Missing Skills (JSON)",
#         data=missing_skills_json_bytes,
#         file_name="missing_skills.json",
#         mime="application/json"
#     )

#     # ✅ Download Optimized Resume JSON
#     optimized_json_bytes = json.dumps(
#         st.session_state.optimized_json,
#         indent=2
#     ).encode("utf-8")
#     st.download_button(
#         label="📥 Download Optimized Resume (JSON)",
#         data=optimized_json_bytes,
#         file_name="optimized_resume.json",
#         mime="application/json"
#     )

#     # ✅ Download Optimized Resume PDF
#     pdf_bytes = generate_pdf(st.session_state.optimized)
#     name = st.session_state.get("name_for_email") or extract_name(st.session_state.optimized)
#     job_title = extract_job_title(st.session_state.jd)

#     st.download_button(
#         "📄 Download Optimized Resume (PDF)",
#         data=pdf_bytes,
#         file_name="optimized_resume.pdf",
#         mime="application/pdf"
#     )

#     # ✅ Email Form
#     st.subheader("📧 Send Resume via Email")
#     with st.form("email_form"):
#         recipient = st.text_input("Recipient Email")
#         cc = st.text_input("CC (optional)", placeholder="comma-separated")
#         subject = st.text_input("Subject", value=f"Application for {job_title} – {name}")
#         body = st.text_area("Message", value=f"Dear Hiring Team,\n\nI am writing to express my interest in the {job_title} position. Please find my optimized resume attached.\n\nRegards,\n{name}")
#         send_btn = st.form_submit_button("Send Email")

#     if send_btn:
#         if not recipient or "@" not in recipient:
#             st.error("❌ Please enter a valid email.")
#         else:
#             cc_list = [x.strip() for x in cc.split(",")] if cc else []
#             success, msg = send_email_with_resume(subject, body, [recipient], pdf_bytes, cc=cc_list)
#             st.success(msg) if success else st.error(msg)

#     # ✅ Navigation Buttons
#     if st.button("⬅️ Back to Resume Form"):
#         st.session_state.page = "fill_resume"

#     if st.button("🔁 Start Over"):
#         st.session_state.page = "start"
#         st.session_state.optimized = ""
#         st.session_state.optimized_json = {}
#         st.session_state.missing_skills = []

# # ---------------- Main ----------------
# import json
# import streamlit as st

# def display_missing_skills():
#     missing_skills = st.session_state.get("missing_skills_json", [])

#     if not missing_skills:
#         st.success("✅ No missing skills found. Your resume matches the JD well!")
#         return

#     st.markdown(
#         """
#         <div style="background-color:#0a58ca; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
#             <h4 style="color: white; margin: 0;">🛑 Missing Skills from JD</h4>
#         </div>
#         """, unsafe_allow_html=True
#     )

#     for skill in missing_skills:
#         st.markdown(f"<li style='font-size: 16px; color: #e0e0e0;'>{skill}</li>", unsafe_allow_html=True)

#     missing_skills_json_bytes = json.dumps(missing_skills, indent=2).encode("utf-8")
#     st.download_button(
#         label="📥 Download Missing Skills (JSON)",
#         data=missing_skills_json_bytes,
#         file_name="missing_skills.json",
#         mime="application/json"
#     )

# def main():
#     page = st.session_state.page
#     if page == "start": start_page()
#     elif page == "signup": signup_page()
#     elif page == "login": login_page()
#     elif page == "upload_jd": upload_jd_page()
#     elif page == "fill_resume": fill_resume_page()
#     elif page == "show_result": result_page()

# if __name__ == "__main__":
#     main()
