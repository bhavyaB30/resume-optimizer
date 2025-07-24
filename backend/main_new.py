from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.query_engine import SubQuestionQueryEngine
import spacy

nlp = spacy.load("en_core_web_sm")

from typing import List

# ---------- Database Setup ----------
DATABASE_URL = "sqlite:///./users.db"
Base = declarative_base()

class UserTable(Base):
    __tablename__ = "users"
    email = Column(String, primary_key=True, index=True)
    name = Column(String)
    password = Column(String)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------- FastAPI Setup ----------
app = FastAPI(
    title="Resume Optimizer API",
    version="1.0.0",
    description="Only signup/login are handled here. Resume parsing/optimization handled on frontend."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\n(?=[a-z0-9])", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def split_resume_sections(text: str) -> dict:
    section_titles = [
        "summary", "objective", "about me", "education", "skills", "coursework", "technical skills",
        "projects", "certifications", "experience", "work experience", "employment",
        "achievements", "co-curricular", "awards", "honors", "internships"
    ]
    pattern = r"(?i)(?<=\n)({})(?:\s*[:\-]?\s*)".format("|".join(map(re.escape, section_titles)))
    parts = re.split(pattern, text)

    sections = {}
    current = "headerless"
    for i in range(len(parts)):
        if parts[i].strip().lower() in section_titles:
            current = parts[i].strip().lower()
            sections[current] = ""
        else:
            sections[current] = sections.get(current, "") + "\n" + parts[i].strip()
    return sections

def fallback_regex_parser(text: str):
    sections = split_resume_sections(text)
    skills_section = sections.get("skills", "") + "\n" + sections.get("coursework", "") + "\n" + sections.get("technical skills", "")
    skills = [s.strip("•-– ") for s in re.split(r"[\n,;•]", skills_section) if 1 < len(s.strip()) < 50]

    name_match = re.findall(r"(?m)^\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s*$", text[:300])
    email_match = re.findall(r"[\w\.\-]+@[\w\.-]+", text)
    phone_match = re.findall(r"(?<!\d)(?:\+91[\-\s]?)?(\d{10})(?!\d)", text)
    linkedin = re.findall(r'https?://(?:www\.)?linkedin\.com/in/[^\s"\']+', text)
    github = re.findall(r'https?://(?:www\.)?github\.com/[^\s"\']+', text)

    return ExtractedResume(
        name=name_match[0] if name_match else "",
        contact=phone_match[0] if phone_match else "",
        email=email_match[0] if email_match else "",
        linkedin=linkedin[0] if linkedin else "",
        github=github[0] if github else "",
        summary=sections.get("about me", "") or sections.get("summary", ""),
        education=sections.get("education", ""),
        skills=skills,
        projects=sections.get("projects", ""),
        certifications=sections.get("certifications", ""),
        work_experience=sections.get("experience", "") + "\n" + sections.get("work experience", "") + "\n" + sections.get("employment", "") + "\n" + sections.get("internships", ""),
        achievements=sections.get("achievements", "") + "\n" + sections.get("co-curricular", "") + "\n" + sections.get("awards", "") + "\n" + sections.get("honors", "")
    )

def open_resume_parse(text: str) -> dict:
    """
    Resume parser using improved regex logic and flexible multi-format section handling.
    """
    def find_section(text, keys):
        pattern = "|".join(re.escape(k) for k in keys)
        all_sections = list(SECTION_TITLES.values())
        flat_all_titles = [item for sub in all_sections for item in sub]
        section_regex = re.compile(
            rf"(?:^|\n)\s*({pattern})\s*[:\-]?\s*(.*?)(?=\n\s*({'|'.join(map(re.escape, flat_all_titles))})\s*[:\-]?|\n\S|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = section_regex.findall(text)
        return matches[0][1].strip() if matches else ""

    def extract_name(text):
        patterns = [
            r"(?:(?<=name[:\-])|(?<=name is)|(?<=I am)|(?<=This is))\s*([A-Z][a-z]+(?:[\s\-][A-Z][a-z]+){1,3})",
            r"^([A-Z][a-z]+(?:[\s\-][A-Z][a-z]+){1,3})$"
        ]
        for pat in patterns:
            match = re.search(pat, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:5]:
            if re.match(r"^[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+){1,3}$", line) and not re.search(r"resume|email|phone|contact", line, re.I):
                return line
        return ""

    def extract_phone(text):
        patterns = [
            r"(\+?\d{1,3}[-\s]?\d{3,5}[-\s]?\d{6,10})",
            r"(?<!\d)(\d{10})(?!\d)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""

    def extract_email(text):
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else ""

    def extract_linkedin(text):
        match = re.search(r"https?://(www\.)?linkedin\.com/in/[^\s\"\'<>]+", text, re.I)
        return match.group(0) if match else ""

    def extract_github(text):
        match = re.search(r"https?://(www\.)?github\.com/[^\s\"\'<>]+", text, re.I)
        return match.group(0) if match else ""

    def extract_skills(section_text):
        skills = []
        skills += re.findall(r"[\-\u2022\*]\s*([A-Za-z0-9\+\-#\/,.() ]+)", section_text)
        if not skills:
            skills += re.split(r"[\n,;/]", section_text)
        return [s.strip() for s in skills if 2 <= len(s.strip()) <= 40]

    SECTION_TITLES = {
        "summary": ["summary", "about me", "objective", "professional summary", "profile"],
        "education": ["education", "academic background", "qualifications", "academics"],
        "skills": ["skills", "technical skills", "core skills", "skill set"],
        "projects": ["projects", "key projects", "project experience"],
        "certifications": ["certifications", "certified", "licenses"],
        "work_experience": ["experience", "work experience", "employment", "internships", "professional experience"],
        "achievements": ["achievements", "honors", "awards", "responsibilities", "activities", "co-curricular"],
    }

    return {
        "name": extract_name(text),
        "contact": extract_phone(text),
        "email": extract_email(text),
        "linkedin": extract_linkedin(text),
        "github": extract_github(text),
        "summary": find_section(text, SECTION_TITLES["summary"]),
        "education": find_section(text, SECTION_TITLES["education"]),
        "skills": extract_skills(find_section(text, SECTION_TITLES["skills"])),
        "projects": find_section(text, SECTION_TITLES["projects"]),
        "certifications": find_section(text, SECTION_TITLES["certifications"]),
        "work_experience": find_section(text, SECTION_TITLES["work_experience"]),
        "achievements": find_section(text, SECTION_TITLES["achievements"]),
    }

def spacy_fallback_parser(text):
    doc = nlp(text)
    candidate_lines = [line for line in text.splitlines()[:5] if line]
    name = ""
    for line in candidate_lines:
        for ent in nlp(line).ents:
            if ent.label_ == "PERSON":
                name = ent.text
                break
        if name:
            break
    email = next((token.text for token in doc if token.like_email), "")
    phone = next((m.group(1).strip() for m in re.finditer(r"(\+?\d[\d\s\-]{8,}\d)", text)), "")
    linkedin = next((m.group(0) for m in re.finditer(r"https?://(www\.)?linkedin\.com/\S+", text)), "")
    github = next((m.group(0) for m in re.finditer(r"https?://(www\.)?github\.com/\S+", text)), "")
    sentences = list(doc.sents)
    summary, education, skill_lines = "", "", []
    for sent in sentences:
        lsent = sent.text.lower()
        if "skill" in lsent:
            skill_lines.append(sent.text)
        elif "education" in lsent or "degree" in lsent:
            education += sent.text.strip() + " "
        elif "project" not in lsent and "internship" not in lsent and "experience" not in lsent:
            if len(summary) < 200:
                summary += sent.text.strip() + " "
    skills = [s.strip() for l in skill_lines for s in re.split(r"[,;\n]", l) if len(s.strip()) > 2]
    return {
        "name": name,
        "contact": phone,
        "email": email,
        "linkedin": linkedin,
        "github": github,
        "summary": summary.strip(),
        "education": education.strip(),
        "skills": skills
    }

def llamaindex_fallback(text: str):
    document = Document(text=text)
    index = VectorStoreIndex.from_documents([document])
    query_engine = index.as_query_engine()
    fields = {
        "name": "What is the full name of the candidate?",
        "summary": "Provide the About Me or Summary section.",
        "skills": "List all the technical skills mentioned."
    }
    output = {}
    for key, question in fields.items():
        response = query_engine.query(question)
        output[key] = response.response.strip()
    return output

# ---------- Pydantic Models ----------
class User(BaseModel):
    name: str
    email: str
    password: str

class LoginData(BaseModel):
    email: str
    password: str

class ResumeData(BaseModel):
    resume: str

class ResumeOptimizationRequest(BaseModel):
    jd: str
    resume: str

class ExtractedResume(BaseModel):
    name: str
    contact: str
    email: str
    linkedin: str
    github: str
    summary: str
    education: str
    skills: list[str]
    projects: str
    certifications: str
    work_experience: str
    achievements: str

# ---------- Auth Endpoints ----------
@app.post("/signup", tags=["Auth"])
def signup(user: User):
    db = SessionLocal()
    try:
        existing = db.query(UserTable).filter(UserTable.email == user.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        new_user = UserTable(
            email=user.email,
            name=user.name,
            password=hash_password(user.password)
        )
        db.add(new_user)
        db.commit()
        return {"message": "Signup successful"}
    finally:
        db.close()

@app.post("/login", tags=["Auth"])
def login(data: LoginData):
    db = SessionLocal()
    try:
        user = db.query(UserTable).filter(UserTable.email == data.email).first()
        if not user or user.password != hash_password(data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"message": "Login successful", "name": user.name}
    finally:
        db.close()

# ---------- Resume Parser & Optimizer Endpoints ----------
@app.post("/extract_resume_data", tags=["Resume"])
def extract_resume_data(data: ResumeData):
    resume_text = clean_text(data.resume.strip())
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text is empty.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="AIzaSyBSGpWM1o1ckDVqk5vSgu9EtntNeADEj8c"
    )

    prompt = PromptTemplate.from_template("""
ou are a resume parser. Extract the following fields from the raw text into complete JSON ONLY (no explanation). Be careful: sections may be merged, out of order, or implicit. Infer sections from keywords:

Return JSON:
{
  "name": "",
  "contact": "",
  "email": "",
  "linkedin": "",
  "github": "",
  "summary": "",
  "education": "",
  "skills": [""],
  "projects": "",
  "certifications": "",
  "work_experience": "",
  "achievements": ""
}

Resume:
{resume}
""")

    try:
        chain = prompt | llm
        result = chain.invoke({"resume": resume_text})
        parsed = json.loads(result.content.strip())

        fallback = fallback_regex_parser(resume_text)
        llama_fallback = llamaindex_fallback(resume_text)
        open_fallback = open_resume_parse(resume_text)

        for key in fallback.dict():
            if parsed.get(key) in [None, "", [], {}]:
                parsed[key] = fallback.dict()[key]
        for key in llama_fallback:
            if parsed.get(key) in [None, "", [], {}]:
                parsed[key] = llama_fallback[key]
        for key in open_fallback:
            if parsed.get(key) in [None, "", [], {}]:
                parsed[key] = open_fallback[key]
        # spaCy fallback for any missing or short fields
        spacy_parsed = spacy_fallback_parser(resume_text)
        for key in ["name", "contact", "email", "linkedin", "github", "summary", "education", "skills"]:
            if parsed.get(key) in [None, "", [], {}] and spacy_parsed.get(key):
                parsed[key] = spacy_parsed[key]

        parsed = {k: (v.strip() if isinstance(v, str) else v) for k, v in parsed.items()}
        return ExtractedResume(**parsed)
    except Exception as e:
        try:
            return fallback_regex_parser(resume_text)
        except Exception as fe:
            raise HTTPException(status_code=500, detail=f"Parsing failed: {fe}")

import json
# ✅ 1. Known skill list (extendable)
KNOWN_SKILLS = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Go", "Ruby", "Swift", "Kotlin", "R", "Perl", "Scala", "Rust", "MATLAB",

    # Web Development
    "HTML", "CSS", "React", "Angular", "Vue.js", "Django", "Flask", "Next.js", "Node.js", "Express.js",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis", "Oracle", "Microsoft SQL Server", "NoSQL",

    # Data Science / ML
    "NumPy", "pandas", "Scikit-learn", "Matplotlib", "Seaborn", "Plotly", "Power BI", "Tableau", "Excel", "Google Sheets",
    "TensorFlow", "PyTorch", "Keras", "OpenCV", "LangChain", "Hugging Face", "RAG", "Agentic AI", "LLMs", "spaCy", "NLTK",

    # Cloud & DevOps
    "Docker", "Kubernetes", "Git", "GitHub", "CI/CD", "AWS", "Azure", "GCP", "Linux", "Heroku",

    # Backend / APIs
    "REST API", "RESTful APIs", "FastAPI", "Spring Boot", "Postman", "GraphQL",

    # Tools
    "Salesforce", "Jira", "Notion", "Figma", "Canva", "Trello", "WordPress", "Zapier", "Slack",

    # Testing
    "Selenium", "Playwright", "Pytest", "Postman", "JUnit",

    # Soft Skills
    "Communication", "Leadership", "Project management", "Problem-solving", "Critical thinking", "Teamwork", "Time management",

    # General Terms
    "web scraping", "data cleaning", "data analysis", "data visualization", "prompt engineering", "cloud computing"
]

# ✅ 2. Skill extraction function
def extract_skills(text: str):
    import re
    text_lower = text.lower()
    matched = {skill for skill in KNOWN_SKILLS if skill.lower() in text_lower}

    # Optional fallback regex-based extraction
    fallback = set([
        s.strip() for s in re.findall(r'\b[A-Za-z0-9\+\-#/.() ]{2,}\b', text)
        if 2 < len(s.strip()) <= 40 and not s.strip().isdigit()
    ])

    return list(matched.union(fallback))

# ✅ 3. Missing skills comparison fallback
def extract_missing_skills_fallback(jd: str, resume: str):
    jd_skills = set(skill.lower() for skill in extract_skills(jd))
    resume_skills = set(skill.lower() for skill in extract_skills(resume))
    
    missing = sorted([skill for skill in jd_skills if skill not in resume_skills])
    return missing

@app.post("/optimize_resume")
def optimize_resume(data: ResumeOptimizationRequest):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="AIzaSyBSGpWM1o1ckDVqk5vSgu9EtntNeADEj8c"
    )

    prompt = PromptTemplate(
        input_variables=["jd", "resume"],
        template="""
You are an expert resume optimizer.

1. Add relevant missing skills from the JD, only if reflected in resume.
2. Do not add any new skill extract it from resume according to jd 
2. Update the summary/About Me section.
3. Use clean ATS-optimized text format (no markdown or labels).
4. Do not remove good content
5.DO NOT hallucinate or invent any content.
ONLY return the final resume JSON in this format:

{{
  "name": "...",
  "contact": "...",
  "email": "...",
  "linkedin": "...",
  "github": "...",
  "summary": "...",
  "education": "...",
  "skills": ["..."],
  "projects": "...",
  "certifications": "...",
  "work_experience": "...",
  "achievements": "..."
}}

Job Description:
{jd}

Resume:
{resume}
"""
    )

    # Step 1: Run Resume Optimization Prompt
    chain = prompt | llm
    try:
        output = chain.invoke({"jd": data.jd, "resume": data.resume})
        if not output or not output.content:
            raise ValueError("LLM returned no content")

        raw = output.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        optimized_resume = json.loads(raw)
    except Exception as e:
        return {
            "error": f"❌ Failed to parse optimized resume: {e}",
            "raw_output": output.content if output else "None"
        }

    # Step 2: Missing Skills Detection (LLM + Fallback)
    missing_prompt = PromptTemplate(
        input_variables=["jd", "resume"],
        template="""
You are a strict skill-matching assistant.

TASK:
1. From the job description (JD), extract **only actual technical or soft skills**.
   - ✅ Examples: Python, SQL, Excel, Leadership, Communication, Scikit-learn
   - 🚫 Ignore: job titles, job type (full-time), locations, years of experience, company name, etc.

2. From the resume, extract all explicitly present skills — match exact wording.

3. Compare both skill sets.

4. Return a list of skills that are in the JD but NOT present in the resume.
⚠️ Do not hallucinate, infer, or assume related skills unless listed as a synonym below.

Use the following synonyms (treat these as identical when comparing):

- "rest api", "restful api", "restful apis"
- "scikit-learn", "sklearn"
- "communication", "communication skills"
- "google sheets", "google spreadsheet"
- "cv", "computer vision"
- "keras", "tensorflow keras"
IMPORTANT RULES:
- Do NOT infer or assume synonyms — match skills word-for-word.
- Ignore general terms like "Responsibilities", "Location", "Job Description", "2 years", etc.
- Only return skill names (not explanations).

OUTPUT FORMAT:
Return only the missing skills in clean JSON array format:
["Skill 1", "Skill 2", "Skill 3"]

If all skills are present, return:
[]
    
JOB DESCRIPTION:
{jd}

RESUME:
{resume}
If no skills are missing, return an empty list: []
"""
    )

    fallback_missing_skills = []
    try:
        skill_chain = missing_prompt | llm
        missing_response = skill_chain.invoke({"jd": data.jd, "resume": data.resume})
        if not missing_response or not missing_response.content:
            raise ValueError("Missing skills LLM returned no content")

        missing_raw = missing_response.content.strip()
        if missing_raw.startswith("```json"):
            missing_raw = missing_raw[7:]
        if missing_raw.endswith("```"):
            missing_raw = missing_raw[:-3]
        missing_raw = missing_raw.strip()

        missing_skills = json.loads(missing_raw)
        if not isinstance(missing_skills, list):
            raise ValueError("Invalid missing skill format")
    except Exception as e:
        print(f"⚠️ Falling back to regex-based skill comparison: {e}")
        missing_skills = extract_missing_skills_fallback(data.jd, data.resume)

    return {
        "optimized_resume": optimized_resume,
        "missing_skills": missing_skills
    }




# from fastapi import APIRouter
# from pydantic import BaseModel
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# import json

# class SkillExtractionRequest(BaseModel):
#     jd: str
#     resume: str

# @app.post("/get_missing_skills")
# def get_missing_skills(data: SkillExtractionRequest):
#     # ✅ Initialize the Gemini LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         google_api_key=""
#     )

#     # ✅ Prompt template for extracting missing skills
#     missing_skills_prompt = PromptTemplate(
#         input_variables=["jd", "resume"],
#         template="""
# You are a strict skill-matching assistant.

# Your task is to:
# 1. Read the job description (JD) and extract all explicitly mentioned skills.
# 2. Read the resume and extract all skills **literally and exactly** mentioned anywhere in the resume text.
# 3. Compare both lists.
# 4. Return a list of skills from the JD that are **not present word-for-word** in the resume.
# 5. Do not infer, assume, or match related terms — only exact word or phrase matches count.

# Return ONLY the missing skills from the JD as a clean JSON list, like this:
# ["Skill1", "Skill2", "Skill3"]

# If no skills are missing, return an empty list: []
# """
#     )

#     try:
#         # ✅ Run the LLM chain
#         missing_chain = missing_skills_prompt | llm
#         missing_output = missing_chain.invoke({
#             "jd": data.jd,
#             "resume": data.resume
#         })

        # ✅ Attempt to parse JSON safely
    #     try:
    #         skills = json.loads(missing_output.content.strip())
    #         if not isinstance(skills, list):
    #             raise ValueError("Parsed output is not a list.")
    #     except Exception as parse_err:
    #         print("❌ JSON parsing error:", parse_err)
    #         skills = []

    #     return {"missing_skills": skills}

    # except Exception as e:
    #     print("❌ LLM invocation error:", e)
    #     return {"error": str(e), "missing_skills": []}