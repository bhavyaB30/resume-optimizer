from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import hashlib
import os
from dotenv import load_dotenv
import json

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List

class ExtractedResume(BaseModel):
    name: str
    contact: str
    email: str
    linkedin: str
    github: str
    summary: str
    education: str
    skills: List[str]
    projects: str
    certifications: str
    work_experience: str
    achievements: str

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Database setup ----------
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

# ---------- FastAPI setup ----------
app = FastAPI(
    title="Resume Optimizer API",
    version="1.0.0",
    description="Signup/Login and Resume Optimization API"
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

# ---------- Pydantic Models ----------
class User(BaseModel):
    name: str
    email: str
    password: str

class LoginData(BaseModel):
    email: str
    password: str

class ResumeOptimizationRequest(BaseModel):
    jd: str
    resume: str

# ---------- API Endpoints ----------
@app.post("/signup", tags=["Auth"])
def signup(user: User):
    db = SessionLocal()
    existing = db.query(UserTable).filter(UserTable.email == user.email).first()
    if existing:
        db.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = UserTable(
        email=user.email,
        name=user.name,
        password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.close()
    return {"message": "Signup successful"}

@app.post("/login", tags=["Auth"])
def login(data: LoginData):
    db = SessionLocal()
    user = db.query(UserTable).filter(UserTable.email == data.email).first()
    db.close()
    if not user or user.password != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "name": user.name}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json

class ResumeText(BaseModel):
    resume: str

@app.post("/extract_resume_data", response_model=ExtractedResume, tags=["Resume"])
def extract_resume_data_api(data: ResumeText):
    resume_text = data.resume.strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text is empty.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=""
    )

    prompt = PromptTemplate.from_template("""
You are a resume parser. Given a raw resume text, extract the following fields in valid JSON format only
                                          you should be able to extract each and evry thing inside of it coorectly and fill it 

Instructions:
- Use only the content from the resume. Do not guess or fabricate.
- If a section is missing, return an empty string or empty list.
- Skills should be extracted as a list of strings.skills should be extracted only from the skills section or section which contains skills but named different 
                                          
- if any field is going empty retryonce more and find that specific thing content in the resume .
- Parse bullet points, colons, and section titles like "SKILLS", "PROJECTS", "EDUCATION", etc.
- there can be some other names for following section in resume it can vary resuume wise so do it efficeintly
- extract in json format so that it is easy to fill deatils tehre 

Return JSON in this format:
{{
  "name": "Candidate's full name",
  "contact": "Phone number",
  "email": "Email address",
  "linkedin": "LinkedIn URL or empty",
  "github": "GitHub URL or empty",
  "summary": "Brief summary or About Me section",
  "education": "Education section content",
  "skills": ["Skill1", "Skill2", ...],
  "projects": "Projects section content",
  "certifications": "Certifications listed",
  "work_experience": "Work Experience section",
  "achievements": "Achievements, awards, extracurriculars"
}}

Resume Text:
{resume}
""")

    chain = prompt | llm
    try:
        result = chain.invoke({"resume": resume_text})
        parsed = json.loads(result.content.strip())
        return ExtractedResume(**parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing error: {e}")

@app.post("/optimize_resume", tags=["Resume"])
def optimize_resume(data: ResumeOptimizationRequest):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=""
    )

    prompt = PromptTemplate(
        input_variables=["jd", "resume"],
        template="""
You are an expert resume optimizer.

Improve the resume below by:
1. Adding **missing skills, technologies, or keywords** from the job description (JD) into relevant sections like Skills, Projects, or Work Experience — but only if they are already **reflected or implied in the project descriptions**. Do not add any skills that are not already mentioned or evident in the projects.
2. Updating the **Summary / About Me** section to reflect the enhanced skill set and better match the job description — while sounding natural and tailored.
3. Keeping everything **ATS-optimized** and written in **standard resume tone** (first-person implied, no "I", no personal pronouns).
4. Preserving good existing content. Do NOT remove strong parts already present.
5. Keeping formatting clean, plain text only (no markdown, no comments).

ONLY return the final improved resume **in valid text  format**, with the following structure:
```j
{{
  "name": "...",
  "contact": "...",
  "email": "...",
  "linkedin": "...",
  "github": "...",
  "summary": "...",
  "education": "...",
  "skills": ["...", "..."],
  "projects": "...",
  "certifications": "...",
  "work_experience": "...",
  "achievements": "..."
}}

ONLY return the final improved resume. No explanation or commentary.

Job Description:
{jd}

Original Resume:
{resume}

Final Resume:
"""
    )

    chain = prompt | llm
    output = chain.invoke({"jd": data.jd, "resume": data.resume})
    
    return {"optimized_resume": output.content}
