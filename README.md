# 🤖 Resume Optimizer using Gemini AI & FastAPI

A smart resume optimization app that compares your resume with a job description (JD), highlights missing skills, and helps you enhance your resume to improve job match. Built with **Streamlit** (frontend) and **FastAPI** (backend), integrated with **Gemini AI**, **LlamaIndex**, and **regex/spaCy fallbacks** for powerful resume parsing.

---

## 🚀 Features

- 📄 **Upload or Create Resume** (PDF or Manual)
- 🧠 **AI-Powered Resume Parsing** using Gemini + LlamaIndex
- 🎯 **Job Description Comparison**: highlights matched and missing skills
- ➕➖ **Interactive Skill Editor**: Add/remove missing skills
- 📥 **Download Optimized Resume** (JSON & PDF)
- 📧 **Send via Email** directly from the app
- 🔒 **User Auth**: Signup/Login

---

## 🛠️ Tech Stack

| Layer      | Tech/Tool                          |
|------------|------------------------------------|
| Frontend   | Streamlit                          |
| Backend    | FastAPI, Pydantic, SQLite          |
| AI Parsing | Gemini AI via LangChain, LlamaIndex|
| NLP Tools  | Regex, spaCy (fallback parsing)    |
| PDF Tools  | PyMuPDF, FPDF                      |
| Auth       | JWT + FastAPI                      |
| Email      | SMTP with Gmail                    |

---
