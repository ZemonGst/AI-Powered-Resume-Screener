from fastapi import FastAPI, File, UploadFile, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pdfplumber
import os
import shutil

# Load the trained model and vectorizer
model = joblib.load("trained model and tfidf_vector/resume_screener_model.pkl")  # Ensure correct path
vectorizer = joblib.load("trained model and tfidf_vector/tfidf_vectorizer.pkl")  # Ensure correct path

# Initialize FastAPI app
app = FastAPI()

# Mount static files (for CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Root endpoint to serve HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text.strip()

# Endpoint to handle PDF upload and make predictions
@app.post("/upload/")
async def upload_resume(request: Request, resume: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_location = f"temp_{resume.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(file_location)

    # Remove temporary file
    os.remove(file_location)

    # Transform text using the TF-IDF vectorizer
    resume_tfidf = vectorizer.transform([extracted_text])

    # Predict category
    prediction = model.predict(resume_tfidf)[0]

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "category": prediction},
    )
