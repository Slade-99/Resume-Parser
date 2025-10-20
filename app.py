# app.py
import os
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from resume_parser import UnifiedResumeParser

# Load .env variables (MISTRAL_API_KEY, etc.)
load_dotenv()

app = FastAPI(
    title="Resume Parser API",
    description="Upload a PDF or image(s) and get structured JSON from the resume.",
    version="1.0.0",
)

# Allow frontend access (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = UnifiedResumeParser()  # initializes with your .env API key

@app.post("/parse-resume")
async def parse_resume(
    mode: str = Form(..., description="Either 'pdf' or 'image'"),
    files: List[UploadFile] = None,
):
    """
    mode: 'pdf' or 'image'
    files: uploaded files (single PDF or multiple images)
    """

    if not files or len(files) == 0:
        return JSONResponse({"error": "No file(s) uploaded."}, status_code=400)

    temp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for f in files:
            file_path = os.path.join(temp_dir, f.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await f.read())
            file_paths.append(file_path)

        if mode.lower() == "pdf":
            result = parser.parse_resume("pdf", file_paths[0])
        elif mode.lower() == "image":
            result = parser.parse_resume("image", file_paths)
        else:
            return JSONResponse({"error": "mode must be 'pdf' or 'image'."}, status_code=400)

        return JSONResponse(result)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/")
def root():
    return {"message": "Resume Parser API is running! Use POST /parse-resume"}
