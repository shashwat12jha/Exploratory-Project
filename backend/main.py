from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

app = FastAPI(title="Traffic Safety API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder for job status
jobs = {}

from core.pipeline import process_video_pipeline

def process_video_task(job_id: str, video_path: str):
    jobs[job_id] = "Processing"
    try:
        # We assume best.pt is in the parent directory (or wherever it was cloned)
        best_pt_path = "../best.pt" 
        stats = process_video_pipeline(video_path, OUTPUT_DIR, job_id, best_pt_path)
        jobs[job_id] = "Completed"
    except Exception as e:
        jobs[job_id] = f"Error: {str(e)}"

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = "Pending"
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id, "status": "Pending"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    status = jobs.get(job_id, "Not Found")
    return {"job_id": job_id, "status": status}

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    # Read the generated JSON/Excel and return
    # For now, placeholder
    return {"job_id": job_id, "metrics": {}}

@app.get("/api/media/{filename}")
async def get_media(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "File not found"})
