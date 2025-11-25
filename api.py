from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
from datetime import datetime
import sqlite3
import os

app = FastAPI()

DATABASE_DIR_FACE_RECOGNATION   = "dataBase/faceRecognation"
DATABASE_DIR_VOICE_RECOGNATION  = "dataBase/voiceRecognation"
DATABASE_DIR_FUTION_RECOGNATION = "dataBase/futionRecognation"

@app.get("/reports/db/list/face-recognation", summary="List available Face recognation  data base reports")
def list_reports_face_recognation_db():
    if not os.path.exists(DATABASE_DIR_FACE_RECOGNATION):
        return []
    files = [f for f in os.listdir(DATABASE_DIR_FACE_RECOGNATION) if f.endswith(".db")]
    return sorted(files)

@app.get("/reports/db/list/voice-recognation", summary="List available voice recognation  data base reports")
def list_reports_voice_recognation_db():
    if not os.path.exists(DATABASE_DIR_VOICE_RECOGNATION):
        return []
    files = [f for f in os.listdir(DATABASE_DIR_VOICE_RECOGNATION) if f.endswith(".db")]
    return sorted(files)

@app.get("/reports/db/list/fution-recognation", summary="List available fution recognation  data base reports")
def list_reports_fution_recognation_db():
    if not os.path.exists(DATABASE_DIR_FUTION_RECOGNATION):
        return []
    files = [f for f in os.listdir(DATABASE_DIR_FUTION_RECOGNATION) if f.endswith(".db")]
    return sorted(files)


@app.get("/emotions/db/face-recognation/{filename}", summary="Get all emotions")
def get_filename_face_recognation_emotions(filename:str):
    path = os.path.join(DATABASE_DIR_FACE_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Daba base {filename} not found")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, emotion, prob, probs FROM emotions")
    data = cursor.fetchall()
    conn.close()

    return [{"timestamp": t, "emotion": e, "prob": p, "probs": ps} for t, e, p, ps in data]


@app.get("/emotions/db/voice-recognation/{filename}", summary="Get all emotions")
def get_filename_voice_recognation_emotions(filename:str):
    path = os.path.join(DATABASE_DIR_VOICE_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Daba base {filename} not found")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, emotion, prob, probs FROM emotions")
    data = cursor.fetchall()
    conn.close()
    return [{"timestamp": t, "emotion": e, "prob": p, "probs": ps} for t, e, p, ps in data]

@app.get("/emotions/db/fution-recognation/{filename}", summary="Get all emotions")
def get_filename_fution_recognation_emotions(filename:str):
    path = os.path.join(DATABASE_DIR_FUTION_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Daba base {filename} not found")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, emotion, prob, probs FROM emotions")
    data = cursor.fetchall()
    conn.close()
    return [{"timestamp": t, "emotion": e, "prob": p, "probs": ps} for t, e, p, ps in data]

# @app.get("/emotions/filter", summary="Filter emotions by type or date")
# def filter_emotions(
#     emotion: Optional[str] = Query(None),
#     start: Optional[str] = Query(None),
#     end: Optional[str] = Query(None)
# ):
#     db_path = get_latest_db()
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     query = "SELECT timestamp, emotion, prob FROM emotions WHERE 1=1"
#     params = []

#     if emotion:
#         query += " AND emotion = ?"
#         params.append(emotion)

#     if start:
#         query += " AND timestamp >= ?"
#         params.append(start)

#     if end:
#         query += " AND timestamp <= ?"
#         params.append(end)

#     cursor.execute(query, tuple(params))
#     data = cursor.fetchall()
#     conn.close()

#     return [{"timestamp": t, "emotion": e, "prob": p} for t, e, p in data]


PDF_DIR_FACE_RECOGNATION = "pdf/faceRecognation"
PDF_DIR_VOICE_RECOGNATION = "pdf/voiceRecognation"
PDF_DIR_FUTION_RECOGNATION = "pdf/futionRecognation"

@app.get("/reports/pdf/faceRecognation", summary="List available PDF reports")
def list_reports_pdf_face_recognation():
    if not os.path.exists(PDF_DIR_FACE_RECOGNATION):
        return []
    files = [f for f in os.listdir(PDF_DIR_FACE_RECOGNATION) if f.endswith(".pdf")]
    return sorted(files)

@app.get("/reports/pdf/voiceRecognation", summary="List available PDF reports")
def list_reports_pdf_voice_recognation():
    if not os.path.exists(PDF_DIR_VOICE_RECOGNATION):
        return []
    files = [f for f in os.listdir(PDF_DIR_VOICE_RECOGNATION) if f.endswith(".pdf")]
    return sorted(files)

@app.get("/reports/pdf/futionRecognatio", summary="List available PDF reports")
def list_reports_pdf_fution_recognation():
    if not os.path.exists(PDF_DIR_FUTION_RECOGNATION):
        return []
    files = [f for f in os.listdir(PDF_DIR_FUTION_RECOGNATION) if f.endswith(".pdf")]
    return sorted(files)

@app.get("/reports/pdf/face-recognation/{filename}", response_class=FileResponse, summary="Download a specific PDF report")
def download_report_face_recognation(filename: str):
    path = os.path.join(PDF_DIR_FACE_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type='application/pdf', filename=filename)

@app.get("/reports/pdf/face-recognation/{filename}", response_class=FileResponse, summary="Download a specific PDF report")
def download_report_voice_recognation(filename: str):
    path = os.path.join(PDF_DIR_VOICE_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type='application/pdf', filename=filename)


@app.get("/reports/pdf/fution-recognation/{filename}", response_class=FileResponse, summary="Download a specific PDF report")
def download_report_face_recognation(filename: str):
    path = os.path.join(PDF_DIR_FUTION_RECOGNATION, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type='application/pdf', filename=filename)