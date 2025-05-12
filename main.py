import os
import cv2
import torch
import numpy as np
import pandas as pd
import shutil
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from typing import List

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from ultralytics import YOLO

from database import SessionLocal, Pothole

import zipfile

app = FastAPI()

# Mount static file directories
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Ensure directories exist
SNAPSHOT_DIR = "snapshots"
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
ZIP_DIR = "zips"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

model = YOLO("best.pt")
PIXEL_TO_M2 = 0.00000007  # 1 pixel² = 0.00000007 m²
PIXEL_TO_MM = 0.265       # 1 pixel = 0.265 mm

def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def get_gps_coordinates(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            gps_info = {TAGS.get(tag): value for tag, value in exif_data.items() if TAGS.get(tag) == "GPSInfo"}
            if gps_info:
                gps_data = gps_info["GPSInfo"]
                gps_data = {GPSTAGS.get(t): v for t, v in gps_data.items()}
                lat = gps_data.get("GPSLatitude")
                lat_ref = gps_data.get("GPSLatitudeRef")
                lon = gps_data.get("GPSLongitude")
                lon_ref = gps_data.get("GPSLongitudeRef")
                if lat and lon and lat_ref and lon_ref:
                    latitude = convert_to_degrees(lat)
                    longitude = convert_to_degrees(lon)
                    if lat_ref != "N":
                        latitude = -latitude
                    if lon_ref != "E":
                        longitude = -longitude
                    return latitude, longitude
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
    return None, None

def draw_largest_contour_and_calculate_area(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)
        return cv2.contourArea(largest_contour)
    return 0

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Pothole Detection API is running!"}

@app.post("/save_snapshot/")
async def save_snapshot(image: UploadFile = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}_{image.filename}"
    save_path = os.path.join(SNAPSHOT_DIR, filename)
    with open(save_path, "wb") as f:
        content = await image.read()
        f.write(content)
    return {"message": "Snapshot saved", "file_path": save_path}

@app.get("/snapshots/")
def list_snapshots() -> List[str]:
    try:
        files = os.listdir(SNAPSHOT_DIR)
        return sorted(files)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Snapshot directory not found")

@app.get("/snapshots/{filename}")
def get_snapshot(filename: str):
    file_path = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(file_path)

@app.get("/snapshots/zip")
def download_all_snapshots():
    if not os.path.exists(SNAPSHOT_DIR):
        raise HTTPException(status_code=404, detail="Snapshot directory does not exist")
    zip_filename = f"snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(ZIP_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(SNAPSHOT_DIR):
            file_path = os.path.join(SNAPSHOT_DIR, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=file)
    return FileResponse(path=zip_path, filename=zip_filename, media_type="application/zip")

@app.post("/detect_pothole/")
async def detect_pothole(image: UploadFile = File(...), db: Session = Depends(get_db)):
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    detected_image_path = os.path.join(UPLOAD_DIR, f"detected_{image.filename}")
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to read image"}

    gps_lat, gps_lon = get_gps_coordinates(image_path)
    results = model(img)

    potholes_detected = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                contour_area_pixels = draw_largest_contour_and_calculate_area(img, x1, y1, x2, y2)
                area_m2 = contour_area_pixels * PIXEL_TO_M2

                diagonal_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                diagonal_mm = diagonal_px * PIXEL_TO_MM

                if diagonal_mm <= 250:
                    depth_mm = 30
                    severity_class = "Minor"
                    color = (0, 255, 0)
                elif 250 < diagonal_mm <= 500:
                    depth_mm = 75
                    severity_class = "Moderate"
                    color = (0, 255, 255)
                else:
                    depth_mm = 100
                    severity_class = "Severe"
                    color = (0, 0, 255)

                volume_m3 = area_m2 * (depth_mm / 1000)
                label = f"{severity_class} ({depth_mm} mm)"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                new_pothole = Pothole(
                    latitude=gps_lat or 0.0,
                    longitude=gps_lon or 0.0,
                    severity=severity_class,
                    area_m2=area_m2,
                    volume_m3=volume_m3,
                    image_path=detected_image_path
                )
                db.add(new_pothole)

                potholes_detected.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": confidence,
                    "severity": severity_class,
                    "area_m2": area_m2,
                    "depth_mm": depth_mm,
                    "volume_m3": volume_m3,
                    "latitude": gps_lat,
                    "longitude": gps_lon
                })

    db.commit()
    cv2.imwrite(detected_image_path, img)

    report_filename = f"report_{image.filename.split('.')[0]}.xlsx"
    report_path = os.path.join(RESULTS_DIR, report_filename)
    pd.DataFrame(potholes_detected).to_excel(report_path, index=False)

    return {
        "potholes": potholes_detected,
        "image_path": detected_image_path,
        "excel_report": report_filename
    }

@app.get("/download_excel/{filename}")
def download_excel(filename: str):
    excel_file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(excel_file_path):
        return FileResponse(
            excel_file_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return {"error": "Excel file not found"}

@app.get("/download_excel_from_db/")
def download_excel_from_db(db: Session = Depends(get_db)):
    potholes = db.query(Pothole).all()
    data = [
        {
            "Latitude": p.latitude,
            "Longitude": p.longitude,
            "Severity": p.severity,
            "Area (m²)": p.area_m2,
            "Volume (m³)": p.volume_m3,
            "Image Path": p.image_path,
        }
        for p in potholes
    ]
    df = pd.DataFrame(data)
    excel_path = os.path.join(RESULTS_DIR, "pothole_results.xlsx")
    df.to_excel(excel_path, index=False)
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="pothole_results.xlsx"
    )

@app.get("/export_excel/")
def export_excel(db: Session = Depends(get_db)):
    potholes = db.query(Pothole).all()
    if not potholes:
        return {"message": "No data to export"}
    data = [
        {
            "ID": p.id,
            "Latitude": p.latitude,
            "Longitude": p.longitude,
            "Severity": p.severity,
            "Area (m²)": p.area_m2,
            "Volume (m³)": p.volume_m3,
            "Timestamp": p.timestamp,
            "Image Path": p.image_path,
        }
        for p in potholes
    ]
    df = pd.DataFrame(data)
    filename = "pothole_report.xlsx"
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_excel(filepath, index=False)
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aipotholedetection.fwh.is/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
