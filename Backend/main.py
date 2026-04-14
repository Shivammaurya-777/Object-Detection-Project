from fastapi import FastAPI, File, UploadFile # pyright: ignore[reportMissingImports]
from fastapi.responses import StreamingResponse, JSONResponse # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]
import numpy as np
import cv2
import io

from model import predict

app = FastAPI(title="YOLO Object Detection API")

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "YOLO API Running "}


@app.post("/predict")
async def detect(file: UploadFile = File(...)):
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Only JPG/PNG allowed"}, status_code=400)

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    boxes, labels, scores = predict(img)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

        text = f"{labels[i]} ({scores[i]:.2f})"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    _, img_encoded = cv2.imencode(".png", img)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/png"
    )


@app.post("/predict-json")
async def detect_json(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    boxes, labels, scores = predict(img)

    return {
        "boxes": [list(map(int, box)) for box in boxes],
        "labels": labels,
        "scores": [float(s) for s in scores]
    }