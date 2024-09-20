from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Importa la clase YOLO desde ultralytics
  # Importa tu clase YOLO
from io import BytesIO
import uvicorn
import os

app = FastAPI()

# Cargar el modelo YOLO
model_path = 'Backend/FastAPI/models/best.pt'
model = YOLO(model_path)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    # Leer la imagen desde el archivo subido
    image_data = await image.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Hacer la predicción
    results = model.predict(image, save=False, conf=0.8)

    for result in results:  # Itera sobre cada resultado (una imagen)
        for box in result.boxes:  # Itera sobre cada caja detectada
            coords = box.xyxy[0].cpu().numpy().astype(int)  # Coordenadas de la caja
            class_id = int(box.cls[0].cpu().numpy())  # ID de la clase
            x_min, y_min, x_max, y_max = coords

            # Dibujar la caja delimitadora
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Poner la etiqueta encima de la caja
            label = f'Class {class_id}'
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Contar las varillas detectadas
    count = sum([len(result.boxes) for result in results])
    return JSONResponse(content={"count": count})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render asigna el puerto dinámicamente
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=12000)