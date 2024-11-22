# import os
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from .utils import process_large_image

app = FastAPI()

REPO_ID = "rasyadlubisdev/waste-classifier"
FILENAME = "waste_classification_cnn_model_HybridModel_WS.h5"

# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, "waste_classification_hybrid_model.h5")
# hybrid_model = load_model(model_path)
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
hybrid_model = load_model(model_path)
class_labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Textile Trash']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    img = image.load_img(file_location, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = hybrid_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    return {
        "predicted_class": class_labels[predicted_class],
        "confidence": f"{confidence:.2f}%"
    }


@app.post("/predict-large/")
async def predict_large(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    patch_size = (128, 128)
    step_size = 64
    results = process_large_image(file_location, hybrid_model, patch_size, step_size, class_labels)

    return {"results": results}