from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from app.utils import get_model_info, get_dataset_info, upload_and_predict, run_explainability, run_all_explainability, get_random_image, download_models
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

download_models()  # Download models and datasets on startup

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change in production)
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],   # Allow all headers
)

# Other route definitions...
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

class ImageFile(BaseModel):
    file: UploadFile

# Model and Dataset Information Routes
@app.get("/get-model-info")
def get_model_info_route():
    model_info = get_model_info()  # Function to return available models and explainability methods
    return JSONResponse(content=model_info)

@app.get("/get-dataset-info")
def get_dataset_info_route(dataset: str):
    dataset_info = get_dataset_info(dataset)  # Function to get details of the dataset
    return JSONResponse(content=dataset_info)

# Image Upload and Inference Routes
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), model_name: str = "mobilenet_mnist"):
    result = upload_and_predict(file, model_name)  # Function to upload image and get inference result
    return JSONResponse(content=result)

@app.post("/run-explainability")
async def run_explainability_route(file: UploadFile = File(...),  model_name: str=Form(...), method: str=Form(...)):
    print("Running explainability")
    print("model_name: ", model_name)
    print("method: ", method)
    result = run_explainability(file, model_name, method)  # Function to run explainability method
    if isinstance(result, np.ndarray):
        result = result.tolist()
    elif isinstance(result, dict):  # If result is a dictionary, handle nested arrays
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
    return JSONResponse(content=result)

@app.post("/run-all-explainability")
async def run_all_explainability_route(file: UploadFile = File(...), model_name: str = "mobilenet_mnist"):
    result = run_all_explainability(file, model_name)  # Function to run all explainability methods
    if isinstance(result, np.ndarray):
        result = result.tolist()
    elif isinstance(result, dict):  # If result is a dictionary, handle nested arrays
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
    return JSONResponse(content=result)

# Fetch Random Dataset Image Route
@app.get("/get-dataset-image")
def get_random_image_route(dataset: str):
    image_url = get_random_image(dataset)  # Function to fetch a random image from the dataset
    return JSONResponse(content={"image_url": image_url})

# Model Prediction Route
@app.post("/get-model-prediction")
async def get_model_prediction_route(file: UploadFile = File(...), model_name: str = "mobilenet_mnist"):
    result = upload_and_predict(file, model_name)  # Simplified prediction function
    return JSONResponse(content=result)
