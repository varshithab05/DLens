import tensorflow as tf
import numpy as np
from lime.lime_image import LimeImageExplainer
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
# from keras.utils import array_to_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image as keras_image
import cv2
# from IPython.display import display
import keras
import matplotlib as mpl
import io
import base64
import json
import gdown
import zipfile

# ====================
# Load class labels
# ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_DRIVE_LINK = "https://drive.google.com/drive/folders/1zUuEmP-03Mkst9yCrvR6fFjK1xb27ARV?usp=sharing"
MODEL_ZIP_PATH = "models.zip"

def download_models():
    """Downloads the model zip file."""
    if os.path.exists(MODEL_ZIP_PATH):
        print("Model file already exists. Skipping download.")
        return
    
    print("Downloading model...")
    gdown.download(GOOGLE_DRIVE_LINK, MODEL_ZIP_PATH, quiet=False)

    # Check if file downloaded successfully
    if not os.path.exists(MODEL_ZIP_PATH):
        raise FileNotFoundError("Model download failed!")

    print("Download complete.")

# Construct paths dynamically
CIFAR100_LABELS_PATH = os.path.join(BASE_DIR, "class_labels", "cifar100.json")
MNIST_LABELS_PATH = os.path.join(BASE_DIR, "class_labels", "mnist.json")
IMAGENET_LABELS_PATH = os.path.join(BASE_DIR, "class_labels", "imagenet.json")


# Load class labels
with open(CIFAR100_LABELS_PATH, "r") as f:
    CIFAR100_LABELS = json.load(f)

with open(MNIST_LABELS_PATH, "r") as f:
    MNIST_LABELS = json.load(f)

with open(IMAGENET_LABELS_PATH, "r") as f:
    IMAGENET_LABELS = json.load(f)


# ====================
# Utility Functions
# ====================
# Load model function
# def load_model(model_name: str):
#     model_path = os.path.join("models", f"{model_name}.h5")
#     model = tf.keras.models.load_model(model_path)
#     return model
def load_model(model_name: str):
    """Loads a specific model by name, downloading if necessary."""
    download_models()  # Ensure models are downloaded

    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}. Make sure the correct model is uploaded.")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    return model

def get_model_info():
    """
    Returns information about available models and explainability methods.
    """
    models = ["mobilenet", "resnet", "vgg16"]
    explain_methods = ["grad-cam", "shap", "lime", "integrated-gradients", "smoothgrad", "saliency-map"]
    return {"models": models, "explainability_methods": explain_methods}

def get_dataset_info(dataset):
    """
    Returns information about the specified dataset, such as the number of images and classes.
    """
    datasets = {
        "mnist": {"num_images": 70000, "num_classes": 10},
        "cifar100": {"num_images": 60000, "num_classes": 10},
        "imagenet": {"num_images": 10000, "num_classes": 9},
    }
    return datasets.get(dataset, {"error": "Dataset not found"})

def get_class_label(dataset, class_index):
    """
    Converts a class index to a human-readable label based on the dataset.
    """
    if dataset == "cifar100":
        return CIFAR100_LABELS[class_index] if class_index < len(CIFAR100_LABELS) else "Unknown"
    elif dataset == "mnist":
        return MNIST_LABELS[class_index] if class_index < len(MNIST_LABELS) else "Unknown"
    elif dataset == "imagenet":
        return IMAGENET_LABELS[class_index] if class_index < len(IMAGENET_LABELS) else "Unknown"
    else:
        return "Dataset not supported"

    
def upload_and_predict(file, model_name, dataset):
    """
    Handles image upload and returns the prediction from the model with class label.
    """
    model = load_model(model_name)
    image = Image.open(file.file)
    image_array = preprocess_image(image)
    preds = model.predict(image_array)
    class_index = np.argmax(preds)
    class_label = get_class_label(dataset, class_index)
    
    return {
        "prediction": int(class_index),
        "class_label": class_label,
        "confidence": float(preds[0][class_index])
    }

def run_explainability(file, model_name, method):
    """
    Runs a specific explainability method on the uploaded image.
    """
    if method == "grad-cam":
        return explain_with_grad_cam(model_name, file.file)
    elif method == "lime":
        return explain_with_lime(model_name, file.file)
    elif method == "saliency-map":
        return explain_with_saliency_map(model_name, file.file)
    else:
        return {"error": "Invalid explainability method"}


def run_all_explainability(file, model_name):
    """
    Runs all explainability methods on the uploaded image.
    """
    methods = ["grad-cam", "lime","saliency-map"]
    results = {}
    for method in methods:
        results[method] = run_explainability(file, model_name, method)
    return results

def get_random_image(dataset):
    """
    Fetches a random image from the specified dataset.
    """
    dataset_path = os.path.join("datasets", dataset)
    if not os.path.exists(dataset_path):
        return {"error": "Dataset not found"}
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return {"error": "No images in dataset"}
    random_image = np.random.choice(image_files)
    return os.path.join(dataset_path, random_image)


def preprocess_image(image, target_size=(32, 32)):
    image = image.resize(target_size)
    
    # Convert to RGB if it has 4 channels (e.g., RGBA) or is grayscale
    if image.mode == 'RGBA':  # Remove alpha channel
        image = image.convert('RGB')
    elif image.mode == 'L':  # Grayscale to RGB
        image = image.convert('RGB')

    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0

    # Ensure it has 3 channels (RGB)
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected 3 channels (RGB), but got shape {image_array.shape}")

    # Add batch dimension for model input
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def dataset_name(model_name):
    if(model_name == "mobilenet_mnist" or model_name == "resnet_mnist" or model_name == "vgg_mnist"):
        return "mnist"
    if(model_name == "mobilenet_cifar100" or model_name == "resnet_cifar100" or model_name == "vgg_cifar100"):
        return "cifar100"
    if(model_name == "mobilenet_imagenet" or model_name == "resnet_imagenet" or model_name == "vgg_imagenet"):
        return "imagenet"

# ====================
# Explainability Methods
# ====================
from app.explainability.grad_cam import GradCAM 

def explain_with_grad_cam(model_name: str, image_file, output_dir="output"):
    import os
    import cv2

    # Load model
    model = load_model(model_name)
    
    # Preprocess the image
    image = Image.open(image_file)
    image_array = preprocess_image(image)
    
    # Resize image array to the expected shape
    expected_shape = model.input_shape[1:3]
    if image_array.shape[1:3] != expected_shape:
        image_array = tf.image.resize(image_array, expected_shape)

    if model_name == "vgg_imagenet":
        image_array = np.array(image, dtype=np.float32)  # Convert to writable float32 array
        image_array = cv2.resize(image_array, (224, 224))
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

        if image_array.shape[-1] != 3:  # Ensure it's RGB
            raise ValueError("Expected an RGB image with 3 channels.")

        image_array = vgg16_preprocess_input(image_array)  

    if model_name == "resnet_imagenet":
        target_size=(224, 224)
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to array and preprocess
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Apply the correct preprocessing function
        image_array = resnet_preprocess_input(image_array) 
    
    # Get the target class index (e.g., the predicted class)
    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions)
    dataset = dataset_name(model_name)
    class_label = get_class_label(dataset=dataset, class_index=class_idx)
    
    # Initialize GradCAM
    layer_name = ''
    if model_name == "mobilenet_imagenet" or model_name == "mobilenet_cifar100":
        layer_name = 'conv_pw_11' 
    elif model_name == "mobilenet_mnist":
        layer_name = 'Conv_1'
    elif model_name == "resnet_imagenet" or model_name == "resnet_cifar100":
        layer_name = 'conv5_block3_out'
    elif model_name == 'resnet_mnist':
        layer_name = 'conv5_block3_1_conv'
    elif model_name == "vgg_mnist" or model_name == "vgg_cifar100" or model_name == "vgg_imagenet":
        layer_name = 'block5_conv1'

    grad_cam = GradCAM(model, classIdx=class_idx, layerName=layer_name)
    
    # Compute heatmap
    heatmap = grad_cam.compute_heatmap(image_array)

    # Convert heatmap to a format suitable for visualization
    original_image = np.array(image.resize((image_array.shape[2], image_array.shape[1])))

    # Upscale the original image for better visualization
    upscale_size = (224, 224)  # Example size
    original_image = cv2.resize(original_image, upscale_size, interpolation=cv2.INTER_CUBIC)

    # Overlay heatmap with more prominent visualization
    heatmap, overlay = grad_cam.overlay_heatmap(heatmap, original_image, alpha=0.7, colormap=cv2.COLORMAP_JET)

    overlay_Image = Image.fromarray(overlay)
    overlay_base64 = encode_image_to_base64(overlay_Image)

    print("class_idx:", class_idx)
    print("prediction", predictions)
    print("class_label:", class_label)
    print("confidence:", float(predictions[0][class_idx]))   
    return {
        "overlay_base64": overlay_base64,
        "prediction": int(class_idx),
        "class_label": class_label,
        "confidence": float(predictions[0][class_idx])
    }

import lime
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

def explain_with_lime(model_name: str, image_file, output_dir="output"):
    """
    Generates an explanation using LIME and saves the overlay image.
    """
    # Load model
    model = load_model(model_name)
    
    # Preprocess image
    image = Image.open(image_file)
    image_array = preprocess_image(image)  # Shape: (1, height, width, channels)
    if(model_name == "mobilenet_imagenet" or model_name == 'resnet_imagenet'):
        image_array = tf.image.resize(image_array, (224, 224))

    image_np = np.squeeze(image_array, axis=0)  # Remove batch dimension
    expected_shape = model.input_shape[1:3]
    if image_array.shape[1:3] != expected_shape:
        image_array = tf.image.resize(image_array, expected_shape)
    
    if model_name == "vgg_imagenet":
        image_array = np.array(image, dtype=np.float32)  # Convert to writable float32 array
        image_array = cv2.resize(image_array, (224, 224))
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

        if image_array.shape[-1] != 3:  # Ensure it's RGB
            raise ValueError("Expected an RGB image with 3 channels.")

        image_array = vgg16_preprocess_input(image_array)  

    if model_name == "resnet_imagenet":
        target_size=(224, 224)
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to array and preprocess
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Apply the correct preprocessing function
        image_array = resnet_preprocess_input(image_array) 
    
    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions)
    dataset = dataset_name(model_name)
    class_label = get_class_label(dataset=dataset,class_index=class_idx)

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    # Get superpixel mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    
    # Overlay the mask on the image
    overlay = mark_boundaries(temp, mask)
    
    # # Ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    # overlay_path = os.path.join(output_dir, "lime_overlay.png")
    
    # # Save image
    # plt.imsave(overlay_path, overlay)
    
    # return {"overlay_path": overlay_path}
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_base64 = encode_image_to_base64(overlay_image)
    # return {"overlay_base64": overlay_base64}
    return {
        "overlay_base64": overlay_base64,
        "prediction": int(class_idx),
        "class_label": class_label,
        "confidence": float(predictions[0][class_idx])
    }


def explain_with_saliency_map(model_name: str, image_file, output_dir="output"):
    """
    Generates a saliency map using gradients and saves the overlay image.
    """
    # Load model
    model = load_model(model_name)
    
    # Preprocess image
    image = Image.open(image_file)
    image_array = preprocess_image(image)
    if(model_name == "mobilenet_imagenet" or model_name == 'resnet_imagenet'):
        image_array = tf.image.resize(image_array, (224, 224))
    expected_shape = model.input_shape[1:3]
    if image_array.shape[1:3] != expected_shape:
        image_array = tf.image.resize(image_array, expected_shape)
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_array)
    # input_tensor = tf.Variable(input_tensor, dtype=tf.float32)
    input_tensor = tf.Variable(tf.cast(input_tensor, tf.float32))

    if model_name == "vgg_imagenet":
        image_array = np.array(image, dtype=np.float32)  # Convert to writable float32 array
        image_array = cv2.resize(image_array, (224, 224))
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

        if image_array.shape[-1] != 3:  # Ensure it's RGB
            raise ValueError("Expected an RGB image with 3 channels.")

        image_array = vgg16_preprocess_input(image_array)  

    if model_name == "resnet_imagenet":
        target_size=(224, 224)
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to array and preprocess
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Apply the correct preprocessing function
        image_array = resnet_preprocess_input(image_array) 

    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions)
    dataset = dataset_name(model_name)
    class_label = get_class_label(dataset=dataset,class_index=class_idx)
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[0][class_idx]
    
    grads = tape.gradient(loss, input_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    
    # Normalize and convert to uint8
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    saliency = np.uint8(255 * saliency.numpy())
    
    # # Ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    # overlay_path = os.path.join(output_dir, "saliency_map.png")
    
    # # Save image
    # cv2.imwrite(overlay_path, saliency)
    
    # return {"overlay_path": overlay_path}
    saliency_image = Image.fromarray(saliency)
    saliency_base64 = encode_image_to_base64(saliency_image)
    # return {"overlay_base64": saliency_base64}
    return {
        "overlay_base64": saliency_base64,
        "prediction": int(class_idx),
        "class_label": class_label,
        "confidence": float(predictions[0][class_idx])
    }
