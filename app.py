# ================================================================
# CAR INSURANCE FRAUD DETECTION - FLASK APP
# ================================================================
#
# Models:
#   1. DINOv2 + Logistic Regression
#   2. EfficientNet-B0
#   3. ConvNeXt-Tiny
#
# Ensemble:
#   Weighted Rank Averaging using deployment_reference.npz
#
# Project folder:
#   C:\Users\kishore\Documents\car_insurance_app
#
# Models are loaded from:
#   .\outputs_dino2
# ================================================================

import json
import uuid
from pathlib import Path


import joblib
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import models, transforms


# ================================================================
# FLASK CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs_dino2"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "bmp"
}


# ================================================================
# DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("CAR INSURANCE FRAUD DETECTION")
print("=" * 80)
print("Project directory :", BASE_DIR)
print("Output directory  :", OUTPUT_FOLDER)
print("Device            :", device)

if torch.cuda.is_available():
    print("GPU               :", torch.cuda.get_device_name(0))


# ================================================================
# MODEL PATHS
# ================================================================

EFF_PATH = OUTPUT_FOLDER / "efficientnet_b0_fraud.pth"
CONV_PATH = OUTPUT_FOLDER / "convnext_tiny_fraud.pth"
DINO_LR_PATH = OUTPUT_FOLDER / "dinov2_logistic_regression.pkl"

RESULTS_PATH = OUTPUT_FOLDER / "results.json"
THRESHOLD_PATH = OUTPUT_FOLDER / "threshold_results.json"
REFERENCE_PATH = OUTPUT_FOLDER / "deployment_reference.npz"


# ================================================================
# VERIFY MODEL FILES
# ================================================================

print()
print("=" * 80)
print("MODEL PATH CHECK")
print("=" * 80)

print("EfficientNet :", EFF_PATH)
print("Exists       :", EFF_PATH.exists())

print("ConvNeXt     :", CONV_PATH)
print("Exists       :", CONV_PATH.exists())

print("DINO + LR    :", DINO_LR_PATH)
print("Exists       :", DINO_LR_PATH.exists())

print("Reference    :", REFERENCE_PATH)
print("Exists       :", REFERENCE_PATH.exists())

missing = [
    path for path in
    [EFF_PATH, CONV_PATH, DINO_LR_PATH]
    if not path.exists()
]

if missing:
    raise FileNotFoundError(
        "\nRequired model file(s) not found:\n"
        + "\n".join(str(path) for path in missing)
        + "\n\nExpected project structure:\n"
        + str(BASE_DIR / "outputs_dino2")
    )


# ================================================================
# ENSEMBLE CONFIGURATION
# ================================================================

DINO_WEIGHT = 0.25
EFFICIENTNET_WEIGHT = 0.30
CONVNEXT_WEIGHT = 0.45

print()
print("=" * 80)
print("ENSEMBLE CONFIGURATION")
print("=" * 80)
print("DINOv2 weight       :", DINO_WEIGHT)
print("EfficientNet weight :", EFFICIENTNET_WEIGHT)
print("ConvNeXt weight     :", CONVNEXT_WEIGHT)


# ================================================================
# IMAGE TRANSFORM
# ================================================================

IMAGE_SIZE = 224

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# LOAD THRESHOLD
# ================================================================

DEFAULT_THRESHOLD = 0.50
best_threshold = DEFAULT_THRESHOLD


def find_threshold(obj):
    """
    Try several common threshold key names so the app works
    with either results.json or threshold_results.json.
    """

    if isinstance(obj, dict):

        possible_keys = [
            "threshold",
            "best_threshold",
            "f2_threshold",
            "optimal_threshold",
            "selected_threshold"
        ]

        for key in possible_keys:
            value = obj.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        # Search nested dictionaries/lists.
        for value in obj.values():

            result = find_threshold(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:

            result = find_threshold(value)

            if result is not None:
                return result

    return None


for threshold_file in [RESULTS_PATH, THRESHOLD_PATH]:

    if not threshold_file.exists():
        continue

    try:

        with open(
            threshold_file,
            "r",
            encoding="utf-8"
        ) as f:

            threshold_data = json.load(f)

        found_threshold = find_threshold(
            threshold_data
        )

        if found_threshold is not None:

            best_threshold = found_threshold
            break

    except Exception as exc:

        print(
            f"Warning: Could not read {threshold_file.name}: {exc}"
        )


print()
print("Threshold:", best_threshold)


# ================================================================
# LOAD EFFICIENTNET-B0
# ================================================================

print()
print("=" * 80)
print("LOADING EFFICIENTNET-B0")
print("=" * 80)

efficientnet = models.efficientnet_b0(
    weights=None
)

eff_feature_dim = (
    efficientnet.classifier[1].in_features
)

efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(eff_feature_dim, 2)
)

efficientnet.load_state_dict(
    torch.load(
        EFF_PATH,
        map_location=device
    )
)

efficientnet = efficientnet.to(device)
efficientnet.eval()

print("EfficientNet-B0 loaded successfully.")


# ================================================================
# LOAD CONVNEXT-TINY
# ================================================================

print()
print("=" * 80)
print("LOADING CONVNEXT-TINY")
print("=" * 80)

convnext = models.convnext_tiny(
    weights=None
)

conv_feature_dim = (
    convnext.classifier[2].in_features
)

convnext.classifier[2] = nn.Linear(
    conv_feature_dim,
    2
)

convnext.load_state_dict(
    torch.load(
        CONV_PATH,
        map_location=device
    )
)

convnext = convnext.to(device)
convnext.eval()

print("ConvNeXt-Tiny loaded successfully.")


# ================================================================
# LOAD DINOV2
# ================================================================

print()
print("=" * 80)
print("LOADING DINOV2")
print("=" * 80)

try:

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

except Exception as exc:

    raise RuntimeError(
        "\nDINOv2 could not be loaded.\n"
        "The first run may need internet access so that "
        "torch.hub can download DINOv2.\n\n"
        f"Original error:\n{exc}"
    )

dinov2 = dinov2.to(device)
dinov2.eval()

print("DINOv2 loaded successfully.")


# ================================================================
# LOAD DINO LOGISTIC REGRESSION
# ================================================================

print()
print("=" * 80)
print("LOADING DINOv2 LOGISTIC REGRESSION")
print("=" * 80)

dino_lr = joblib.load(
    DINO_LR_PATH
)

print("DINOv2 Logistic Regression loaded successfully.")


# ================================================================
# LOAD DEPLOYMENT REFERENCE
# ================================================================

reference_data = None

if REFERENCE_PATH.exists():

    try:

        reference_data = np.load(
            REFERENCE_PATH,
            allow_pickle=False
        )

        print()
        print("=" * 80)
        print("DEPLOYMENT REFERENCE")
        print("=" * 80)
        print(
            "Reference arrays:",
            reference_data.files
        )

    except Exception as exc:

        print(
            "Warning: deployment_reference.npz could not be loaded:",
            exc
        )


# ================================================================
# REFERENCE HELPERS
# ================================================================

def find_reference_array(possible_names):
    """
    Find a reference probability array inside
    deployment_reference.npz.
    """

    if reference_data is None:
        return None

    for name in possible_names:

        if name in reference_data.files:

            array = np.asarray(
                reference_data[name]
            ).astype(float).ravel()

            array = array[
                np.isfinite(array)
            ]

            if len(array) > 0:
                return array

    return None


def reference_rank(score, reference):
    """
    Convert a model probability into a percentile-like
    rank using the deployment reference scores.
    """

    if reference is None:
        return float(score)

    reference = np.asarray(
        reference
    ).astype(float).ravel()

    reference = reference[
        np.isfinite(reference)
    ]

    if len(reference) == 0:
        return float(score)

    reference = np.sort(reference)

    position = np.searchsorted(
        reference,
        score,
        side="right"
    )

    rank = position / len(reference)

    rank = max(
        rank,
        1.0 / len(reference)
    )

    return float(rank)


# ================================================================
# PREDICTION
# ================================================================

def predict_image(image):

    # ------------------------------------------------------------
    # Prepare image
    # ------------------------------------------------------------

    image_tensor = eval_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # ------------------------------------------------------------
    # EfficientNet prediction
    # ------------------------------------------------------------

    with torch.no_grad():

        eff_output = efficientnet(
            image_tensor
        )

        eff_probability = torch.softmax(
            eff_output,
            dim=1
        )[0, 1].item()


    # ------------------------------------------------------------
    # ConvNeXt prediction
    # ------------------------------------------------------------

    with torch.no_grad():

        conv_output = convnext(
            image_tensor
        )

        conv_probability = torch.softmax(
            conv_output,
            dim=1
        )[0, 1].item()


    # ------------------------------------------------------------
    # DINOv2 feature extraction
    # ------------------------------------------------------------

    with torch.no_grad():

        dino_features = dinov2(
            image_tensor
        )

    dino_features = (
        dino_features
        .detach()
        .cpu()
        .numpy()
    )


    # ------------------------------------------------------------
    # DINOv2 + Logistic Regression
    # ------------------------------------------------------------

    dino_probability = float(
        dino_lr.predict_proba(
            dino_features
        )[0, 1]
    )


    # ------------------------------------------------------------
    # Reference probability arrays
    # ------------------------------------------------------------

    dino_reference = find_reference_array([
        "dino_val_prob",
        "dino_validation_prob",
        "dino_prob",
        "dino_scores",
        "dino"
    ])

    eff_reference = find_reference_array([
        "eff_val_prob",
        "efficientnet_val_prob",
        "efficientnet_prob",
        "eff_prob",
        "eff_scores",
        "efficientnet"
    ])

    conv_reference = find_reference_array([
        "conv_val_prob",
        "convnext_val_prob",
        "convnext_prob",
        "conv_prob",
        "conv_scores",
        "convnext"
    ])


    # ------------------------------------------------------------
    # Rank conversion
    # ------------------------------------------------------------

    dino_rank = reference_rank(
        dino_probability,
        dino_reference
    )

    eff_rank = reference_rank(
        eff_probability,
        eff_reference
    )

    conv_rank = reference_rank(
        conv_probability,
        conv_reference
    )


    # ------------------------------------------------------------
    # Weighted rank ensemble
    # ------------------------------------------------------------

    ensemble_score = (
        DINO_WEIGHT * dino_rank
        + EFFICIENTNET_WEIGHT * eff_rank
        + CONVNEXT_WEIGHT * conv_rank
    )


    # ------------------------------------------------------------
    # Final classification
    # ------------------------------------------------------------

    is_fraud = (
        ensemble_score >= best_threshold
    )


    # ------------------------------------------------------------
    # Risk percentage
    # ------------------------------------------------------------

    risk_percentage = max(
        0.0,
        min(
            100.0,
            ensemble_score * 100.0
        )
    )


    # ------------------------------------------------------------
    # Risk level
    # ------------------------------------------------------------

    if risk_percentage >= 75:
        risk_level = "HIGH"

    elif risk_percentage >= 50:
        risk_level = "MEDIUM"

    elif risk_percentage >= 25:
        risk_level = "LOW"

    else:
        risk_level = "VERY LOW"


    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    return {
        "prediction": (
            "FRAUD"
            if is_fraud
            else "NON-FRAUD"
        ),

        "fraud": bool(is_fraud),

        "risk_percentage": round(
            risk_percentage,
            2
        ),

        "risk_level": risk_level,

        "ensemble_score": round(
            ensemble_score,
            6
        ),

        "threshold": round(
            best_threshold,
            4
        ),

        "models": {
            "DINOv2 + Logistic Regression": round(
                dino_probability * 100,
                2
            ),

            "EfficientNet-B0": round(
                eff_probability * 100,
                2
            ),

            "ConvNeXt-Tiny": round(
                conv_probability * 100,
                2
            )
        },

        "rank_scores": {
            "DINOv2": round(
                dino_rank,
                4
            ),

            "EfficientNet-B0": round(
                eff_rank,
                4
            ),

            "ConvNeXt-Tiny": round(
                conv_rank,
                4
            )
        },

        "weights": {
            "DINOv2": DINO_WEIGHT,
            "EfficientNet-B0": EFFICIENTNET_WEIGHT,
            "ConvNeXt-Tiny": CONVNEXT_WEIGHT
        }
    }


# ================================================================
# FILE VALIDATION
# ================================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# ================================================================
# HOME
# ================================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ================================================================
# PREDICT
# ================================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": (
                    "No image uploaded. "
                    "The HTML form must use "
                    "name='image'."
                )
            }), 400


        file = request.files["image"]


        if not file or file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400


        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error": (
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG, WEBP or BMP."
                )
            }), 400


        # --------------------------------------------------------
        # Create a unique safe filename
        # --------------------------------------------------------

        original_filename = file.filename

        safe_name = secure_filename(
            original_filename
        )

        unique_name = (
            f"{uuid.uuid4().hex}_{safe_name}"
        )

        image_path = (
            UPLOAD_FOLDER /
            unique_name
        )


        # --------------------------------------------------------
        # Save image
        # --------------------------------------------------------

        file.save(
            image_path
        )


        # --------------------------------------------------------
        # Open image
        # --------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # --------------------------------------------------------
        # Predict
        # --------------------------------------------------------

        result = predict_image(
            image
        )

        result["success"] = True
        result["filename"] = original_filename


        # --------------------------------------------------------

        # Remove uploaded file after prediction
        # --------------------------------------------------------

        try:
            image_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


        return jsonify(
            result
        )


    except Exception as exc:

        print()
        print("=" * 80)
        print("PREDICTION ERROR")
        print("=" * 80)
        print(exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ================================================================
# HEALTH CHECK
# ================================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "OK",

        "device": str(device),

        "project_directory": str(
            BASE_DIR
        ),

        "output_directory": str(
            OUTPUT_FOLDER
        ),

        "models": {
            "DINOv2": True,
            "EfficientNet-B0": True,
            "ConvNeXt-Tiny": True
        },

        "threshold": best_threshold,

        "ensemble_weights": {
            "DINOv2": DINO_WEIGHT,
            "EfficientNet-B0": EFFICIENTNET_WEIGHT,
            "ConvNeXt-Tiny": CONVNEXT_WEIGHT
        }
    })


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("ALL MODELS LOADED")
    print("=" * 80)

    print(
        "Open browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )