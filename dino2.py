# ================================================================
# CAR INSURANCE FRAUD DETECTION
#
# DINOv2 + Logistic Regression
# EfficientNet-B0
# ConvNeXt-Tiny
#
# Weighted/Rank Ensemble
# Validation F2 Threshold Optimization
# One Untouched Test Set
# ================================================================

import os
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from scipy.stats import rankdata

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import models, transforms

from tqdm import tqdm


warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

IMAGE_SIZE = 224

NUM_WORKERS = 0

# Dataset root
DATASET_PATH = Path(
    r"C:\Users\kishore\Documents\archive\Insurance-Fraud-Detection\Insurance-Fraud-Detection"
)

OUTPUT_PATH = DATASET_PATH / "outputs_dino2"

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ================================================================
# DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 80)
print("DEVICE INFORMATION")
print("=" * 80)

print(
    "PyTorch version :",
    torch.__version__
)

print(
    "CUDA available  :",
    torch.cuda.is_available()
)

print(
    "Device          :",
    device
)

if torch.cuda.is_available():

    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# DATASET PATH
# ================================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

print(
    "Dataset root:"
)

print(
    DATASET_PATH
)


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        "\nDataset folder does not exist:\n"
        f"{DATASET_PATH}"
    )


# ================================================================
# IMAGE EXTENSIONS
# ================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ================================================================
# GET IMAGES
# ================================================================

def get_images(folder):

    folder = Path(folder)

    if not folder.exists():

        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


# ================================================================
# FIND DATASET FOLDERS
# ================================================================

TRAIN_FRAUD = DATASET_PATH / "train" / "Fraud"

TRAIN_NON_FRAUD = DATASET_PATH / "train" / "Non-Fraud"

TEST_FRAUD = DATASET_PATH / "test" / "Fraud"

TEST_NON_FRAUD = DATASET_PATH / "test" / "Non-Fraud"


# ================================================================
# CHECK FOLDER STRUCTURE
# ================================================================

folder_structure_exists = (

    TRAIN_FRAUD.exists()
    and
    TRAIN_NON_FRAUD.exists()
    and
    TEST_FRAUD.exists()
    and
    TEST_NON_FRAUD.exists()

)


# ================================================================
# LOAD FOLDER DATASET
# ================================================================

if folder_structure_exists:

    print()
    print(
        "Folder-based dataset detected."
    )

    train_fraud_images = get_images(
        TRAIN_FRAUD
    )

    train_nonfraud_images = get_images(
        TRAIN_NON_FRAUD
    )

    test_fraud_images = get_images(
        TEST_FRAUD
    )

    test_nonfraud_images = get_images(
        TEST_NON_FRAUD
    )


# ================================================================
# OTHERWISE SEARCH FOR LABELS.CSV
# ================================================================

else:

    print()
    print(
        "Folder structure not complete."
    )

    print(
        "Searching recursively for labels.csv..."
    )

    label_files = [

        p

        for p in DATASET_PATH.rglob("*")

        if p.is_file()
        and p.name.lower() == "labels.csv"

    ]


    if len(label_files) == 0:

        raise FileNotFoundError(

            "\nCould not find either:\n\n"

            "1. train/Fraud\n"
            "2. train/Non-Fraud\n"
            "3. test/Fraud\n"
            "4. test/Non-Fraud\n\n"

            "OR labels.csv inside:\n"

            f"{DATASET_PATH}"

        )


    LABEL_FILE = label_files[0]


    print()
    print(
        "Using labels.csv:"
    )

    print(
        LABEL_FILE
    )


    import pandas as pd


    labels_df = pd.read_csv(
        LABEL_FILE
    )


    print()
    print(
        "Labels columns:"
    )

    print(
        labels_df.columns.tolist()
    )


    # ------------------------------------------------------------
    # Find image column
    # ------------------------------------------------------------

    possible_image_columns = [

        "image",
        "image_name",
        "filename",
        "file_name",
        "file",
        "path",
        "image_path"

    ]


    image_column = None


    for col in possible_image_columns:

        if col in labels_df.columns:

            image_column = col

            break


    if image_column is None:

        raise ValueError(

            "\nCould not find image column."

            "\nAvailable columns: "

            f"{labels_df.columns.tolist()}"

        )


    if "label" not in labels_df.columns:

        raise ValueError(

            "\n'label' column missing."

            "\nAvailable columns: "

            f"{labels_df.columns.tolist()}"

        )


    # ------------------------------------------------------------
    # Find image
    # ------------------------------------------------------------

    def find_image(filename):

        filename = str(
            filename
        ).strip()


        # Direct path

        direct = Path(
            filename
        )

        if direct.exists():

            return direct


        # Relative to labels.csv

        candidate = (

            LABEL_FILE.parent
            /
            filename

        )

        if candidate.exists():

            return candidate


        # Relative to dataset

        candidate = (

            DATASET_PATH
            /
            filename

        )

        if candidate.exists():

            return candidate


        # Recursive search

        matches = list(

            DATASET_PATH.rglob(
                Path(filename).name
            )

        )

        if matches:

            return matches[0]


        return None


    labels_df["image_path"] = (

        labels_df[image_column]
        .apply(find_image)

    )


    # ------------------------------------------------------------
    # Normalize labels
    # ------------------------------------------------------------

    def normalize_label(value):

        if isinstance(
            value,
            str
        ):

            value = value.strip().lower()


            if value in [
                "fraud",
                "fake",
                "1",
                "true",
                "yes"
            ]:

                return 1


            if value in [
                "non-fraud",
                "nonfraud",
                "real",
                "0",
                "false",
                "no"
            ]:

                return 0


        return int(value)


    labels_df["label"] = (

        labels_df["label"]
        .apply(normalize_label)

    )


    # ------------------------------------------------------------
    # Remove missing images
    # ------------------------------------------------------------

    labels_df = labels_df[
        labels_df["image_path"].notna()
    ].copy()


    # ------------------------------------------------------------
    # Split labels dataset
    # ------------------------------------------------------------

    train_df, test_df = train_test_split(

        labels_df,

        test_size=0.25,

        random_state=SEED,

        stratify=labels_df["label"]

    )


    train_df, val_df = train_test_split(

        train_df,

        test_size=0.20,

        random_state=SEED,

        stratify=train_df["label"]

    )


    train_fraud_images = [

        Path(x)

        for x in train_df[
            train_df["label"] == 1
        ]["image_path"]

    ]


    train_nonfraud_images = [

        Path(x)

        for x in train_df[
            train_df["label"] == 0
        ]["image_path"]

    ]


    # For CSV mode we keep validation separately

    val_fraud_images = [

        Path(x)

        for x in val_df[
            val_df["label"] == 1
        ]["image_path"]

    ]


    val_nonfraud_images = [

        Path(x)

        for x in val_df[
            val_df["label"] == 0
        ]["image_path"]

    ]


    test_fraud_images = [

        Path(x)

        for x in test_df[
            test_df["label"] == 1
        ]["image_path"]

    ]


    test_nonfraud_images = [

        Path(x)

        for x in test_df[
            test_df["label"] == 0
        ]["image_path"]

    ]


# ================================================================
# CHECK IMAGE COUNTS
# ================================================================

if not train_fraud_images:

    raise RuntimeError(
        "No training Fraud images found."
    )


if not train_nonfraud_images:

    raise RuntimeError(
        "No training Non-Fraud images found."
    )


if not test_fraud_images:

    raise RuntimeError(
        "No test Fraud images found."
    )


if not test_nonfraud_images:

    raise RuntimeError(
        "No test Non-Fraud images found."
    )


# ================================================================
# FOLDER MODE VALIDATION SPLIT
# ================================================================

if folder_structure_exists:

    all_train_paths = np.array(

        train_fraud_images
        +
        train_nonfraud_images,

        dtype=object

    )


    all_train_labels = np.array(

        [1] * len(train_fraud_images)
        +
        [0] * len(train_nonfraud_images),

        dtype=np.int64

    )


    X_train_paths, X_val_paths, y_train, y_val = (

        train_test_split(

            all_train_paths,

            all_train_labels,

            test_size=0.20,

            random_state=SEED,

            stratify=all_train_labels

        )

    )


else:

    X_train_paths = np.array(

        train_fraud_images
        +
        train_nonfraud_images,

        dtype=object

    )


    y_train = np.array(

        [1] * len(train_fraud_images)
        +
        [0] * len(train_nonfraud_images),

        dtype=np.int64

    )


    X_val_paths = np.array(

        val_fraud_images
        +
        val_nonfraud_images,

        dtype=object

    )


    y_val = np.array(

        [1] * len(val_fraud_images)
        +
        [0] * len(val_nonfraud_images),

        dtype=np.int64

    )


# ================================================================
# TEST DATA
# ================================================================

test_paths = np.array(

    test_fraud_images
    +
    test_nonfraud_images,

    dtype=object

)


test_labels = np.array(

    [1] * len(test_fraud_images)
    +
    [0] * len(test_nonfraud_images),

    dtype=np.int64

)


# ================================================================
# DATASET INFORMATION
# ================================================================

print()
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print(
    "Train Fraud     :",
    np.sum(y_train == 1)
)

print(
    "Train Non-Fraud :",
    np.sum(y_train == 0)
)

print(
    "Validation Fraud:",
    np.sum(y_val == 1)
)

print(
    "Validation Non-Fraud:",
    np.sum(y_val == 0)
)

print(
    "Test Fraud      :",
    np.sum(test_labels == 1)
)

print(
    "Test Non-Fraud  :",
    np.sum(test_labels == 0)
)


# ================================================================
# DATA SPLIT
# ================================================================

print()
print("=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(
    "Training samples   :",
    len(X_train_paths)
)

print(
    "Validation samples :",
    len(X_val_paths)
)

print(
    "Test samples       :",
    len(test_paths)
)


print()
print("Training:")

print(
    "Fraud     :",
    np.sum(y_train == 1)
)

print(
    "Non-Fraud :",
    np.sum(y_train == 0)
)


print()
print("Validation:")

print(
    "Fraud     :",
    np.sum(y_val == 1)
)

print(
    "Non-Fraud :",
    np.sum(y_val == 0)
)


print()
print("Test:")

print(
    "Fraud     :",
    np.sum(test_labels == 1)
)

print(
    "Non-Fraud :",
    np.sum(test_labels == 0)
)


# ================================================================
# TRANSFORMS
# ================================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.00)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.20,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


eval_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# ================================================================
# DATASET CLASS
# ================================================================

class CarInsuranceDataset(
    Dataset
):

    def __init__(
        self,
        image_paths,
        labels,
        transform
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.transform = transform


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(self, idx):

        path = self.image_paths[idx]

        try:

            image = Image.open(
                path
            ).convert(
                "RGB"
            )

        except Exception as exc:

            raise RuntimeError(

                f"\nCould not read image:\n"
                f"{path}\n"
                f"{exc}"

            )


        image = self.transform(
            image
        )


        label = torch.tensor(

            int(
                self.labels[idx]
            ),

            dtype=torch.long

        )


        return image, label


# ================================================================
# DATASETS
# ================================================================

train_dataset = CarInsuranceDataset(

    X_train_paths,
    y_train,
    train_transform

)


val_dataset = CarInsuranceDataset(

    X_val_paths,
    y_val,
    eval_transform

)


test_dataset = CarInsuranceDataset(

    test_paths,
    test_labels,
    eval_transform

)


# ================================================================
# BALANCED SAMPLING
# ================================================================

class_counts = np.bincount(
    y_train
)


class_sample_weights = (

    1.0 /
    class_counts

)


sample_weights = np.array([

    class_sample_weights[y]

    for y in y_train

])


balanced_sampler = WeightedRandomSampler(

    torch.as_tensor(
        sample_weights,
        dtype=torch.double
    ),

    num_samples=len(
        sample_weights
    ),

    replacement=True

)


# ================================================================
# DATALOADERS
# ================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=balanced_sampler,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)


# IMPORTANT:
# DINO uses each original training image once.
# It does NOT use replacement sampling.

train_feature_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)


# ================================================================
# BALANCING INFORMATION
# ================================================================

print()
print("=" * 80)
print("CLASS BALANCING")
print("=" * 80)

print(
    "Fraud samples     :",
    np.sum(y_train == 1)
)

print(
    "Non-Fraud samples :",
    np.sum(y_train == 0)
)

print(
    "Fraud ratio       :",
    f"{np.mean(y_train == 1) * 100:.4f}%"
)

print(
    "Method            :",
    "WeightedRandomSampler"
)

print(
    "Training batches  :",
    len(train_loader)
)


# ================================================================
# EFFICIENTNET-B0
# ================================================================

print()
print("=" * 80)
print("LOADING PRETRAINED EFFICIENTNET-B0")
print("=" * 80)


eff_weights = (

    models.EfficientNet_B0_Weights.DEFAULT

)


efficientnet = models.efficientnet_b0(

    weights=eff_weights

)


eff_feature_dim = (

    efficientnet.classifier[
        1
    ].in_features

)


silu_count = sum(

    isinstance(
        m,
        nn.SiLU
    )

    for m in efficientnet.modules()

)


print(
    "SiLU / Swish layers:",
    silu_count
)


efficientnet.classifier = nn.Sequential(

    nn.Dropout(
        0.30
    ),

    nn.Linear(
        eff_feature_dim,
        2
    )

)


efficientnet = efficientnet.to(
    device
)


print(
    "EfficientNet-B0 loaded successfully."
)

print(
    "Feature dimension:",
    eff_feature_dim
)


# ================================================================
# CONVNEXT-TINY
# ================================================================

print()
print("=" * 80)
print("LOADING PRETRAINED CONVNEXT-T")
print("=" * 80)


conv_weights = (

    models.ConvNeXt_Tiny_Weights.DEFAULT

)


convnext = models.convnext_tiny(

    weights=conv_weights

)


conv_feature_dim = (

    convnext.classifier[
        2
    ].in_features

)


convnext.classifier[2] = nn.Linear(

    conv_feature_dim,

    2

)


convnext = convnext.to(
    device
)


print(
    "ConvNeXt-T loaded successfully."
)

print(
    "Feature dimension:",
    conv_feature_dim
)


# ================================================================
# LOSS FUNCTION
# ================================================================

# WeightedRandomSampler already balances
# CNN training batches.

criterion = nn.CrossEntropyLoss()


# ================================================================
# OPTIMIZERS
# ================================================================

optimizer_eff = optim.AdamW(

    efficientnet.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)


optimizer_conv = optim.AdamW(

    convnext.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(

    model,
    loader,
    optimizer,
    model_name

):

    model.train()


    total_loss = 0.0

    correct = 0

    total = 0


    for images, labels in tqdm(

        loader,

        desc=model_name,

        leave=True

    ):


        images = images.to(

            device,

            non_blocking=True

        )


        labels = labels.to(

            device,

            non_blocking=True

        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(

            outputs,

            labels

        )


        loss.backward()


        optimizer.step()


        total_loss += (

            loss.item()
            *
            images.size(0)

        )


        predictions = (

            outputs.argmax(
                dim=1
            )

        )


        correct += (

            predictions == labels

        ).sum().item()


        total += labels.size(0)


    return (

        total_loss / total,

        correct / total

    )


# ================================================================
# GET CNN PROBABILITIES
# ================================================================

def get_model_probabilities(

    model,
    loader

):

    model.eval()


    probabilities = []

    labels_all = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(

                device,

                non_blocking=True

            )


            outputs = model(
                images
            )


            probs = torch.softmax(

                outputs,

                dim=1

            )[:, 1]


            probabilities.extend(

                probs.cpu().numpy()

            )


            labels_all.extend(

                labels.numpy()

            )


    return (

        np.asarray(
            probabilities
        ),

        np.asarray(
            labels_all
        )

    )


# ================================================================
# TRAIN CNN
# ================================================================

def train_cnn_model(

    model,
    optimizer,
    name

):


    print()
    print("=" * 80)
    print(
        f"TRAINING {name}"
    )
    print("=" * 80)


    best_f1 = -1.0

    best_state = None


    for epoch in range(
        NUM_EPOCHS
    ):


        print()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )


        train_loss, train_acc = (

            train_one_epoch(

                model,

                train_loader,

                optimizer,

                name

            )

        )


        val_prob, val_true = (

            get_model_probabilities(

                model,

                val_loader

            )

        )


        val_pred = (

            val_prob >= 0.50

        ).astype(int)


        val_precision = precision_score(

            val_true,

            val_pred,

            zero_division=0

        )


        val_recall = recall_score(

            val_true,

            val_pred,

            zero_division=0

        )


        val_f1 = f1_score(

            val_true,

            val_pred,

            zero_division=0

        )


        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_acc:.4f}"
        )

        print(
            f"Val Precision  : {val_precision:.4f}"
        )

        print(
            f"Val Recall     : {val_recall:.4f}"
        )

        print(
            f"Val F1         : {val_f1:.4f}"
        )


        if val_f1 > best_f1:

            best_f1 = val_f1


            best_state = {

                k: v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()

            }


    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        model.to(
            device
        )


    print()

    print(
        f"Best {name} Validation F1: "
        f"{best_f1:.4f}"
    )


    return model


# ================================================================
# TRAIN EFFICIENTNET
# ================================================================

efficientnet = train_cnn_model(

    efficientnet,

    optimizer_eff,

    "EfficientNet-B0"

)


# ================================================================
# TRAIN CONVNEXT
# ================================================================

convnext = train_cnn_model(

    convnext,

    optimizer_conv,

    "ConvNeXt-T"

)


# ================================================================
# LOAD DINOv2
# ================================================================

print()
print("=" * 80)
print("LOADING PRETRAINED DINOV2")
print("=" * 80)


try:

    dinov2 = torch.hub.load(

        "facebookresearch/dinov2",

        "dinov2_vits14"

    )

except Exception as exc:

    raise RuntimeError(

        "DINOv2 could not be loaded.\n"
        "First run requires internet access.\n\n"
        f"Original error:\n{exc}"

    )


dinov2 = dinov2.to(
    device
)


dinov2.eval()


print(
    "DINOv2 loaded successfully."
)


# ================================================================
# DINO FEATURE EXTRACTION
# ================================================================

def extract_dino_features(

    loader,
    split_name

):


    features = []

    labels_all = []


    dinov2.eval()


    with torch.no_grad():

        for images, labels in tqdm(

            loader,

            desc=f"DINOv2 - {split_name}"

        ):


            images = images.to(

                device,

                non_blocking=True

            )


            feature_batch = (

                dinov2(images)

            )


            features.append(

                feature_batch
                .cpu()
                .numpy()

            )


            labels_all.extend(

                labels.numpy()

            )


    return (

        np.concatenate(
            features,
            axis=0
        ),

        np.asarray(
            labels_all
        )

    )


# ================================================================
# EXTRACT DINO TRAIN FEATURES
# ================================================================

print()

print(
    "Extracting DINOv2 training features..."
)


X_train_dino, y_train_dino = (

    extract_dino_features(

        train_feature_loader,

        "Training"

    )

)


# ================================================================
# EXTRACT DINO VALIDATION FEATURES
# ================================================================

print()

print(
    "Extracting DINOv2 validation features..."
)


X_val_dino, y_val_dino = (

    extract_dino_features(

        val_loader,

        "Validation"

    )

)


# ================================================================
# EXTRACT DINO TEST FEATURES
# ================================================================

print()

print(
    "Extracting DINOv2 test features..."
)


X_test_dino, y_test_dino = (

    extract_dino_features(

        test_loader,

        "Test"

    )

)


# ================================================================
# FEATURE INFORMATION
# ================================================================

print()
print("=" * 80)
print("DINOv2 FEATURE EXTRACTION COMPLETED")
print("=" * 80)

print(
    "Training features   :",
    X_train_dino.shape
)

print(
    "Validation features :",
    X_val_dino.shape
)

print(
    "Test features       :",
    X_test_dino.shape
)


# ================================================================
# DINO + LOGISTIC REGRESSION
# ================================================================

print()
print("=" * 80)
print("TRAINING DINOV2 + LOGISTIC REGRESSION")
print("=" * 80)


dino_lr = LogisticRegression(

    class_weight="balanced",

    max_iter=2000,

    C=1.0,

    solver="lbfgs",

    random_state=SEED

)


dino_lr.fit(

    X_train_dino,

    y_train_dino

)


print(
    "DINOv2 Logistic Regression trained successfully."
)


# ================================================================
# MODEL PREDICTIONS
# ================================================================

print()
print("=" * 80)
print("GENERATING MODEL PREDICTIONS")
print("=" * 80)


eff_val_prob, y_val_eff = (

    get_model_probabilities(

        efficientnet,

        val_loader

    )

)


eff_test_prob, y_test_eff = (

    get_model_probabilities(

        efficientnet,

        test_loader

    )

)


conv_val_prob, y_val_conv = (

    get_model_probabilities(

        convnext,

        val_loader

    )

)


conv_test_prob, y_test_conv = (

    get_model_probabilities(

        convnext,

        test_loader

    )

)


dino_val_prob = (

    dino_lr
    .predict_proba(
        X_val_dino
    )[:, 1]

)


dino_test_prob = (

    dino_lr
    .predict_proba(
        X_test_dino
    )[:, 1]

)


# ================================================================
# LABEL ORDER CHECK
# ================================================================

if not np.array_equal(

    y_val_eff,

    y_val_dino

):

    raise RuntimeError(

        "Validation label order mismatch."

    )


if not np.array_equal(

    y_test_eff,

    y_test_dino

):

    raise RuntimeError(

        "Test label order mismatch."

    )


y_val = y_val_eff

y_test = y_test_eff


# ================================================================
# WEIGHTED RANK ENSEMBLE
# ================================================================

print()
print("=" * 80)
print("RANK AVERAGING ENSEMBLE")
print("=" * 80)


# ConvNeXt gets slightly higher weight
# because it performed strongly in your previous results.

ENSEMBLE_WEIGHTS = (

    0.25,   # DINOv2 + LR

    0.30,   # EfficientNet

    0.45    # ConvNeXt

)


print(
    "DINOv2 + LR weight :",
    ENSEMBLE_WEIGHTS[0]
)

print(
    "EfficientNet weight:",
    ENSEMBLE_WEIGHTS[1]
)

print(
    "ConvNeXt-T weight  :",
    ENSEMBLE_WEIGHTS[2]
)


def normalized_rank(values):

    return (

        rankdata(

            values,

            method="average"

        )

        /

        len(values)

    )


def rank_average(

    dino_prob,

    eff_prob,

    conv_prob

):


    dino_rank = normalized_rank(
        dino_prob
    )


    eff_rank = normalized_rank(
        eff_prob
    )


    conv_rank = normalized_rank(
        conv_prob
    )


    return (

        ENSEMBLE_WEIGHTS[0]
        *
        dino_rank

        +

        ENSEMBLE_WEIGHTS[1]
        *
        eff_rank

        +

        ENSEMBLE_WEIGHTS[2]
        *
        conv_rank

    )


val_score = rank_average(

    dino_val_prob,

    eff_val_prob,

    conv_val_prob

)


test_score = rank_average(

    dino_test_prob,

    eff_test_prob,

    conv_test_prob

)


# ================================================================
# VALIDATION F2 THRESHOLD OPTIMIZATION
# ================================================================

print()
print("=" * 80)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


best_threshold = 0.50

best_f2 = -1.0


threshold_results = []


for threshold in np.arange(

    0.01,

    1.00,

    0.01

):


    pred = (

        val_score >= threshold

    ).astype(int)


    current_precision = precision_score(

        y_val,

        pred,

        zero_division=0

    )


    current_recall = recall_score(

        y_val,

        pred,

        zero_division=0

    )


    current_f1 = f1_score(

        y_val,

        pred,

        zero_division=0

    )


    current_f2 = fbeta_score(

        y_val,

        pred,

        beta=2,

        zero_division=0

    )


    threshold_results.append({

        "threshold": float(threshold),

        "precision": float(
            current_precision
        ),

        "recall": float(
            current_recall
        ),

        "f1": float(
            current_f1
        ),

        "f2": float(
            current_f2
        )

    })


    if current_f2 > best_f2:

        best_f2 = current_f2

        best_threshold = float(
            threshold
        )


print()

print(
    f"Best Threshold    : "
    f"{best_threshold:.2f}"
)

print(
    f"Best Validation F2: "
    f"{best_f2:.4f}"
)


# ================================================================
# VALIDATION RESULTS
# ================================================================

val_pred = (

    val_score >= best_threshold

).astype(int)


val_accuracy = accuracy_score(

    y_val,

    val_pred

)


val_precision = precision_score(

    y_val,

    val_pred,

    zero_division=0

)


val_recall = recall_score(

    y_val,

    val_pred,

    zero_division=0

)


val_f1 = f1_score(

    y_val,

    val_pred,

    zero_division=0

)


val_pr_auc = average_precision_score(

    y_val,

    val_score

)


val_roc_auc = roc_auc_score(

    y_val,

    val_score

)


print()
print("=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(
    f"Accuracy  : {val_accuracy:.4f}"
)

print(
    f"Precision : {val_precision:.4f}"
)

print(
    f"Recall    : {val_recall:.4f}"
)

print(
    f"F1        : {val_f1:.4f}"
)

print(
    f"F2        : {best_f2:.4f}"
)

print(
    f"PR-AUC    : {val_pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {val_roc_auc:.4f}"
)


# ================================================================
# FINAL TEST
# ================================================================

test_pred = (

    test_score >= best_threshold

).astype(int)


test_accuracy = accuracy_score(

    y_test,

    test_pred

)


test_precision = precision_score(

    y_test,

    test_pred,

    zero_division=0

)


test_recall = recall_score(

    y_test,

    test_pred,

    zero_division=0

)


test_f1 = f1_score(

    y_test,

    test_pred,

    zero_division=0

)


test_f2 = fbeta_score(

    y_test,

    test_pred,

    beta=2,

    zero_division=0

)


test_pr_auc = average_precision_score(

    y_test,

    test_score

)


test_roc_auc = roc_auc_score(

    y_test,

    test_score

)


# ================================================================
# TEST RESULTS
# ================================================================

print()
print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)


print(
    "Test set was NOT used for:"
)

print(
    "  - CNN training"
)

print(
    "  - DINO training"
)

print(
    "  - model selection"
)

print(
    "  - ensemble weight selection"
)

print(
    "  - threshold tuning"
)


print()

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-Score  : {test_f1:.4f}"
)

print(
    f"F2-Score  : {test_f2:.4f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.4f}"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print()
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


print(

    classification_report(

        y_test,

        test_pred,

        target_names=[

            "Non-Fraud",

            "Fraud"

        ],

        digits=4,

        zero_division=0

    )

)


# ================================================================
# CONFUSION MATRIX
# ================================================================

print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)


cm = confusion_matrix(

    y_test,

    test_pred

)


print(
    "                 Predicted"
)

print(
    "                 Non-Fraud   Fraud"
)

print(

    f"Actual Non-Fraud    "
    f"{cm[0,0]:5d}      "
    f"{cm[0,1]:5d}"

)

print(

    f"Actual Fraud        "
    f"{cm[1,0]:5d}      "
    f"{cm[1,1]:5d}"

)


# ================================================================
# SAVE MODELS
# ================================================================

print()
print("=" * 80)
print("SAVING MODELS")
print("=" * 80)


torch.save(

    efficientnet.state_dict(),

    OUTPUT_PATH
    /
    "efficientnet_b0_fraud.pth"

)


torch.save(

    convnext.state_dict(),

    OUTPUT_PATH
    /
    "convnext_tiny_fraud.pth"

)


joblib.dump(

    dino_lr,

    OUTPUT_PATH
    /
    "dinov2_logistic_regression.pkl"

)


# ================================================================
# SAVE THRESHOLD RESULTS
# ================================================================

with open(

    OUTPUT_PATH
    /
    "threshold_results.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        threshold_results,

        f,

        indent=4

    )


# ================================================================
# SAVE COMPLETE RESULTS
# ================================================================

results = {

    "models": [

        "DINOv2 + Logistic Regression",

        "EfficientNet-B0",

        "ConvNeXt-T"

    ],


    "ensemble":

        "Weighted Rank Averaging",


    "dino_weight":

        ENSEMBLE_WEIGHTS[0],


    "efficientnet_weight":

        ENSEMBLE_WEIGHTS[1],


    "convnext_weight":

        ENSEMBLE_WEIGHTS[2],


    "threshold_method":

        "Validation F2 Optimization",


    "threshold":

        best_threshold,


    "validation_accuracy":

        val_accuracy,


    "validation_precision":

        val_precision,


    "validation_recall":

        val_recall,


    "validation_f1":

        val_f1,


    "validation_f2":

        best_f2,


    "validation_pr_auc":

        val_pr_auc,


    "validation_roc_auc":

        val_roc_auc,


    "test_accuracy":

        test_accuracy,


    "test_precision":

        test_precision,


    "test_recall":

        test_recall,


    "test_f1":

        test_f1,


    "test_f2":

        test_f2,


    "test_pr_auc":

        test_pr_auc,


    "test_roc_auc":

        test_roc_auc,


    "confusion_matrix":

        cm.tolist()

}


with open(

    OUTPUT_PATH
    /
    "results.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print(
    "Model 1           : DINOv2 + Logistic Regression"
)

print(
    "Model 2           : EfficientNet-B0"
)

print(
    "Model 3           : ConvNeXt-T"
)

print(
    "Ensemble          : Weighted Rank Averaging"
)

print(
    "DINO Weight       :",
    ENSEMBLE_WEIGHTS[0]
)

print(
    "EfficientNet Weight:",
    ENSEMBLE_WEIGHTS[1]
)

print(
    "ConvNeXt Weight   :",
    ENSEMBLE_WEIGHTS[2]
)

print(
    "Threshold Method  : Validation F2 Optimization"
)

print(
    f"Threshold         : "
    f"{best_threshold:.2f}"
)

print(
    f"Test Accuracy     : "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Precision    : "
    f"{test_precision:.4f}"
)

print(
    f"Test Recall       : "
    f"{test_recall:.4f}"
)

print(
    f"Test F1           : "
    f"{test_f1:.4f}"
)

print(
    f"Test F2           : "
    f"{test_f2:.4f}"
)

print(
    f"Test PR-AUC       : "
    f"{test_pr_auc:.4f}"
)

print(
    f"Test ROC-AUC      : "
    f"{test_roc_auc:.4f}"
)


print()

print(
    "Files saved in:"
)

print(
    OUTPUT_PATH
)


print()
print("=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)