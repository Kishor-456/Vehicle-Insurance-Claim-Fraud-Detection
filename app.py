import os
from huggingface_hub import hf_hub_download

# ================================================================
# DOWNLOAD MODELS FROM HUGGING FACE
# ================================================================

HF_REPO_ID = "Joy0Antony/DinoV2"

def download_model(filename):
    local_path = OUTPUT_FOLDER / filename

    if local_path.exists():
        print(f"{filename} already exists.")
        return local_path

    print(f"Downloading {filename} from Hugging Face...")

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="model"
    )

    # Copy/downloaded file into our expected outputs_dino2 location
    import shutil
    shutil.copy2(downloaded_path, local_path)

    print(f"{filename} downloaded successfully.")

    return local_path


download_model("efficientnet_b0_fraud.pth")
download_model("convnext_tiny_fraud.pth")
download_model("dinov2_logistic_regression.pkl")
download_model("deployment_reference.npz")