


import os
import re
import json
import time
import gdown
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
import pandas as pd

# -----------------------------
# CONFIG YOU WILL EDIT (Drive links)
# -----------------------------
DRIVE_FILES = {
    # Models
    "teacher": "https://drive.google.com/file/d/1ONNWGs1dF1QDZ1PHhUfLoln6ghs0KHV4/view?usp=sharing",
    "vit_baseline": "https://drive.google.com/file/d/1MnnO7201oKELM8qg2GhXUwaPykxQxeMI/view?usp=sharing",
    "vit_kd": "https://drive.google.com/file/d/1_IsRThzM0S-qG45bzwh-tmlECOy_sVsC/view?usp=sharing",

    # Result images (we will add later)
    # Result images
    "cm_teacher": "https://drive.google.com/file/d/16Ui9tuooBsPe0DsQRvCydZ2I2ofYLvUt/view?usp=sharing",
    "cm_vit": "https://drive.google.com/file/d/1sav1gtWkpaHW-pwvYzTEOUry3K0YUt3N/view?usp=sharing",
    "cm_kd": "https://drive.google.com/file/d/1wRcVCftYitwr8nT8ydIgjrK2TSotTHBh/view?usp=sharing",
    "kd_curve_acc": "https://drive.google.com/file/d/1qG2-cuV_zETqd_8MW2sBnTpLw6JJ3vZg/view?usp=sharing",
    "kd_curve_loss": "https://drive.google.com/file/d/1pzhugMV9dlk7IphJxWcFJuShOVBZjzkl/view?usp=sharing",

}


APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "cache")
ASSETS_DIR = os.path.join(APP_DIR, "assets")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

NUM_CLASSES = 10
VIT_NAME = "vit_tiny_patch16_224"

CLASS_NAMES = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# Your final reported results (from your experiments)
FINAL_RESULTS = pd.DataFrame([
    {"Model": "Teacher CNN (ResNet18)", "Test Accuracy": 0.8407},
    {"Model": "ViT Baseline",           "Test Accuracy": 0.8328},
    {"Model": "ViT + Knowledge Distillation", "Test Accuracy": 0.8564},
])

# -----------------------------
# HELPERS
# -----------------------------
def extract_drive_id(url: str) -> str:
    """
    Accepts a Google Drive share link and extracts file id.
    Works for:
    - https://drive.google.com/file/d/<ID>/view?...
    - https://drive.google.com/uc?id=<ID>
    """
    if "uc?id=" in url:
        return url.split("uc?id=")[-1].split("&")[0].strip()
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return ""
    return m.group(1)

def download_if_needed(name: str, url: str, out_path: str):
    if not url or "PASTE_" in url:
        return False, f"Missing link for {name}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, "already exists"
    file_id = extract_drive_id(url)
    if not file_id:
        return False, f"Could not extract Drive ID for {name}"
    gdown_url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(gdown_url, out_path, quiet=False)
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        return ok, "downloaded" if ok else "download failed"
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_models(teacher_path: str, vit_base_path: str, vit_kd_path: str):
    device = torch.device("cpu")  # Streamlit Cloud is CPU
    torch.set_num_threads(2)

    # Teacher ResNet18
    teacher = models.resnet18(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    # ViT baseline
    vit_base = timm.create_model(VIT_NAME, pretrained=False, num_classes=NUM_CLASSES)
    vit_base.load_state_dict(torch.load(vit_base_path, map_location=device))
    vit_base.eval()

    # ViT KD
    vit_kd = timm.create_model(VIT_NAME, pretrained=False, num_classes=NUM_CLASSES)
    vit_kd.load_state_dict(torch.load(vit_kd_path, map_location=device))
    vit_kd.eval()

    return teacher, vit_base, vit_kd

def get_preprocess():
    # Use ImageNet normalization (standard for both ResNet + ViT from timm)
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def predict(model, pil_img: Image.Image):
    tfm = get_preprocess()
    x = tfm(pil_img.convert("RGB")).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]
    return pred, conf, top3, probs

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Galaxy10 ViT + KD Demo", page_icon="🌌", layout="wide")

st.title("🌌 Galaxy Morphology Classification — Teacher CNN vs ViT vs ViT+KD")
st.caption("Interactive demo for FYP: Improving Vision Transformer performance on small dataset using Knowledge Distillation from a CNN teacher.")

page = st.sidebar.radio("Navigation", ["Predict", "Results", "About"])

# -----------------------------
# Ensure files downloaded
# -----------------------------
st.sidebar.markdown("### App Setup")
teacher_file = os.path.join(CACHE_DIR, "teacher_resnet18_final.pt")
vit_base_file = os.path.join(CACHE_DIR, "vit_tiny_patch16_224_baseline_best.pt")
vit_kd_file = os.path.join(CACHE_DIR, "vit_tiny_patch16_224_KD_best.pt")

needed = [
    ("teacher", DRIVE_FILES["teacher"], teacher_file),
    ("vit_baseline", DRIVE_FILES["vit_baseline"], vit_base_file),
    ("vit_kd", DRIVE_FILES["vit_kd"], vit_kd_file),
]

status_lines = []
all_ok = True
for name, url, path in needed:
    ok, msg = download_if_needed(name, url, path)
    status_lines.append(f"- **{name}**: {msg}")
    all_ok = all_ok and ok

st.sidebar.write("\n".join(status_lines))

if not all_ok:
    st.error("Some model files are missing. Please paste correct Google Drive share links in app.py (DRIVE_FILES).")
    st.stop()

teacher, vit_base, vit_kd = load_models(teacher_file, vit_base_file, vit_kd_file)

# -----------------------------
# PREDICT PAGE
# -----------------------------
if page == "Predict":
    st.subheader("🔎 Predict Galaxy Class")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Upload a galaxy image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        model_choice = st.selectbox("Choose model", ["Teacher CNN (ResNet18)", "ViT Baseline", "ViT + KD"])
        run_btn = st.button("Predict")

    with col2:
        st.markdown("### Classes")
        for i, n in enumerate(CLASS_NAMES):
            st.write(f"**{i}** — {n}")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input image", use_container_width=True)

        if run_btn:
            if model_choice.startswith("Teacher"):
                model = teacher
            elif model_choice.startswith("ViT Baseline"):
                model = vit_base
            else:
                model = vit_kd

            pred, conf, top3, probs = predict(model, img)

            st.success(f"**Prediction:** {CLASS_NAMES[pred]}  \n**Confidence:** {conf*100:.2f}%")

            # Top-3 table
            top3_df = pd.DataFrame(
                [{"Rank": r+1, "Class": CLASS_NAMES[c], "Probability": p} for r, (c, p) in enumerate(top3)]
            )
            st.dataframe(top3_df, use_container_width=True)

            # Probability bar chart
            chart_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
            st.bar_chart(chart_df.set_index("Class"))

    else:
        st.info("Upload an image to start prediction.")

# -----------------------------
# RESULTS PAGE
# -----------------------------
elif page == "Results":
    st.subheader("📊 Experimental Results Summary")

    st.markdown("### Final Comparison (Test Accuracy)")
    st.dataframe(FINAL_RESULTS, use_container_width=True)

    st.markdown("### Key Findings")
    st.write("- ViT baseline performs close to teacher on this dataset.")
    st.write("- Knowledge Distillation improves ViT generalisation (KD > baseline).")
    st.write("- Final reported accuracies are from full test evaluation; curves are for training behaviour visualisation.")

    # Optional: show images if you provide links
    def show_asset(name, url, filename):
        if not url or "PASTE_" in url:
            return
        out_path = os.path.join(ASSETS_DIR, filename)
        ok, _ = download_if_needed(name, url, out_path)
        if ok:
            st.image(out_path, caption=filename, use_container_width=True)

    st.markdown("### Confusion Matrices")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_asset("cm_teacher", DRIVE_FILES["cm_teacher"], "confusion_teacher.png")
    with c2:
        show_asset("cm_vit", DRIVE_FILES["cm_vit"], "confusion_vit.png")
    with c3:
        show_asset("cm_kd", DRIVE_FILES["cm_kd"], "confusion_kd.png")

    st.markdown("### KD Training Curves")
    c4, c5 = st.columns(2)
    with c4:
        show_asset("kd_curve_acc", DRIVE_FILES["kd_curve_acc"], "kd_curve_accuracy.png")
    with c5:
        show_asset("kd_curve_loss", DRIVE_FILES["kd_curve_loss"], "kd_curve_loss.png")

# -----------------------------
# ABOUT PAGE
# -----------------------------
else:
    st.subheader("ℹ️ About This Project")
    st.write("""
**Project Title:** Improving Vision Transformer (ViT) performance on a small dataset using Knowledge Distillation (KD) from a CNN teacher.

**Dataset:** Galaxy10 DECaLS (17,736 images, 10 morphology classes, 256×256 RGB).

**Motivation:** ViTs usually need large datasets. On small datasets they may overfit or learn weaker features than CNNs.

**Approach:**
1) Train a strong CNN teacher (ResNet18).
2) Train a ViT baseline (no KD).
3) Train ViT with KD (teacher guides student).

**Knowledge Distillation (KD):**
Student learns from ground-truth labels AND teacher soft targets.

Loss:
L = (1 − α) * CE(y, s) + α * T² * KL( softmax(t/T) || softmax(s/T) )

Where:
- CE = cross entropy with true labels
- KL = KL divergence between teacher and student probabilities
- T = temperature, α = mixing weight
""")
