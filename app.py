"""CerviScope AI — polished local MOEA-SSL research dashboard.

Launch in your project folder:
    python -m streamlit run app_moea_ssl_pro_fixed.py

This uses the original trained MobileNetV3-Small checkpoint and does not retrain it.
Research only: not for patient care, screening, or clinical diagnosis.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parent
PRIMARY_MODEL = ROOT / "results" / "pareto_160_10epochs" / "best_model.pt"
FALLBACK_MODEL = ROOT / "best_model.pt"
MODEL_PATH = PRIMARY_MODEL if PRIMARY_MODEL.is_file() else FALLBACK_MODEL
LABELS = ("Normal", "Abnormal")
MAX_IMAGES = 10
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

st.set_page_config(
    page_title="CerviScope AI | MOEA-SSL",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use st.html for custom HTML. st.markdown can interpret indented HTML as code!
st.html("""
<style>
:root { --ink:#14283f; --navy:#10233f; --teal:#0c807c; --muted:#64758b; }
[data-testid="stAppViewContainer"] { background:#f5f8fc; }
.block-container { max-width:1330px; padding-top:1.5rem; }
[data-testid="stSidebar"] { background:#10233f; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color:#eef5ff; }
.cs-hero {
    background:linear-gradient(115deg,#122542 0%,#173e5c 63%,#087e82 130%);
    color:white; padding:30px 36px; border-radius:20px;
    box-shadow:0 12px 30px rgba(18,37,66,.12);
    margin-bottom:18px;
}
.cs-kicker { color:#8be6d8; font-size:11px; font-weight:800;
    letter-spacing:.16em; text-transform:uppercase; }
.cs-hero h1 { font-size:clamp(28px,3.7vw,44px); letter-spacing:-.04em;
    line-height:1.14; font-weight:800; margin:11px 0; color:#fff; }
.cs-hero p { color:#d9e6f1; font-size:15px; line-height:1.65;
    max-width:900px; margin:0 0 16px; }
.cs-tag { display:inline-block; color:#edfaff; border:1px solid #62869e;
    background:rgba(255,255,255,.06); border-radius:30px;
    padding:5px 12px; margin:0 6px 6px 0; font-size:12px; }
.cs-alert { background:#fff8e9; color:#72521c; border:1px solid #f0dbaa;
    padding:13px 16px; border-radius:13px; line-height:1.55; margin:14px 0; }
.cs-section { color:#077e83; text-transform:uppercase; letter-spacing:.14em;
    font-size:11px; font-weight:800; margin:14px 0 5px; }
.cs-result { background:#fff; border:1px solid #e0e7ef;
    border-left:6px solid #0e8982; border-radius:15px; padding:20px 22px; }
.cs-result.abnormal { border-left-color:#c57b37; }
.cs-small { color:#64748b; font-size:12px; font-weight:750;
    text-transform:uppercase; letter-spacing:.08em; }
.cs-large { font-size:32px; font-weight:800; color:#182c46; margin:7px 0; }
.cs-note { color:#5d7285; font-size:13px; line-height:1.6; }
.cs-insight { background:#fff; border:1px solid #e0e7ef; border-radius:15px;
    padding:18px 20px; margin:9px 0 16px; }
.cs-insight strong { color:#153d55; }
.cs-insight p { font-size:13px; line-height:1.7; color:#52677d; margin:8px 0 0; }
[data-testid="stMetric"] { border:1px solid #e0e7ef; background:white;
    padding:12px 15px; border-radius:14px; }
[data-testid="stFileUploader"] section { background:#fff;
    border-radius:12px; border-color:#b4c8d9; }
</style>
""")


def safe(text: object) -> str:
    return html.escape(str(text), quote=True)


@st.cache_resource(show_spinner="Loading your trained research checkpoint...")
def load_model(path: str):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != "mobilenet_v3_small":
        raise ValueError("The checkpoint is not MobileNetV3-Small.")
    if saved.get("class_names") != ["normal", "abnormal"]:
        raise ValueError("Unexpected class order. Predictions cannot be trusted.")
    size = int(saved["input_size"])
    if size not in (96, 128, 160, 192, 224):
        raise ValueError(f"Unsupported checkpoint image size: {size}")

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    preprocessing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, preprocessing, saved, size


def classify(model, preprocessing, image):
    tensor = preprocessing(image).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(tensor)
        scores = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    return int(np.argmax(scores)), scores


def image_checks(image):
    gray = np.asarray(image.convert("L").resize((160, 160)), dtype=np.float32)
    bright = float(gray.mean())
    contrast = float(gray.std())
    warnings = []
    if min(image.size) < 96:
        warnings.append("Small source image; resizing may obscure subtle detail.")
    if bright < 48:
        warnings.append("The overall image is relatively dark.")
    elif bright > 207:
        warnings.append("The overall image is relatively bright.")
    if contrast < 18:
        warnings.append("Overall tonal contrast is relatively low.")
    if not warnings:
        warnings.append("No extreme size, brightness or contrast flags were detected.")
    return bright, contrast, warnings


def classification_comment(predicted, scores):
    if predicted == 1:
        message = (
            "The model assigns the higher score to the Abnormal class. "
            "In the Herlev dataset this combines several abnormal cell categories. "
            "It does not establish lesion grade, cancer stage, or a patient diagnosis."
        )
    else:
        message = (
            "The model assigns the higher score to the Normal class. "
            "This is not proof of disease absence; false-negative predictions "
            "remain possible."
        )
    gap = abs(scores[0] - scores[1])
    if gap >= .80:
        separation = "The two output scores are widely separated."
    elif gap >= .40:
        separation = "The two output scores are moderately separated."
    else:
        separation = "The two output scores are relatively close."
    separation += " This is not calibrated clinical confidence."
    return message, separation


def make_gradcam(model, preprocessing, image, category):
    """Visualise model gradients, not lesions or verified medical features."""
    layer = model.features[-1]
    captured = {}

    def capture(_layer, _inputs, output):
        output.retain_grad()
        captured["activations"] = output

    handle = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            tensor = preprocessing(image).unsqueeze(0)
            output = model(tensor)
            output[0, category].backward()
            acts = captured["activations"]
            if acts.grad is None:
                return None
            weights = acts.grad.mean(dim=(2, 3), keepdim=True)
            heat = (weights * acts).sum(dim=1)[0].relu().detach().cpu().numpy()
        heat -= float(heat.min())
        high = float(heat.max())
        if high <= 1e-9:
            return None
        heat = (255 * heat / high).astype(np.uint8)
        resized = Image.fromarray(heat).resize(image.size, Image.Resampling.BICUBIC)
        alpha = np.asarray(resized, dtype=np.float32)[..., None] / 255.0 * .53
        raw = np.asarray(image.convert("RGB"), dtype=np.float32)
        red = np.zeros_like(raw)
        red[..., 0] = 255
        red[..., 1] = 72
        red[..., 2] = 34
        overlay = np.clip(raw * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def report(filename, prediction, scores, image, notes, epoch, size):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    comment, separation = classification_comment(prediction, scores)
    return (
        "CERVISCOPE AI — MOEA-SSL RESEARCH REPORT\n"
        "NOT A CLINICAL REPORT OR DIAGNOSIS\n\n"
        f"Generated: {timestamp}\n"
        f"Image: {filename}\n"
        f"Original pixels: {image.width} x {image.height}\n"
        f"Model: MobileNetV3-Small; input {size}x{size}; saved epoch {epoch}\n"
        f"Predicted image class: {LABELS[prediction]}\n"
        f"Normal model score: {scores[0]:.2%}\n"
        f"Abnormal model score: {scores[1]:.2%}\n\n"
        f"Interpretation: {comment}\n"
        f"Score interpretation: {separation}\n"
        f"Image-level observations: {' '.join(notes)}\n\n"
        "The output describes a research model's prediction for a cropped cell image.\n"
        "The software does not measure nuclear morphology or diagnose cancer.\n"
        "Softmax scores are uncalibrated; results require independent evaluation.\n"
        "Use only de-identified research data.\n"
    )


# Load saved checkpoint; make failures actionable.
if not MODEL_PATH.is_file():
    st.error("Could not find your model checkpoint.")
    st.code(str(PRIMARY_MODEL))
    st.info("Keep the saved best_model.pt in results/pareto_160_10epochs/.")
    st.stop()

try:
    model, preprocessing, checkpoint, size = load_model(str(MODEL_PATH.resolve()))
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

best_epoch = checkpoint.get("best_epoch", "unknown")
metrics = checkpoint.get("validation_metrics") or {}

with st.sidebar:
    st.header("🔬 CerviScope AI")
    st.caption("MOEA-SSL Research Console")
    st.divider()
    st.subheader("Analysis options")
    show_attention = st.toggle("Show Grad-CAM attention", value=False)
    show_details = st.toggle("Show image and model details", value=True)
    st.divider()
    st.subheader("Active model")
    st.write("**MobileNetV3-Small**")
    st.write(f"Input: **{size} × {size}**")
    st.write(f"Checkpoint: **Epoch {best_epoch}**")
    st.write("Inference device: **CPU**")
    st.success("Model loaded")
    st.divider()
    st.caption("Bindura University of Science Education")
    st.caption("MSc Data Science • Research demonstration")
    st.warning("Not for diagnosis or clinical screening")

st.html("""
<div class="cs-hero">
  <div class="cs-kicker">MOEA-SSL / CERVICAL CELL RESEARCH STUDIO</div>
  <h1>🔬 CerviScope AI</h1>
  <p>Explore how a multi-objective evolutionary, semi-supervised learning model
  classifies individual cervical-cell microscopy images. Review model scores,
  image-level observations and optional model-attention visualisations.</p>
  <span class="cs-tag">Model ready</span>
  <span class="cs-tag">MobileNetV3-Small</span>
  <span class="cs-tag">FixMatch-style SSL</span>
  <span class="cs-tag">Research prototype</span>
</div>
""")

cols = st.columns(4)
for col, title, key, note in (
    (cols[0], "Validation accuracy", "accuracy", "Development data"),
    (cols[1], "Sensitivity", "sensitivity", "Abnormal class recall"),
    (cols[2], "Specificity", "specificity", "Normal class recall"),
    (cols[3], "F1-score", "f1", "Development data"),
):
    value = metrics.get(key)
    with col:
        st.metric(title, f"{value:.1%}" if value is not None else "N/A", help=note)

st.html("""
<div class="cs-alert"><strong>Research demonstration only.</strong>
The classifier was developed on individual Herlev-style cervical-cell images.
Its scores are not calibrated probabilities and must not be used for screening,
patient diagnosis or treatment decisions. Validation metrics are not independent
clinical validation.</div>
""")

st.html('<div class="cs-section">01 / IMAGE INPUT</div>')
st.subheader("Upload cervical-cell microscopy images")
st.caption("BMP, PNG, JPG or TIFF • up to 10 images • 15 MB per image. Use de-identified, cropped single-cell research images.")
uploads = st.file_uploader(
    "Choose images", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Choose a cropped cervical-cell image above to see a classification.")
    with st.expander("What exactly does the model predict?"):
        st.write("The trained classifier distinguishes **Normal** from **Abnormal** images. ")
        st.write("It does **not** identify disease grade, cancer stage, or measure specific nuclear characteristics.")
    st.stop()

if len(uploads) > MAX_IMAGES:
    st.warning(f"Only the first {MAX_IMAGES} images will be processed.")

st.html('<div class="cs-section">02 / IMAGE ANALYSIS</div>')
export_rows = []
for index, upload in enumerate(uploads[:MAX_IMAGES], start=1):
    filename = Path(upload.name).name
    try:
        if upload.size > MAX_UPLOAD_BYTES:
            raise ValueError("Image is larger than the 15 MB limit.")
        with Image.open(io.BytesIO(upload.getvalue())) as opened:
            if opened.width * opened.height > 30_000_000:
                raise ValueError("Image dimensions exceed the 30-million-pixel limit.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if min(image.size) < 8:
            raise ValueError("Image dimensions are too small.")
        predicted, scores = classify(model, preprocessing, image)
        brightness, contrast, notes = image_checks(image)
        interpretation, separation = classification_comment(predicted, scores)
    except (ValueError, RuntimeError, OSError, UnidentifiedImageError) as exc:
        st.error(f"Cannot process {filename}: {exc}")
        continue

    st.subheader(f"Image {index:02d} — {filename}")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(image, caption=f"Uploaded image • {image.width} × {image.height} pixels", use_container_width=True)
        if show_attention:
            with st.spinner("Generating Grad-CAM..."):
                try:
                    overlay = make_gradcam(model, preprocessing, image, predicted)
                except (RuntimeError, ValueError, IndexError) as exc:
                    overlay = None
                    st.warning(f"Attention map unavailable: {exc}")
            if overlay is not None:
                st.image(overlay, caption="Grad-CAM: exploratory model attention", use_container_width=True)
                st.caption("Red regions contribute more to this prediction; they are not verified lesions.")
            else:
                st.caption("Grad-CAM was not informative for this image.")

    with right:
        css_class = "abnormal" if predicted == 1 else "normal"
        st.html(
            f'<div class="cs-result {css_class}">'
            '<div class="cs-small">Predicted image category</div>'
            f'<div class="cs-large">{safe(LABELS[predicted])} cell</div>'
            '<div class="cs-note">Output of the trained MOEA-SSL research model.</div>'
            '</div>'
        )
        st.write("**Normal model score**")
        st.progress(float(scores[0]), text=f"{scores[0]:.1%}")
        st.write("**Abnormal model score**")
        st.progress(float(scores[1]), text=f"{scores[1]:.1%}")
        st.caption("Softmax scores are not calibrated clinical probabilities.")
        st.info(separation)

    st.html('<div class="cs-section">03 / RESEARCH COMMENTARY</div>')
    a, b = st.columns(2, gap="large")
    with a:
        st.html(
            '<div class="cs-insight"><strong>🧠 Model interpretation</strong>'
            f'<p>{safe(interpretation)}</p></div>'
        )
    with b:
        st.html(
            '<div class="cs-insight"><strong>🔍 Image-quality observations</strong>'
            f'<p>{safe(" ".join(notes))} Pixel-statistic checks only; '
            'not medically validated image quality or morphology analysis.</p></div>'
        )

    if show_details:
        with st.expander("View image details and cytology context"):
            x, y, z = st.columns(3)
            x.metric("Width", f"{image.width} px")
            y.metric("Brightness", f"{brightness:.0f}/255")
            z.metric("Tonal contrast", f"{contrast:.1f}")
            st.write("Cytology professionals may inspect the nucleus, cytoplasm, chromatin and nuclear-to-cytoplasmic ratio.")
            st.warning("This prototype does not explicitly measure these features or verify their presence in the uploaded image.")
            st.code("RGB → Resize → ImageNet normalisation → MobileNetV3-Small → Normal / Abnormal")

    report_body = report(filename, predicted, scores, image, notes, best_epoch, size)
    digest = hashlib.sha256(upload.getvalue()).hexdigest()[:10]
    st.download_button(
        "Download individual research report (.txt)",
        data=report_body.encode("utf-8"),
        file_name=f"cerviscope_report_{index}_{digest}.txt",
        mime="text/plain",
        key=f"report_{index}_{digest}",
    )
    export_rows.append({
        "Image": filename,
        "Predicted class": LABELS[predicted],
        "Normal model score (%)": round(scores[0] * 100, 2),
        "Abnormal model score (%)": round(scores[1] * 100, 2),
        "Input width": image.width,
        "Input height": image.height,
        "Model checkpoint epoch": best_epoch,
    })
    st.divider()

if export_rows:
    st.html('<div class="cs-section">04 / EXPORT RESULTS</div>')
    st.subheader("Batch analysis summary")
    st.dataframe(export_rows, hide_index=True, use_container_width=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(export_rows[0]))
    writer.writeheader()
    writer.writerows(export_rows)
    st.download_button("Download all predictions (.csv)",
                       data=buffer.getvalue().encode("utf-8-sig"),
                       file_name="cerviscope_predictions.csv", mime="text/csv")

with st.expander("About the model, evaluation and limitations"):
    st.write("**Architecture:** MobileNetV3-Small; **learning:** FixMatch-style SSL; **optimisation research:** NSGA-II.")
    st.write("The displayed performance comes from validation on development data, not an independent clinical evaluation.")
    st.write("Grad-CAM shows approximate model attention, not lesion boundaries or medically verified cell features.")
    st.write("Only de-identified individual-cell research images should be used.")

st.caption("CerviScope AI • Bindura University of Science Education • MSc research demonstration • Not for clinical use")
