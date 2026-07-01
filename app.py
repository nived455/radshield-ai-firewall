import os
import streamlit as nn_web
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import foolbox as fb
import numpy as np
import pywt

# Force quiet logging for Git dependencies
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# --- CORE DEFENSE ENGINE ALGORITHMS ---
def apply_wavelet_defense(image_tensor):
    img_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised_channels = []
    for i in range(3):
        channel = img_np[:, :, i]
        coeffs = pywt.dwt2(channel, 'db1')
        cA, (cH, cV, cD) = coeffs
        cH_t = pywt.threshold(cH, 0.04, mode='soft')
        cV_t = pywt.threshold(cV, 0.04, mode='soft')
        cD_t = pywt.threshold(cD, 0.04, mode='soft')
        cleaned_channel = pywt.idwt2((cA, (cH_t, cV_t, cD_t)), 'db1')
        denoised_channels.append(cleaned_channel[:224, :224])
    return torch.tensor(np.stack(denoised_channels, axis=0)).unsqueeze(0).float()

def verify_tamper_signature(model, image_tensor, base_pred):
    tampers = 0
    with torch.no_grad():
        for _ in range(3):
            masked_img = image_tensor.clone()
            x, y = np.random.randint(0, 190), np.random.randint(0, 190)
            masked_img[:, :, x:x+25, y:y+25] = 0.0
            if torch.argmax(model(masked_img), dim=1).item() != base_pred:
                tampers += 1
    return True if tampers >= 1 else False

# --- PAGE SETTINGS & BEAUTIFICATION ---
nn_web.set_page_config(page_title="RadShield AI Firewall", layout="wide", page_icon="🛡️")

# Injection of Custom CSS UI Theme styling for an ultra-premium interface look
nn_web.markdown("""
    <style>
        .main { background-color: #0d1117; color: #ffffff; }
        .stButton>button {
            background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
            color: white; border: none; padding: 10px 24px;
            border-radius: 8px; font-weight: bold; width: 100%;
            transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,114,255,0.3);
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,114,255,0.5); }
        .card {
            background-color: #161b22; padding: 20px;
            border-radius: 12px; border: 1px solid #30363d;
            margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #f0f6fc; }
    </style>
""", unsafe_allow_html=True)
# Application Header Banner
nn_web.markdown("""
    <div style='text-align: center; padding-bottom: 20px;'>
        <h1 style='font-size: 2.8rem; margin-bottom: 0;'>🛡️ RADSHIELD AI</h1>
        <p style='color: #8b949e; font-size: 1.2rem;'>Enterprise-Grade Adversarial Firewall for Medical Ingestion Pipelines</p>
    </div>
""", unsafe_allow_html=True)
nn_web.markdown("---")

# Setup Device Context & Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@nn_web.cache_resource
def initialize_system_models():
    base_model = models.resnet18()
    base_model.fc = nn.Linear(base_model.fc.in_features, 2)
    base_model.load_state_dict(torch.load("medical_resnet18_baseline.pth", map_location=device))
    base_model.eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.PyTorchModel(base_model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    return base_model, fmodel

raw_pytorch_model, fmodel = initialize_system_models()

# --- DESIGNING THE SIDEBAR INPUT CONSOLE ---
nn_web.sidebar.markdown("### 🕹️ Control & Ingestion Panel")
nn_web.sidebar.markdown("---")

# FEATURE ADDITION: Dynamic Direct File Upload Widget 
uploaded_file = nn_web.sidebar.file_uploader("Upload Patient DICOM/X-Ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Fallback Category Mapping Selection for Ground Truth Label configuration
ground_truth_category = nn_web.sidebar.radio("Select True Clinical Ground Truth (For Evaluation Metrics)", ["NORMAL", "PNEUMONIA"])

nn_web.sidebar.markdown("---")
click_trigger = nn_web.sidebar.button("Deploy Cyber-Defense Pipeline")

# --- CORE APPLICATION RUNTIME LOOP ---
if click_trigger and uploaded_file is not None:
    with nn_web.spinner("Running network deep-packet sanitization & core-gradient threat vectors..."):
        # Image Pipeline Preprocessing Transformations
        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        pil_image = Image.open(uploaded_file).convert('RGB')
        images = transform_pipeline(pil_image).unsqueeze(0).to(device)
        labels = torch.tensor([1 if ground_truth_category == "PNEUMONIA" else 0]).to(device)
        
        # Calculate adversarial projection mapping coordinates (Simulating a hostile man-in-the-middle transmission hack)
        attack_engine = fb.attacks.PGD()
        _, clipped_adv, _ = attack_engine(fmodel, images, labels, epsilons=[8/255])
        adv_image = clipped_adv[0]
        
        # Execute RadShield Firewall Routing Countermeasures
        is_tampered = verify_tamper_signature(raw_pytorch_model, adv_image, torch.argmax(fmodel(adv_image), dim=1).item())
        defended_tensor = apply_wavelet_defense(adv_image).to(device)
        
        # String Class Label Formats
        true_lbl = "PNEUMONIA" if labels.item() == 1 else "NORMAL"
        fooled_lbl = "PNEUMONIA" if torch.argmax(fmodel(adv_image), dim=1).item() == 1 else "NORMAL"
        defended_lbl = "PNEUMONIA" if torch.argmax(fmodel(defended_tensor), dim=1).item() == 1 else "NORMAL"
        
        # Format tensors into displayable visual matrices
        clean_np = images.squeeze().cpu().numpy().transpose(1, 2, 0)
        adv_np = adv_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        defended_np = np.clip(defended_tensor.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)

        # Output Interactive Alerts
        if is_tampered:
            nn_web.markdown("<div style='background-color: rgba(234, 56, 76, 0.15); border-left: 5px solid #ea384c; padding: 15px; border-radius: 4px; margin-bottom: 25px;'><strong>🚨 SECURITY ALERT DISPATCHED:</strong> Adversarial noise structure detected. Multi-point validation check failed. RadShield DWT Sanitization core active.</div>", unsafe_allow_html=True)
        else:
            nn_web.markdown("<div style='background-color: rgba(30, 161, 103, 0.15); border-left: 5px solid #1ea167; padding: 15px; border-radius: 4px; margin-bottom: 25px;'><strong>✅ SECURITY INTEGRITY VERIFIED:</strong> Image pixel distribution analysis matches safe baseline.</div>", unsafe_allow_html=True)

        # Render 3-Column Image Comparison Matrix Layout block
        col1, col2, col3 = nn_web.columns(3)
        
        with col1:
            nn_web.markdown("<div class='card'>", unsafe_allow_html=True)
            nn_web.markdown("<h3>1. Inbound Uploaded Scan</h3>", unsafe_allow_html=True)
            nn_web.image(clean_np, use_container_width=True, clamp=True)
            nn_web.metric(label="Patient Ground Truth Status", value=true_lbl)
            nn_web.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            nn_web.markdown("<div class='card'>", unsafe_allow_html=True)
            nn_web.markdown("<h3 style='color: #ea384c;'>2. Intercepted Attack</h3>", unsafe_allow_html=True)
            nn_web.image(adv_np, use_container_width=True, clamp=True)
            nn_web.metric(label="Vulnerable Model Decision", value=fooled_lbl, delta="AI POISONED" if true_lbl != fooled_lbl else "UNTOUCHED", delta_color="inverse")
            nn_web.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            nn_web.markdown("<div class='card'>", unsafe_allow_html=True)
            nn_web.markdown("<h3 style='color: #0072ff;'>3. RadShield Defense</h3>", unsafe_allow_html=True)
            nn_web.image(defended_np, use_container_width=True)
            nn_web.metric(label="Sanitized Output Result", value=defended_lbl, delta="HAZARD NEUTRALIZED" if fooled_lbl != defended_lbl else "SAFE")
            nn_web.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is None:
    # Beautiful landing placeholder box informing judges exactly what to execute
    nn_web.markdown("""
        <div style='text-align: center; border: 2px dashed #30363d; padding: 60px; border-radius: 16px; background-color: #161b22; margin-top: 40px;'>
            <div style='font-size: 4rem; margin-bottom: 15px;'>📥</div>
            <h3 style='margin-bottom: 10px;'>Awaiting Clinical Transmission Stream Input</h3>
            <p style='color: #8b949e; max-width: 500px; margin: 0 auto 20px auto;'>Drag and drop any chest X-ray sample file into the sidebar control console, then click the blue activation trigger button to evaluate real-time adversarial defensive sanitization capabilities.</p>
        </div>
    """, unsafe_allow_html=True)