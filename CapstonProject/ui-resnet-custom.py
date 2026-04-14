import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
import pandas as pd
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings

# --- 0. Warning Control ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "detr_edodo_final.pth"

# 92 classes: 91 from COCO + 1 custom class "edodo" for Edward
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 
    'edodo' # Custom class for "edward"
]

@st.cache_resource
def load_all_models():
    # Faster R-CNN (classification + detection) 
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 92  # 91(COCO) + 1(edodo)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            st.sidebar.success(f"V Loaded: {MODEL_PATH}")
        except Exception as e:
            st.sidebar.error(f"X Weights load failed: {e}")
    
    model.to(device)
    model.eval()
    
    g_model = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            g_model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        g_model = None
    return model, g_model

detr_model, gemini_model = load_all_models()

# --- 2. UI Layout ---
st.set_page_config(page_title="Edodo AI Explorer", layout="wide")
st.title("Custom AI Multi-Object Detection")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    target_items_raw = st.text_input("Items to find (Target)", "edodo, laptop")
    
    st.divider()
    st.subheader("Model Parameter")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    process_btn = st.button("Run Analysis", type="primary")

# --- 3. Main Analysis Logic ---
if process_btn and uploaded_file:
    try:
        img_bytes = uploaded_file.getvalue()
        original_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = original_image.size
        targets = [t.strip().lower() for t in target_items_raw.split(",") if t.strip()]

        # [Step 1] Inference
        img_tensor = F.to_tensor(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = detr_model(img_tensor)[0]
        
        # [Step 2] Filtering
        scores = prediction['scores']
        keep = scores >= conf_threshold
        filtered_scores = scores[keep]
        filtered_boxes = prediction['boxes'][keep]
        filtered_labels = prediction['labels'][keep]

        # [Step 3] Visualization
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(original_image)
        results_data = []
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))

        for i, (score, box, label_idx) in enumerate(zip(filtered_scores, filtered_boxes, filtered_labels)):
            color_rgb = colors[i % 10][:3]
            hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_rgb)
            xmin, ymin, xmax, ymax = box.tolist()
            
            # Get label name and check if it matches any target
            label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"ID {label_idx}"
            is_match = any(t in label_name.lower() for t in targets)

            # Draw
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color_rgb, linewidth=3))
            ax.text(xmin, ymin - 10, f"{label_name} {score:.1%}", 
                    fontsize=12, color='white', fontweight='bold', 
                    bbox=dict(facecolor=color_rgb, alpha=0.8, edgecolor='none'))

            results_data.append({
                "Object ID": f"Obj {i+1}",
                "Color Indicator": hex_color,
                "Detected Item": label_name,
                "Confidence": f"{score:.1%}",
                "Is Target?": "✅ YES" if is_match else "-"
            })

        ax.axis('off')
        plt.tight_layout()

        # --- 4. Display Results ---
        res_col, rep_col = st.columns([1.2, 0.8])
        with res_col:
            st.subheader("Detection Result")
            st.pyplot(fig)
        
        with rep_col:
            st.subheader("Intelligent Report")
            detected_summary = ", ".join([d["Detected Item"] for d in results_data]) if results_data else "No objects"
            
            if gemini_model:
                try:
                    # Configure the prompt to add an explanation about Jonathan Edwards if 'edodo' is present.
                    custom_context = "If 'edodo' is detected, it refers to Jonathan Edwards." if "edodo" in detected_summary.lower() else ""
                    prompt = f"The model detected: {detected_summary}. {custom_context} Please provide a brief explanation of these items."
                    gen_resp = gemini_model.generate_content([prompt, original_image])
                    st.info(gen_resp.text)
                except:
                    st.write(f"Detected: {detected_summary}")
            else:
                st.write(f"Detected: {detected_summary}")

        # --- 5. Data Table ---
        st.divider()
        st.subheader(f"Object Analysis Table ({len(results_data)} items)")
        if results_data:
            df = pd.DataFrame(results_data)
            def apply_color(val): return f'background-color: {val}; color: {val};'
            st.table(df.style.applymap(apply_color, subset=['Color Indicator']))

    except Exception as e:
        st.error(f"Error: {e}")