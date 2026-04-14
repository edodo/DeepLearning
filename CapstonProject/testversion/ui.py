import streamlit as st
import torch
import numpy as np
import requests
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import clip
import google.generativeai as genai
import pandas as pd
from torchvision.transforms import functional as F
import warnings

# --- 0. Warning Control ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- API Helper Functions ---
def get_huggingface_caption(api_key, image_bytes):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=15)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        return None
    except:
        return None

def get_groq_response(api_key, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Groq Error: {response.status_code}"
    except Exception as e:
        return f"Groq Connection Failed: {e}"

@st.cache_resource
def load_all_models():
    detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
    detr.eval()
    c_model, c_prep = clip.load("ViT-B/32", device=device)
    g_model = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            g_model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        g_model = None
    return detr, c_model, c_prep, g_model

detr_model, clip_model, clip_preprocess, gemini_model = load_all_models()

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(1)
    return torch.stack([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)], dim=1)

# --- 2. UI Layout ---
st.set_page_config(page_title="AI Object Explorer", layout="wide")
st.title("🎯 AI Multi-Object Detection (Confidence Tuning)")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    target_items_raw = st.text_input("Items to find (Target)", "laptop")
    
    # --- 슬라이더 추가 부분 ---
    st.divider()
    st.subheader("Model Parameter")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="값이 낮을수록 더 많은 객체를 찾지만 오탐지가 늘어납니다."
    )
    # -----------------------
    
    st.info("Sequence: Gemini → HuggingFace+Groq → Groq Only")
    process_btn = st.button("Run Analysis", type="primary")

# --- 3. Main Analysis Logic ---
if process_btn:
    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.getvalue()
            original_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            w, h = original_image.size
            targets = [t.strip().lower() for t in target_items_raw.split(",") if t.strip()]

            # [Step 1] DETR Object Detection
            img_tensor = F.to_tensor(original_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = detr_model(img_tensor)
            
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            
            # --- 슬라이더 값 적용 부분 ---
            keep = probas.max(-1).values > conf_threshold 
            # -----------------------
            
            boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0, keep]) * torch.tensor([w, h, w, h], device=device)
            scores = probas[keep]

            # [Step 2] Visualization & Classification
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(original_image)
            colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(boxes)))
            results_data = []

            for i, (prob, box) in enumerate(zip(scores, boxes)):
                color_rgb = colors[i % 10][:3] 
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_rgb)
                xmin, ymin, xmax, ymax = box.tolist()
                
                detr_idx = prob.argmax()
                original_label = COCO_CLASSES[detr_idx]
                detr_conf = prob[detr_idx]
                is_match = any(t in original_label.lower() for t in targets)

                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color_rgb, linewidth=4))
                ax.text(xmin, ymin - 10, f"ID {i+1}: {original_label} ({detr_conf:.1%})", 
                        fontsize=12, color='white', fontweight='bold', 
                        bbox=dict(facecolor=color_rgb, alpha=0.8, edgecolor='none'))

                results_data.append({
                    "Object ID": f"Obj {i+1}",
                    "Color Indicator": hex_color,
                    "Detected Item": original_label,
                    "Confidence": f"{detr_conf:.1%}",
                    "Is Target?": "✅ YES" if is_match else "-"
                })

            ax.axis('off')
            plt.tight_layout()

            # --- 4. Display & Report Fallback Logic ---
            res_col, rep_col = st.columns([1.2, 0.8])
            with res_col:
                st.subheader(f"Detection Visualization (Threshold: {conf_threshold})")
                st.pyplot(fig)
            
            with rep_col:
                st.subheader("Intelligent Report")
                detected_names = [COCO_CLASSES[p.argmax()] for p in scores]
                obj_summary = ", ".join(detected_names) if detected_names else "No objects detected"
                success = False
                
                # Report Logic (Gemini/HF/Groq)
                if gemini_model:
                    try:
                        prompt = f"Explain the objects detected: {obj_summary}. Focus on {target_items_raw}."
                        gen_resp = gemini_model.generate_content([prompt, original_image])
                        st.info(f"**[Gemini Report]**\n\n{gen_resp.text}")
                        success = True
                    except:
                        st.warning("Gemini failed...")

                if not success and "HF_API_KEY" in st.secrets and "GROQ_API_KEY" in st.secrets:
                    try:
                        hf_caption = get_huggingface_caption(st.secrets["HF_API_KEY"], img_bytes)
                        if hf_caption:
                            refined_prompt = f"Description: '{hf_caption}'. Detected: {obj_summary}. Focus: {target_items_raw}."
                            groq_resp = get_groq_response(st.secrets["GROQ_API_KEY"], refined_prompt)
                            st.info(f"**[HF + Groq]**\n\n{groq_resp}")
                            success = True
                    except:
                        st.warning("HF combination failed...")

                if not success and "GROQ_API_KEY" in st.secrets:
                    try:
                        prompt = f"Detected: {obj_summary}. Analyze if {target_items_raw} is present."
                        groq_resp = get_groq_response(st.secrets["GROQ_API_KEY"], prompt)
                        st.info(f"**[Groq Solo]**\n\n{groq_resp}")
                        success = True
                    except:
                        pass

                if not success:
                    st.error("Report generation failed.")

            # --- 5. Data Table ---
            st.divider()
            st.subheader(f"📊 Object List ({len(results_data)} items found)")
            if results_data:
                df = pd.DataFrame(results_data)
                def apply_color(val):
                    return f'background-color: {val}; color: {val}; border-radius: 5px;'
                try:
                    styled_df = df.style.map(apply_color, subset=['Color Indicator'])
                except AttributeError:
                    styled_df = df.style.applymap(apply_color, subset=['Color Indicator'])
                st.table(styled_df)
            else:
                st.write("No objects found with current threshold.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
    else:
        st.warning("Please upload an image.")