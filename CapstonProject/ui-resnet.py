import streamlit as st
import torch
import numpy as np
import requests
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
import pandas as pd
from torchvision.transforms import functional as F
import warnings

# --- 0. Warning Control ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# DETR Default COCO Classes (91 classes)
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

@st.cache_resource
def load_all_models():
    # Load DETR with ResNet-50 backbone from TorchHub
    resnet_detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
    resnet_detr.eval()
    
    g_model = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            g_model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        g_model = None
    return resnet_detr, g_model

detr_model, gemini_model = load_all_models()

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(1)
    return torch.stack([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)], dim=1)

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

# --- 2. UI Layout ---
st.set_page_config(page_title="AI Object Explorer (ResNet-50)", layout="wide")
st.title("AI Multi-Object Detection (DETR ResNet-50)")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    target_items_raw = st.text_input("Items to find (Target)", "laptop")
    
    st.divider()
    st.subheader("Model Parameter")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="DETR works best with a higher threshold (0.7+)."
    )
    
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

            # [Step 1] Image Preprocessing for ResNet/DETR
            img_tensor = F.to_tensor(original_image).unsqueeze(0).to(device)
            
            # [Step 2] DETR ResNet-50 Inference
            with torch.no_grad():
                outputs = detr_model(img_tensor)
            
            # Post-processing: extract probabilities and boxes
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > conf_threshold
            
            # Convert boxes from [0; 1] format to pixel coordinates
            boxes_scaled = box_cxcywh_to_xyxy(outputs['pred_boxes'][0, keep]) * torch.tensor([w, h, w, h], device=device)
            scores = probas[keep]

            # [Step 3] Visualization & Data Prep
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(original_image)
            results_data = []

            colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))

            for i, (prob, box) in enumerate(zip(scores, boxes_scaled)):
                color_rgb = colors[i % 10][:3]
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_rgb)
                xmin, ymin, xmax, ymax = box.tolist()
                
                cl = prob.argmax()
                label = COCO_CLASSES[cl]
                score = prob[cl]
                
                is_match = any(t in label.lower() for t in targets)

                # Drawing Bounding Box
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color_rgb, linewidth=3))
                ax.text(xmin, ymin - 10, f"{label} {score:.1%}", 
                        fontsize=10, color='white', fontweight='bold', 
                        bbox=dict(facecolor=color_rgb, alpha=0.8, edgecolor='none'))

                results_data.append({
                    "Object ID": f"Obj {i+1}",
                    "Color Indicator": hex_color,
                    "Detected Item": label,
                    "Confidence": f"{score:.1%}",
                    "Is Target?": "✅ YES" if is_match else "-"
                })

            ax.axis('off')
            plt.tight_layout()

            # --- 4. Display & Report ---
            res_col, rep_col = st.columns([1.2, 0.8])
            with res_col:
                st.subheader(f"ResNet-50 Visualization  (Threshold: {conf_threshold})")
                st.pyplot(fig)
            
            with rep_col:
                st.subheader("Intelligent Report")
                detected_names = [d["Detected Item"] for d in results_data]
                obj_summary = ", ".join(detected_names) if detected_names else "No objects detected"
                success = False
                
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
            st.subheader(f"Full Object List ({len(results_data)} items found)")
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