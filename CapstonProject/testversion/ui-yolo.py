import streamlit as st
import torch
import numpy as np
import requests
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO  # YOLOv10/v11
import google.generativeai as genai
import pandas as pd
import warnings

# --- 0. Warning Control ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_all_models():
    # YOLOv10x
    yolo = YOLO('yolov10x.pt').to(device) 
    
    g_model = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            g_model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        g_model = None
    return yolo, g_model

yolo_model, gemini_model = load_all_models()

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
st.set_page_config(page_title="AI Object Explorer (YOLOv10)", layout="wide")
st.title("AI Multi-Object Detection (YOLOv10 Power)")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    target_items_raw = st.text_input("Items to find (Target)", "laptop")
    
    st.divider()
    st.subheader("Model Parameter")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.05, 
        max_value=1.0, 
        value=0.25, # YOLOv10's default is often around 0.25
        step=0.05,
        help="It's recommended to keep this between 0.2 and 0.5 for best results with YOLOv10. Adjust based on your needs."
    )
    
    st.info("Sequence: Gemini → HuggingFace+Groq → Groq Only")
    process_btn = st.button("Run Analysis", type="primary")

# --- 3. Main Analysis Logic ---
if process_btn:
    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.getvalue()
            original_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            targets = [t.strip().lower() for t in target_items_raw.split(",") if t.strip()]

            # [Step 1] YOLOv10 Object Detection
            results = yolo_model.predict(original_image, conf=conf_threshold, device=device)[0]
            
            # [Step 2] Visualization & Classification
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(original_image)
            results_data = []

            # YOLOv10 outputs: boxes (xyxy), scores, class_ids, and names dict
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            names = results.names  

            cmap = plt.get_cmap('tab10')

            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                color_rgb = cmap(i % 10)[:3]
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_rgb)
                xmin, ymin, xmax, ymax = box
                
                label = names[cls_id]
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
                st.subheader(f"YOLOv10 Visualization (Count: {len(results_data)})")
                st.pyplot(fig)
            
            with rep_col:
                st.subheader("Intelligent Report")
                detected_names = [names[c] for c in class_ids]
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
                            print("HF + Groq prompt:", refined_prompt)
                            groq_resp = get_groq_response(st.secrets["GROQ_API_KEY"], refined_prompt)
                            st.info(f"**[HF + Groq]**\n\n{groq_resp}")
                            success = True
                    except:
                        st.warning("HF combination failed...")

                if not success and "GROQ_API_KEY" in st.secrets:
                    try:
                        prompt = f"Detected: {obj_summary}. Analyze if {target_items_raw} is present."
                        print("groq prompt:", prompt)
                        groq_resp = get_groq_response(st.secrets["GROQ_API_KEY"], prompt)
                        st.info(f"**[Groq Solo]**\n\n{groq_resp}")
                        success = True
                    except:
                        pass

                if not success:
                    st.error("Report generation failed.")

            # --- 5. Data Table ---
            st.divider()
            st.subheader(f"Full Object List")
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