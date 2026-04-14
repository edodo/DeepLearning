import streamlit as st
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import google.generativeai as genai
import pandas as pd
import warnings

# --- 0. Warning Control ---
warnings.filterwarnings("ignore")

# --- 1. Model Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# 21 Classes for COCO Segmentation (Based on PASCAL VOC)
SEG_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

@st.cache_resource
def load_all_models():
    # Load DeepLabV3 which has excellent pixel analysis performance (UNet Style)
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
    seg_model.eval()
    
    g_model = None
    if "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            # Default model name setup to prevent 404 errors
            g_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.sidebar.error(f"Gemini Initialization Error: {e}")
    
    return seg_model, g_model

seg_model, gemini_model = load_all_models()

# --- API Helper Functions ---
def get_groq_response(api_key, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful AI that analyzes image segmentation results."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Groq Error: {response.status_code}"
    except Exception as e:
        return f"Groq Connection Failed: {e}"

# --- 2. UI Layout ---
st.set_page_config(page_title="AI Pixel Explorer", layout="wide")
st.title("🎭 AI Pixel-wise Segmentation (UNet Style)")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    target_items_raw = st.text_input("Items to find (Target)", "person, car")
    
    st.divider()
    st.info("💡 **Workflow**\n1. Pixel-wise Analysis (DeepLabV3)\n2. Attempt Gemini Analysis\n3. Auto-fallback to Groq if Gemini fails")
    process_btn = st.button("Run Pixel Analysis", type="primary")

# --- 3. Main Analysis Logic ---
if process_btn and uploaded_file:
    try:
        # Load Image
        img_bytes = uploaded_file.getvalue()
        input_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Preprocessing (DeepLabV3 Standard)
        preprocess = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)

        # [Step 1] Segmentation Inference
        with st.spinner("Analyzing pixels..."):
            with torch.no_grad():
                output = seg_model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).cpu().numpy()

        # [Step 2] Visualization and Data Preparation
        found_classes = np.unique(output_predictions)
        results_data = []
        mask_image = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
        cmap = plt.get_cmap('tab20')

        for idx, cls_id in enumerate(found_classes):
            if cls_id == 0: continue # Skip Background
            
            cls_name = SEG_CLASSES[cls_id]
            color = np.array(cmap(cls_id % 20)[:3]) * 255
            mask_image[output_predictions == cls_id] = color.astype(np.uint8)
            
            # Check Target Match
            is_target = any(t.strip().lower() in cls_name.lower() for t in target_items_raw.split(","))
            
            results_data.append({
                "Area ID": f"Region {idx}",
                "Color Indicator": '#%02x%02x%02x' % tuple(color.astype(int)),
                "Detected Class": cls_name,
                "Is Target?": "✅ YES" if is_target else "-"
            })

        # Display Layout
        col1, col2 = st.columns([1.2, 0.8])
        
        with col1:
            st.subheader("🎯 Segmentation Results")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(input_image)
            ax1.set_title("Original")
            ax1.axis('off')
            ax2.imshow(mask_image)
            ax2.set_title("Mask (UNet Style)")
            ax2.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("📝 Intelligent Report")
            summary = ", ".join([d['Detected Class'] for d in results_data]) if results_data else "No specific objects"
            
            success_report = False
            
            # [1] Attempt Gemini Call (With Error Handling)
            if gemini_model:
                try:
                    with st.spinner("Gemini is analyzing the image..."):
                        prompt = f"The segmentation model found: {summary}. Focus on {target_items_raw}. Please describe the scene."
                        gen_resp = gemini_model.generate_content([prompt, input_image])
                        st.info(f"**[Gemini Report]**\n\n{gen_resp.text}")
                        success_report = True
                except Exception as e:
                    st.warning(f"Gemini Call Failed (404 or Unsupported Model). Switching to Groq.")
                    success_report = False

            # [2] Auto-fallback to Groq if Gemini fails
            if not success_report:
                if "GROQ_API_KEY" in st.secrets:
                    try:
                        with st.spinner("Groq is performing text-based analysis..."):
                            prompt = f"Analyze these detected areas: {summary}. The user is specifically interested in: {target_items_raw}. Summarize the content."
                            resp = get_groq_response(st.secrets["GROQ_API_KEY"], prompt)
                            st.success(f"**[Groq Report]**\n\n{resp}")
                            success_report = True
                    except Exception as e:
                        st.error(f"Groq Call Failed: {e}")
                else:
                    st.error("No API keys configured for analysis.")

        # --- 5. Data Table ---
        st.divider()
        st.subheader("📊 Area Analysis Table")
        if results_data:
            df = pd.DataFrame(results_data)
            def apply_color(val):
                return f'background-color: {val}; color: {val};'
            st.table(df.style.map(apply_color, subset=['Color Indicator']))
        else:
            st.info("No objects detected besides the background.")

    except Exception as e:
        st.error(f"Error during analysis: {e}")
else:
    st.info("Please upload an image and click the 'Run Pixel Analysis' button.")