# Intelligent Object Detection & Image Search (DETR Resnet50)

This project features an advanced object detection and intelligent reporting system built with the **DETR (Detection Transformer)** model and a **Resnet50** backbone. It is designed to recognize standard COCO objects while attempting to identify custom-trained items through an interactive Streamlit interface.

## Key Features
* **Object Detection:** High-precision real-time detection using Facebook's DETR-Resnet50.
* **Custom Class Fine-tuning:** Implemented a pipeline to detect the user-defined class **'edodo'** (referring to Jonathan Edwards).
* **Multi-LLM Strategy (Intelligent Reporting):**
    * **Primary:** Uses **Google Gemini Pro Vision** for multimodal analysis (image + text).
    * **Secondary (Hugging Face):** Integrates vision-language models for assisted image captioning.
    * **Fallback (Groq):** Automatically switches to the **Groq API** (Llama 3/Mixtral) to ensure continuous report generation when Gemini tokens are exhausted.
* **Interactive Analysis:** Provides visual bounding boxes and a detailed data table with confidence scores.

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision, Hugging Face Transformers
* **Frontend:** Streamlit
* **LLM APIs:** Google Generative AI (Gemini), Groq, Hugging Face
* **Visualization:** Matplotlib, Pandas, NumPy

## Project Structure
* `ui-resnet-custom.py`: Main application script with custom class detection and multi-API reporting.
* `ui-resnet.py`: Standard detection UI implementation.
* `detr-resnet-demo.ipynb`: Jupyter notebook covering the training process, data preprocessing, and testing.
* `detr_edodo_final.pth`: Trained weight file for the 'edodo' custom class.

## Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/edodo/DeepLearning.git
    cd DeepLearning
    ```

2.  **Usage - Object detection**
There are no extra compiled components in DETR and package dependencies are minimal, so the code is very simple to use. We provide instructions how to install dependencies via conda. First, clone the repository locally:

git clone https://github.com/facebookresearch/detr.git
Then, install PyTorch 1.5+ and torchvision 0.6+:

conda install -c pytorch pytorch torchvision
Install pycocotools (for evaluation on COCO) and scipy (for training):

conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
That's it, should be good to train and evaluate detection models.

3.  **Setup Secrets**
    Create a `.streamlit/secrets.toml` file and add your API keys:
    ```toml
    GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
    GROQ_API_KEY = "YOUR_GROQ_API_KEY"
    HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
    ```

4.  **Run the App**
    ```bash
    streamlit run ui-resnet-custom.py
    ```

## Results & Discussion
* **Standard Detection:** The model performs exceptionally well on standard categories like people, vehicles, and furniture.
* **Custom Object Detection:** While the system successfully integrates 'edodo', detection accuracy is currently in an iterative improvement phase due to the time-intensive nature of high-quality sample generation.
* **Fail-safe Logic:** The dual-API strategy effectively maintains system uptime, providing descriptive reports even under high-load or token-limit scenarios.

## References
* [Hugging Face DETR Resnet-50](https://huggingface.co/facebook/detr-resnet-50)
* [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

---
**Author:** Edward (Seongmin Choi)  
**Program:** AIM1 (Postgraduate in AI & Machine Learning) at Fanshawe College