import os
import gdown
import streamlit as st

def download_models():
    """Download models from Google Drive if not present"""
    
    # Your Google Drive file IDs (replace with your actual IDs)
    SEG_MODEL_ID = "https://drive.google.com/uc?export=download&id=1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R"
    QUANTUM_MODEL_ID = "https://drive.google.com/uc?export=download&id=1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs"
    
    seg_model_path = 'resnet_segmentation_model.pth'
    quantum_model_path = 'quantum_classifier_fixed.pth'
    
    # Download segmentation model
    if not os.path.exists(seg_model_path):
        try:
            with st.spinner("📥 Downloading segmentation model (first time only)..."):
                url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
                gdown.download(url, seg_model_path, quiet=False)
                st.success("✅ Segmentation model downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download segmentation model: {str(e)}")
            return False
    
    # Download quantum model
    if not os.path.exists(quantum_model_path):
        try:
            with st.spinner("📥 Downloading quantum classifier (first time only)..."):
                url = f'https://drive.google.com/uc?id={QUANTUM_MODEL_ID}'
                gdown.download(url, quantum_model_path, quiet=False)
                st.success("✅ Quantum classifier downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download quantum model: {str(e)}")
            return False
    
    return True

