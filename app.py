"""
Brain Tumor AI Detection System using Res-U-Net and quantum computing
Copyright (C) 2025 Komal Dahiya
B.Tech CSE (AI & Data Science)
Panipat Institute of Engineering & Technology

This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

GitHub: https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
from model_loader import BrainTumorPredictor
from database import PredictionDatabase
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from skimage.transform import resize
import io

# Download models from Google Drive
def download_models():
    """Download models from Google Drive if not present"""
    try:
        # Get model IDs from Streamlit secrets (secure method)
        SEG_MODEL_ID = st.secrets.get("SEG_MODEL_ID", "1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R")
        QUANTUM_MODEL_ID = st.secrets.get("QUANTUM_MODEL_ID", "1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs")
    except:
        # Fallback to default values if secrets not configured
        SEG_MODEL_ID = "1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R"
        QUANTUM_MODEL_ID = "1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs"
    
    import gdown
    
    seg_model_path = 'resnet_segmentation_model.pth'
    quantum_model_path = 'quantum_classifier_fixed.pth'
    
    # Download segmentation model
    if not os.path.exists(seg_model_path):
        try:
            st.info("Downloading segmentation model (first time only, ~100MB)...")
            url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
            gdown.download(url, seg_model_path, quiet=False)
            st.success("Segmentation model downloaded!")
        except Exception as e:
            st.error(f"Failed to download segmentation model: {str(e)}")
            st.info("Please make sure the Google Drive link is set to 'Anyone with the link can view'")
            return False
    
    # Download quantum model
    if not os.path.exists(quantum_model_path):
        try:
            st.info("Downloading quantum classifier (first time only, ~3MB)...")
            url = f'https://drive.google.com/uc?id={QUANTUM_MODEL_ID}'
            gdown.download(url, quantum_model_path, quiet=False)
            st.success("Quantum classifier downloaded!")
        except Exception as e:
            st.error(f"Failed to download quantum model: {str(e)}")
            st.info("Please make sure the Google Drive link is set to 'Anyone with the link can view'")
            return False
    
    return True

# Initialize
@st.cache_resource
def load_predictor():
    """Load the predictor with trained models"""
    try:
        return BrainTumorPredictor(
            seg_model_path='resnet_segmentation_model.pth',
            quantum_model_path='quantum_classifier_fixed.pth'
        )
    except Exception as e:
        st.error(f"Error loading predictor: {str(e)}")
        return None

@st.cache_resource  
def load_database():
    """Initialize database"""
    try:
        return PredictionDatabase()
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None

def create_overlay_image(original_img, tumor_mask, alpha=0.5):
    """Create image with tumor overlay"""
    try:
        if original_img.shape != tumor_mask.shape:
            tumor_mask = resize(tumor_mask, original_img.shape, preserve_range=True, anti_aliasing=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(original_img, cmap='gray', vmin=0, vmax=255)
        
        tumor_overlay = np.zeros((*tumor_mask.shape, 4))
        tumor_mask_binary = tumor_mask > 0.5
        tumor_overlay[tumor_mask_binary, 0] = 1.0
        tumor_overlay[tumor_mask_binary, 3] = alpha
        
        ax.imshow(tumor_overlay, interpolation='nearest')
        ax.axis('off')
        ax.set_title('MRI with Tumor Detection (Red = Detected Tumor)', fontsize=14, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return None

def generate_pdf_report(patient_name, results, original_img, tumor_mask):
    """Generate PDF report with copyright watermark"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title = Paragraph("<b>Brain Tumor Analysis Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        info = f"""
        <b>Patient Name:</b> {patient_name}<br/>
        <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <br/>
        <b>TUMOR DETECTION:</b><br/>
        <b>Tumor Detected:</b> {'Yes' if results['tumor_present'] else 'No'}<br/>
        <b>Tumor Area:</b> {results['tumor_area']:.0f} pixels<br/>
        <br/>
        <b>GRADE CLASSIFICATION:</b><br/>
        <b>Predicted Grade:</b> Grade {results['predicted_grade']}<br/>
        <b>Confidence Score:</b> {results['grade_confidence']:.4f}<br/>
        <br/>
        <b>SEGMENTATION STATISTICS:</b><br/>
        <b>Mean Probability:</b> {results['segmentation_stats']['mean_prob']:.4f}<br/>
        <b>Std Probability:</b> {results['segmentation_stats']['std_prob']:.4f}<br/>
        <b>Max Probability:</b> {results['segmentation_stats']['max_prob']:.4f}<br/>
        <b>Tumor Ratio:</b> {results['segmentation_stats']['tumor_ratio']:.4f}<br/>
        <br/>
        <i>Note: This is an AI-generated analysis for research purposes only. 
        Please consult medical professionals for diagnosis.</i>
        """
        story.append(Paragraph(info, styles['Normal']))
        story.append(Spacer(1, 30))
        
        overlay_buf = create_overlay_image(original_img, tumor_mask)
        if overlay_buf:
            img = RLImage(overlay_buf, width=400, height=400)
            story.append(img)
        
        story.append(Spacer(1, 30))
        
        # Copyright watermark
        copyright_text = """
        <br/><br/>
        ───────────────────────────────────────────────────────────<br/>
        <b>Brain Tumor AI Detection System © 2025 Komal Dahiya</b><br/>
        <i>Powered by Deep Learning & Quantum Computing</i><br/>
        GitHub: github.com/komaldahiya912/Brain-Tumor-AI-Detection-System<br/>
        <i>Licensed under GNU GPL-3.0</i><br/>
        ───────────────────────────────────────────────────────────
        """
        story.append(Paragraph(copyright_text, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Brain Tumor AI Detector", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Download models first
    if not download_models():
        st.error("Failed to download models. Please check your Google Drive file IDs.")
        st.info("""
        **Instructions to fix:**
        1. Upload your model files to Google Drive
        2. Right-click each file → Share → Set to 'Anyone with the link'
        3. Copy the file ID from the share link
        4. Add them to Streamlit secrets (Settings → Secrets)
        """)
        st.stop()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load models and database
    predictor = load_predictor()
    db = load_database()
    
    if predictor is None or db is None:
        st.error("Failed to load models or database. Please check your model files.")
        st.info("Models should be downloaded automatically from Google Drive.")
        st.stop()
    
    # Header
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
    <p><b>Brain Tumor AI Detection System v1.0</b></p>
    <p>Powered by Deep Learning & Quantum Computing</p>
    <p style="font-size: 0.8em;">© 2024 Komal Dahiya | Licensed under GPL-3.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload & Analyze", "Prediction History", "About"])
    
    if page == "Upload & Analyze":
        st.header("Upload MRI Scan for Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            patient_name = st.text_input(
                "Patient Name *", 
                placeholder="Enter patient name",
                help="Enter the full name of the patient"
            )
        
        uploaded_file = st.file_uploader(
            "Choose MRI Image *", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload brain MRI scan in standard image formats (PNG, JPG, JPEG, TIFF)"
        )
        
        if uploaded_file and patient_name:
            try:
                os.makedirs("static/uploads", exist_ok=True)
                upload_path = os.path.join("static", "uploads", uploaded_file.name)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original MRI Scan")
                    try:
                        original_img = Image.open(upload_path).convert('L')
                        st.image(np.array(original_img), use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                        st.stop()
                
                if st.button("Analyze MRI Scan", type="primary"):
                    with st.spinner("Running AI analysis... This may take a moment..."):
                        try:
                            results = predictor.predict(upload_path)
                            prediction_id = db.save_prediction(patient_name, upload_path, results)
                            
                            with col2:
                                st.subheader("Analysis Results")
                                original_array = np.array(original_img)
                                overlay_buf = create_overlay_image(original_array, results['tumor_mask'])
                                if overlay_buf:
                                    st.image(overlay_buf, use_column_width=True)
                            
                            st.markdown("---")
                            st.subheader("Analysis Summary")
                            
                            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                            with result_col1:
                                if results['tumor_present']:
                                    st.error("Tumor Detected")
                                else:
                                    st.success("No Tumor Detected")
                            with result_col2:
                                st.metric("Predicted Grade", f"Grade {results['predicted_grade']}")
                            with result_col3:
                                st.metric("Confidence", f"{results['grade_confidence']:.3f}")
                            with result_col4:
                                st.metric("Tumor Area", f"{results['tumor_area']:.0f} px")
                            
                            st.subheader("Detailed Results")
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.markdown("**Detection Results:**")
                                detection_df = pd.DataFrame({
                                    'Metric': ['Tumor Present', 'Tumor Area (pixels)', 'Detection Threshold'],
                                    'Value': [
                                        'Yes' if results['tumor_present'] else 'No',
                                        f"{results['tumor_area']:.0f}",
                                        '50 pixels (>0.5 prob)'
                                    ]
                                })
                                st.dataframe(detection_df, hide_index=True)
                            with detail_col2:
                                st.markdown("**Classification Results:**")
                                classification_df = pd.DataFrame({
                                    'Metric': ['Predicted Grade', 'Grade Confidence', 'Classification Method'],
                                    'Value': [
                                        f"Grade {results['predicted_grade']}",
                                        f"{results['grade_confidence']:.4f}",
                                        'Quantum Neural Network'
                                    ]
                                })
                                st.dataframe(classification_df, hide_index=True)
                            
                            st.subheader("Segmentation Statistics")
                            seg_stats = results['segmentation_stats']
                            seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)
                            with seg_col1:
                                st.metric("Mean Probability", f"{seg_stats['mean_prob']:.4f}")
                            with seg_col2:
                                st.metric("Std Probability", f"{seg_stats['std_prob']:.4f}")
                            with seg_col3:
                                st.metric("Max Probability", f"{seg_stats['max_prob']:.4f}")
                            with seg_col4:
                                st.metric("Tumor Ratio", f"{seg_stats['tumor_ratio']:.4f}")
                            
                            st.markdown("---")
                            
                            pdf_buffer = generate_pdf_report(
                                patient_name, results, original_array, results['tumor_mask']
                            )
                            
                            if pdf_buffer:
                                st.download_button(
                                    "Download PDF Report",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"{patient_name.replace(' ', '_')}_brain_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            
                            st.success(f"Analysis complete! Saved to database with ID: {prediction_id}")
                            st.warning("Important: This is an AI research system. Results should be verified by medical professionals before any clinical decision.")
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        elif not patient_name and not uploaded_file:
            st.info("Please enter patient name and upload an MRI scan to begin analysis.")
        elif not patient_name:
            st.warning("Please enter patient name.")
        elif not uploaded_file:
            st.warning("Please upload an MRI scan.")
    
    elif page == "Prediction History":
        st.header("Prediction History")
        
        # Search by Patient ID
        col1, col2 = st.columns([3, 1])
        with col1:
            search_id = st.text_input(
                "Search by Patient ID", 
                placeholder="Enter Patient ID to search (e.g., 1, 2, 3...)",
                help="Enter the ID number to find a specific patient record",
                key="search_input"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear Search"):
                st.rerun()
        
        try:
            predictions = db.get_all_predictions()
            
            if predictions:
                df_data = []
                for pred in predictions:
                    df_data.append({
                        'ID': pred[0],
                        'Patient': pred[1],
                        'Date': pred[2],
                        'Tumor': 'Yes' if pred[4] else 'No',
                        'Grade': f"Grade {pred[5]}",
                        'Confidence': f"{pred[6]:.3f}",
                        'Area (px)': f"{pred[7]:.0f}"
                    })
                df = pd.DataFrame(df_data)
                
                # Filter by search ID if provided
                if search_id:
                    try:
                        search_id_int = int(search_id)
                        filtered_df = df[df['ID'] == search_id_int]
                        
                        if not filtered_df.empty:
                            st.success(f"Found 1 record with Patient ID: {search_id_int}")
                            st.dataframe(filtered_df, hide_index=True, use_container_width=True)
                            
                            # Show detailed info for searched patient
                            st.markdown("---")
                            st.subheader("Patient Details")
                            record = filtered_df.iloc[0]
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            with detail_col1:
                                st.metric("Patient Name", record['Patient'])
                                st.metric("Analysis Date", record['Date'])
                            with detail_col2:
                                st.metric("Tumor Status", record['Tumor'])
                                st.metric("Tumor Grade", record['Grade'])
                            with detail_col3:
                                st.metric("Confidence", record['Confidence'])
                                st.metric("Tumor Area", record['Area (px)'])
                        else:
                            st.warning(f"No record found with Patient ID: {search_id_int}")
                            st.info("Showing all records below:")
                            st.dataframe(df, hide_index=True, use_container_width=True)
                    except ValueError:
                        st.error("Please enter a valid numeric Patient ID")
                        st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("Showing all patient records. Use search box above to find specific Patient ID.")
                    st.dataframe(df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Summary Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Scans", len(predictions))
                with col2:
                    tumor_count = sum(1 for p in predictions if p[4])
                    st.metric("Tumors Detected", tumor_count)
                with col3:
                    no_tumor_count = len(predictions) - tumor_count
                    st.metric("No Tumor", no_tumor_count)
                with col4:
                    if tumor_count > 0:
                        avg_grade = sum(p[5] for p in predictions if p[4]) / tumor_count
                        st.metric("Avg Grade", f"{avg_grade:.2f}")
                    else:
                        st.metric("Avg Grade", "N/A")
                with col5:
                    if tumor_count > 0:
                        avg_conf = sum(p[6] for p in predictions if p[4]) / tumor_count
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                if tumor_count > 0:
                    st.subheader("Grade Distribution")
                    grade_data = {}
                    for p in predictions:
                        if p[4]:
                            grade = p[5]
                            grade_data[f"Grade {grade}"] = grade_data.get(f"Grade {grade}", 0) + 1
                    grade_df = pd.DataFrame(list(grade_data.items()), columns=['Grade', 'Count'])
                    st.bar_chart(grade_df.set_index('Grade'))
            else:
                st.info("No predictions in database yet. Upload an MRI scan to get started!")
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")
            st.exception(e)
    
    elif page == "About":
        st.header("About This System")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ## Brain Tumor AI Detection System
            
            This application demonstrates a complete AI pipeline for brain tumor detection 
            and classification using state-of-the-art deep learning and quantum computing.
            
            ### Technology Stack
            
            #### Segmentation Model
            - **Architecture**: ResNet50-based U-Net with Attention Mechanism
            - **Purpose**: Identifies tumor regions in MRI scans
            - **Training**: 3,500+ medical images with data augmentation
            - **Performance**: 
              - Dice Score: 85.71%
              - IoU: 82.30%
              - Pixel Accuracy: 99.61%
            
            #### Quantum Classifier  
            - **Architecture**: 4-qubit variational quantum circuit
            - **Purpose**: Classifies tumor grades (Grade 1 vs Grade 2)
            - **Technology**: PennyLane quantum machine learning framework
            - **Features**: Hybrid classical-quantum processing
            
            ### Pipeline Process
            
            1. **Image Upload** → Patient uploads brain MRI scan
            2. **Segmentation** → AI identifies and highlights tumor regions  
            3. **Feature Extraction** → Extracts quantum-compatible features
            4. **Classification** → Quantum circuit predicts tumor grade
            5. **Visualization** → Overlays results on original scan
            6. **Report Generation** → Creates downloadable PDF analysis
            
            ### Key Features
            
            - Real-time MRI analysis
            - Tumor detection and grading
            - Patient history database
            - PDF report generation
            - Interactive visualization
            - Quantum-enhanced AI processing
            - Comprehensive statistics
            
            ### Technical Details
            
            **Models Used:**
            - Segmentation: ImprovedResUNet (ResNet50 + U-Net + Attention)
            - Classification: 4-Qubit Quantum Neural Network
            
            **Training:**
            - Segmentation: 7 epochs, 85.71% Dice Score
            - Quantum: 7 epochs, Binary classification
            
            **Loss Functions:**
            - Segmentation: Combined (BCE + Dice + Focal Tversky)
            - Quantum: Binary Cross Entropy with Logits
            """)
        
        with col2:
            st.markdown("""
            ### Model Performance
            
            **Segmentation Model:**
            - Dice Score: 85.71%
            - IoU: 82.30%
            - Pixel Accuracy: 99.61%
            
            **Quantum Classifier:**
            - Classes: 2 (Grade 1/2)
            - Qubits: 4
            - Layers: 2
            
            ### Important Notice
            
            This is a **research demonstration** system:
            
            - NOT for clinical diagnosis
            - NOT FDA approved
            - NOT a medical device
            
            For research and educational purposes only.
            
            **Always consult qualified medical professionals.**
            
            ### Privacy & Data
            
            - Patient data stored locally
            - No cloud transmission
            - SQLite database
            - Can be deleted anytime
            
            ### System Requirements
            
            - Python 3.8+
            - PyTorch
            - PennyLane
            - Streamlit
            
            ### Copyright & License
            
            **© 2025 Komal Dahiya**  
            B.Tech CSE (AI & Data Science)  
            Panipat Institute of Engineering & Technology
            
            Licensed under GNU GPL-3.0  
            [GitHub Repository](https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System)
            
            ### Citation
            
            If you use this system in your research or work, please cite:
            
            ```
            Dahiya, K. (2025). Brain Tumor AI Detection System.
            GitHub: komaldahiya912/Brain-Tumor-AI-Detection-System
            ```
            """)

if __name__ == "__main__":
    main()
