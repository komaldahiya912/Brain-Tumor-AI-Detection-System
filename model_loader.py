import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pennylane as qml
import os
import gdown
import streamlit as st

# Attention Block for the segmentation model
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define the segmentation model architecture (matching saved model with 1 channel input)
class ImprovedResUNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(ImprovedResUNet, self).__init__()
        
        # Load pretrained ResNet50 and modify first conv for grayscale
        resnet = models.resnet50(pretrained=False)
        
        # Encoder - modify conv1 for single channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Attention gates
        self.att4 = AttentionBlock(1024, 1024, 512)
        self.att3 = AttentionBlock(512, 512, 256)
        self.att2 = AttentionBlock(256, 256, 128)
        
        # Decoder - match the saved model structure
        self.up1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        # Decoder with attention and upsampling
        d5 = self.up1(x5)
        d5 = nn.functional.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True)
        x4_att = self.att4(d5, x4)
        d5 = torch.cat([d5, x4_att], dim=1)
        d5 = self.up2(d5)
        
        d4 = nn.functional.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True)
        x3_att = self.att3(d4, x3)
        d4 = torch.cat([d4, x3_att], dim=1)
        d4 = self.up3(d4)
        
        d3 = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        x2_att = self.att2(d3, x2)
        d3 = torch.cat([d3, x2_att], dim=1)
        d3 = self.up4(d3)
        
        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.up5(d2)
        
        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(self.final(d1))


# Define the quantum classifier
class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super(QuantumClassifier, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Quantum weights (named 'weights' to match saved model)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        
        # Define quantum circuit
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = circuit
        
    def forward(self, x):
        # Quantum processing directly (no classical preprocessing in saved model)
        outputs = []
        for sample in x:
            output = self.circuit(sample, self.weights)
            outputs.append(output)
        
        return torch.stack(outputs).unsqueeze(1)


class BrainTumorPredictor:
    def __init__(self, seg_model_path, quantum_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load segmentation model
        self.seg_model = ImprovedResUNet(num_classes=1, in_channels=1)
        if os.path.exists(seg_model_path):
            try:
                checkpoint = torch.load(seg_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.seg_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.seg_model.load_state_dict(checkpoint)
                st.success("✅ Segmentation model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading segmentation model: {str(e)}")
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        # Load quantum classifier
        self.quantum_model = QuantumClassifier(n_qubits=4, n_layers=2)
        if os.path.exists(quantum_model_path):
            try:
                checkpoint = torch.load(quantum_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.quantum_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.quantum_model.load_state_dict(checkpoint)
                st.success("✅ Quantum model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading quantum model: {str(e)}")
        self.quantum_model.to(self.device)
        self.quantum_model.eval()
        
        # Define transforms for grayscale
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, image_path):
        """Run complete prediction pipeline"""
        try:
            # Load and preprocess image as grayscale
            image = Image.open(image_path).convert('L')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Keep as 1 channel for grayscale model
            
            # Segmentation
            with torch.no_grad():
                tumor_mask = self.seg_model(image_tensor)
                tumor_mask = tumor_mask.squeeze().cpu().numpy()
            
            # Calculate tumor statistics
            tumor_area = np.sum(tumor_mask > 0.5)
            tumor_present = tumor_area > 50
            
            seg_stats = {
                'mean_prob': float(np.mean(tumor_mask)),
                'std_prob': float(np.std(tumor_mask)),
                'max_prob': float(np.max(tumor_mask)),
                'tumor_ratio': float(tumor_area / (512 * 512))
            }
            
            # Classification
            if tumor_present:
                features = self.extract_features(tumor_mask)
                with torch.no_grad():
                    grade_output = self.quantum_model(features.unsqueeze(0).to(self.device))
                    grade_prob = torch.sigmoid(grade_output).item()
                    predicted_grade = 2 if grade_prob > 0.5 else 1
                    grade_confidence = grade_prob if predicted_grade == 2 else (1 - grade_prob)
            else:
                predicted_grade = 0
                grade_confidence = 0.0
            
            return {
                'tumor_present': tumor_present,
                'tumor_mask': tumor_mask,
                'tumor_area': float(tumor_area),
                'predicted_grade': predicted_grade,
                'grade_confidence': float(grade_confidence),
                'segmentation_stats': seg_stats
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def extract_features(self, tumor_mask):
        """Extract features from tumor mask for classification"""
        features = []
        
        # Statistical features
        features.append(np.mean(tumor_mask))
        features.append(np.std(tumor_mask))
        features.append(np.max(tumor_mask))
        features.append(np.min(tumor_mask))
        
        return torch.tensor(features[:4], dtype=torch.float32)


def download_models():
    """Download models from Google Drive if not present"""
    
    SEG_MODEL_ID = "1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R"
    QUANTUM_MODEL_ID = "1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs"
    
    seg_model_path = 'resnet_segmentation_model.pth'
    quantum_model_path = 'quantum_classifier_fixed.pth'
    
    if not os.path.exists(seg_model_path):
        try:
            st.info("📥 Downloading segmentation model (first time only, ~100MB)...")
            url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
            gdown.download(url, seg_model_path, quiet=False)
            st.success("✅ Segmentation model downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download segmentation model: {str(e)}")
            return False
    
    if not os.path.exists(quantum_model_path):
        try:
            st.info("📥 Downloading quantum classifier (first time only, ~3MB)...")
            url = f'https://drive.google.com/uc?id={QUANTUM_MODEL_ID}'
            gdown.download(url, quantum_model_path, quiet=False)
            st.success("✅ Quantum classifier downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download quantum model: {str(e)}")
            return False
    
    return True
