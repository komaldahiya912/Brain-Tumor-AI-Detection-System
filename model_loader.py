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

# Define the segmentation model architecture
class ImprovedResUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedResUNet, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=False)
        
        # Encoder
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_block(2048, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._make_decoder_block(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        # Bridge
        bridge = self.bridge(enc5)
        
        # Decoder with skip connections
        dec5 = self.upconv5(bridge)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, x], dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final(dec1))


# Define the quantum classifier
class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super(QuantumClassifier, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
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
        self.weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        
        # Classical preprocessing
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, n_qubits)
        
        # Initialize quantum weights
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        
    def forward(self, x):
        # Classical preprocessing
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Quantum processing
        outputs = []
        for sample in x:
            output = self.circuit(sample, self.q_weights)
            outputs.append(output)
        
        return torch.stack(outputs).unsqueeze(1)


class BrainTumorPredictor:
    def __init__(self, seg_model_path, quantum_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load segmentation model
        self.seg_model = ImprovedResUNet(num_classes=1)
        if os.path.exists(seg_model_path):
            self.seg_model.load_state_dict(torch.load(seg_model_path, map_location=self.device))
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        # Load quantum classifier
        self.quantum_model = QuantumClassifier(n_qubits=4, n_layers=2)
        if os.path.exists(quantum_model_path):
            self.quantum_model.load_state_dict(torch.load(quantum_model_path, map_location=self.device))
        self.quantum_model.to(self.device)
        self.quantum_model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, image_path):
        """Run complete prediction pipeline"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Expand to 3 channels for ResNet
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
            
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
        # Simple feature extraction
        features = []
        
        # Statistical features
        features.append(np.mean(tumor_mask))
        features.append(np.std(tumor_mask))
        features.append(np.max(tumor_mask))
        features.append(np.min(tumor_mask))
        
        # Shape features
        binary_mask = (tumor_mask > 0.5).astype(np.uint8)
        features.append(np.sum(binary_mask))
        
        # Pad to 256 features
        while len(features) < 256:
            features.extend(features[:min(5, 256 - len(features))])
        
        return torch.tensor(features[:256], dtype=torch.float32)


def download_models():
    """Download models from Google Drive if not present"""
    
    # Google Drive file IDs
    SEG_MODEL_ID = "1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R"
    QUANTUM_MODEL_ID = "1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs"
    
    seg_model_path = 'resnet_segmentation_model.pth'
    quantum_model_path = 'quantum_classifier_fixed.pth'
    
    # Download segmentation model
    if not os.path.exists(seg_model_path):
        try:
            st.info("📥 Downloading segmentation model (first time only, ~100MB)...")
            url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
            gdown.download(url, seg_model_path, quiet=False)
            st.success("✅ Segmentation model downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download segmentation model: {str(e)}")
            st.info("Please make sure the Google Drive link is set to 'Anyone with the link can view'")
            return False
    
    # Download quantum model
    if not os.path.exists(quantum_model_path):
        try:
            st.info("📥 Downloading quantum classifier (first time only, ~3MB)...")
            url = f'https://drive.google.com/uc?id={QUANTUM_MODEL_ID}'
            gdown.download(url, quantum_model_path, quiet=False)
            st.success("✅ Quantum classifier downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download quantum model: {str(e)}")
            st.info("Please make sure the Google Drive link is set to 'Anyone with the link can view'")
            return False
    
    return True

