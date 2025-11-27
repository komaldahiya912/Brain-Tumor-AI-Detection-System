import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from PIL import Image

# ==============================================================
# ATTENTION BLOCK
# ==============================================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ==============================================================
# IMPROVED RESUNET (SEGMENTATION MODEL)
# ==============================================================
class ImprovedResUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        backbone.conv1.weight.data = old.weight.data.mean(dim=1, keepdim=True)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.att4 = AttentionBlock(1024, 1024, 512)
        self.att3 = AttentionBlock(512, 512, 256)
        self.att2 = AttentionBlock(256, 256, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        d1 = self.up1(x5)
        d1 = torch.cat([d1, self.att4(d1, x4)], dim=1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.att3(d2, x3)], dim=1)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.att2(d3, x2)], dim=1)

        d4 = self.up4(d3)
        d5 = self.up5(d4)

        logits = self.final(d5)
        return logits, x5

# ==============================================================
# QUANTUM CIRCUIT
# ==============================================================
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, weights):
    """Quantum circuit for binary classification"""
    for i in range(n_qubits):
        qml.RY(np.pi * inputs[i], wires=i)
    
    for layer in range(2):
        for i in range(n_qubits):
            qml.RY(weights[layer * n_qubits * 2 + i], wires=i)
            qml.RZ(weights[layer * n_qubits * 2 + n_qubits + i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
    
    return qml.expval(qml.PauliZ(0))

class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        n_params = 2 * 2 * n_qubits
        self.weights = nn.Parameter(0.01 * torch.randn(n_params))
    
    def forward(self, x):
        """Process batch of features through quantum circuit"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            result = quantum_circuit(x[i], self.weights)
            outputs.append(result)
        
        return torch.stack(outputs)

# ==============================================================
# BRAIN TUMOR PREDICTOR
# ==============================================================
class BrainTumorPredictor:
    def __init__(self, seg_model_path='resnet_segmentation_model.pth', 
                 quantum_model_path='quantum_classifier_fixed.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.seg_model_path = seg_model_path
        self.quantum_model_path = quantum_model_path
        
        self.load_models()
        
    def load_models(self):
        """Load both segmentation and quantum models"""
        # Load segmentation model
        self.seg_model = ImprovedResUNet(pretrained=False)
        seg_checkpoint = torch.load(self.seg_model_path, map_location=self.device)
        
        if isinstance(seg_checkpoint, dict) and 'model_state_dict' in seg_checkpoint:
            self.seg_model.load_state_dict(seg_checkpoint['model_state_dict'])
            print(f"Segmentation model loaded (Dice: {seg_checkpoint.get('dice', 'N/A')})")
        else:
            self.seg_model.load_state_dict(seg_checkpoint)
            print("Segmentation model loaded")
        
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        # Load quantum classifier
        self.quantum_model = QuantumClassifier(n_qubits)
        quantum_checkpoint = torch.load(self.quantum_model_path, map_location=self.device)
        
        if isinstance(quantum_checkpoint, dict) and 'model_state_dict' in quantum_checkpoint:
            self.quantum_model.load_state_dict(quantum_checkpoint['model_state_dict'])
            print(f"Quantum classifier loaded (Accuracy: {quantum_checkpoint.get('accuracy', 'N/A')})")
        else:
            self.quantum_model.load_state_dict(quantum_checkpoint)
            print("Quantum classifier loaded")
        
        self.quantum_model.to(self.device)
        self.quantum_model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, image_path):
        """
        Make prediction on brain MRI image
        Returns dict with tumor detection and grade classification
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Segmentation - detect tumor
            seg_logits, _ = self.seg_model(img_tensor)
            seg_probs = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            
            # 2. Extract features for quantum classifier
            mean_prob = float(seg_probs.mean())
            std_prob = float(seg_probs.std())
            max_prob = float(seg_probs.max())
            tumor_ratio = float((seg_probs > 0.5).mean())
            
            features = torch.tensor([mean_prob, std_prob, max_prob, tumor_ratio], 
                                   dtype=torch.float32)
            features = torch.clamp(features, 0, 1).unsqueeze(0).to(self.device)
            
            # 3. Quantum classification - predict grade
            quantum_output = self.quantum_model(features)
            grade_prob = torch.sigmoid(quantum_output).item()
            predicted_grade = 2 if grade_prob > 0.5 else 1
            
            # 4. Tumor detection (threshold: 50 pixels with >0.5 probability)
            tumor_present = (seg_probs > 0.5).sum() > 50
            tumor_area = float((seg_probs > 0.5).sum())
        
        return {
            'tumor_present': bool(tumor_present),
            'tumor_mask': seg_probs,
            'predicted_grade': int(predicted_grade),
            'grade_confidence': float(grade_prob),
            'tumor_area': float(tumor_area),
            'segmentation_stats': {
                'mean_prob': mean_prob,
                'std_prob': std_prob,
                'max_prob': max_prob,
                'tumor_ratio': tumor_ratio
            }
        }

