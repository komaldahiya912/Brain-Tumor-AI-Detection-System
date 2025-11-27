# Brain Tumor AI Detection System

> **Powered by Deep Learning & Quantum Computing**

Brain tumor detection application using ResUNet deep learning architecture and Quantum computing classifiers for accurate MRI scan analysis.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg)](https://streamlit.io)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

---

## Overview

This system demonstrates a complete AI pipeline for brain tumor detection and classification using state-of-the-art deep learning and quantum computing techniques. The application processes MRI brain scans to detect tumors, classify their grades, and generate comprehensive medical reports.

**DISCLAIMER:** This is a research demonstration system for educational purposes only. NOT for clinical diagnosis or medical use.

---

## Features

- Tumor Detection**: Automatically identifies tumor regions in brain MRI scans
- Grade Classification**: Classifies tumors into Grade 1 or Grade 2 using quantum computing
- Visual Analysis**: Interactive visualization with tumor overlay on original scans
- PDF Reports**: Generates downloadable medical-style analysis reports
- Patient Database**: Stores and manages prediction history with SQLite
- Statistics Dashboard**: Comprehensive analytics and summary statistics
- Quantum Enhanced**: Uses 4-qubit quantum neural network for classification

---

## Technology Stack

### Deep Learning Models

**Segmentation Model**
- Architecture: ResNet50-based U-Net with Attention Mechanism
- Framework: PyTorch
- Performance:
  - Dice Score: 85.71%
  - IoU: 82.30%
  - Pixel Accuracy: 99.61%

**Quantum Classifier**
- Architecture: 4-qubit variational quantum circuit
- Framework: PennyLane
- Classes: Binary (Grade 1 vs Grade 2)
- Layers: 2 variational layers

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.12
- **Deep Learning**: PyTorch, torchvision
- **Quantum ML**: PennyLane
- **Image Processing**: PIL, scikit-image, OpenCV
- **Database**: SQLite3
- **Visualization**: Matplotlib, Pandas
- **Reports**: ReportLab

---

## Installation

### Prerequisites
- Python 3.12
- pip package manager
- Virtual environment (recommended)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System.git
cd Brain-Tumor-AI-Detection-System
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure secrets** (Optional for local development)
```bash
mkdir -p .streamlit
cp secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your Google Drive file IDs
```

5. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy your forked repository
4. Add secrets in Settings → Secrets (use `secrets.toml.template` as reference)

---

## Usage

### 1. Upload MRI Scan
- Navigate to "Upload & Analyze" page
- Enter patient name
- Upload brain MRI scan (PNG, JPG, JPEG, TIFF formats supported)

### 2. Run Analysis
- Click "Analyze MRI Scan" button
- Wait for AI processing (typically 5-10 seconds)
- View results with tumor overlay visualization

### 3. Review Results
- **Tumor Detection**: Whether tumor is present
- **Grade Classification**: Predicted tumor grade (1 or 2)
- **Confidence Score**: Model confidence in prediction
- **Statistics**: Detailed segmentation metrics

### 4. Download Report
- Click "Download PDF Report" to get a comprehensive analysis document
- Report includes all metrics, visualizations, and copyright information

### 5. View History
- Navigate to "Prediction History" page
- Search by Patient ID
- View summary statistics and grade distribution

---

## Model Performance

### Segmentation Model (ResUNet)
| Metric | Score |
|--------|-------|
| Dice Score | 85.71% |
| IoU (Intersection over Union) | 82.30% |
| Pixel Accuracy | 99.61% |
| Training Epochs | 7 |
| Dataset Size | 3,500+ images |

### Quantum Classifier
| Specification | Value |
|---------------|-------|
| Qubits | 4 |
| Layers | 2 |
| Classes | 2 (Grade 1/2) |
| Framework | PennyLane |
| Training Epochs | 3 |

---

## Citation

If you use this system in your research, academic work, or projects, please cite:

### BibTeX
```bibtex
@software{dahiya2025braintumor,
  author = {Dahiya, Komal},
  title = {Brain Tumor AI Detection System: A Deep Learning and Quantum Computing Approach},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System},
  institution = {Panipat Institute of Engineering \& Technology}
}
```

### APA Format
```
Dahiya, K. (2025). Brain Tumor AI Detection System: A Deep Learning and Quantum 
Computing Approach [Computer software]. GitHub. 
https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System
```

### IEEE Format
```
K. Dahiya, "Brain Tumor AI Detection System: A Deep Learning and Quantum Computing 
Approach," 2025. [Online]. Available: 
https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System
```

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

### What this means:

**You CAN:**
- Use this software for any purpose
- Study how the software works and modify it
- Distribute copies of the software
- Distribute modified versions

**You MUST:**
- Include the original copyright notice
- State significant changes made to the software
- Release the source code when you distribute the software
- License your derivative work under GPL-3.0

**You CANNOT:**
- Sublicense the software
- Hold the author liable for damages
- Make it proprietary/closed source

See [LICENSE](LICENSE) file for full details.

### Copyright Notice

```
Brain Tumor AI Detection System
Copyright (C) 2025 Komal Dahiya

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
```

---

## Author

**Komal Dahiya**  
B.Tech Computer Science & Engineering (AI & Data Science)  
Panipat Institute of Engineering & Technology

### Connect
- GitHub: [@komaldahiya912](https://github.com/komaldahiya912)
- Project Link: [Brain-Tumor-AI-Detection-System](https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System)

---

## Contributing

Contributions, issues, and feature requests are welcome! However, please note:

1. All contributions must be licensed under GPL-3.0
2. Significant changes should be discussed in an issue first
3. Follow the existing code style and structure
4. Add appropriate tests and documentation

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- Brain MRI dataset providers
- PyTorch and PennyLane communities
- Streamlit for the amazing framework
- My academic advisors and peers

---

## Important Disclaimers

### Medical Disclaimer
This software is provided for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**. It is:
- NOT a medical device
- NOT FDA approved
- NOT intended for clinical diagnosis
- NOT a substitute for professional medical advice

**Always consult qualified healthcare professionals for medical decisions.**

### Academic Integrity
This project represents original academic work. If you:
- Use this code in your academic work
- Build upon this project
- Reference this methodology

Please provide proper attribution using the citation formats above.

### Privacy Notice
- Patient data is stored locally
- No data is transmitted to external servers (except model downloads)
- Users are responsible for data protection compliance in their jurisdiction
- Delete patient data responsibly per regulations (HIPAA, GDPR, etc.)

---

## Support

For issues, questions, or suggestions:

1. Check existing [Issues](https://github.com/komaldahiya912/Brain-Tumor-AI-Detection-System/issues)
2. Create a new issue with detailed description
3. Tag appropriately (bug, enhancement, question)

---

## Changelog

### Version 1.0 (2025)
- Initial release
- ResUNet segmentation model
- Quantum classifier integration
- PDF report generation
- Patient database system
- Streamlit web interface

---

## 🔮 Future Enhancements

- [ ] Multi-class tumor classification
- [ ] 3D MRI scan support
- [ ] Real-time processing optimization
- [ ] Mobile app development
- [ ] Cloud-based deployment options
- [ ] Integration with PACS systems
- [ ] Multi-language support

---

<div align="center">

** Star this repository if you find it helpful!**

Made with ❤️ by [Komal Dahiya](https://github.com/komaldahiya912)

© 2025 | Licensed under GPL-3.0

</div>
