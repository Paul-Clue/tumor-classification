# Brain Tumor Classification with Deep Learning

An interactive web application that uses deep learning to classify brain tumor MRI scans and provides expert-level analysis using OpenAI's GPT-4 Vision.

## Features
- Upload and analyze brain MRI scans
- Real-time tumor classification using deep learning
- Saliency map visualization for model interpretability
- Expert-level analysis of results using GPT-4 Vision
- Interactive web interface using Streamlit

## Prerequisites
- Python 3.10 or higher
- Conda (conda-forge / miniforge required for tensorflow-metal) or pip
- OpenAI API key
- Mac with M1/M2 chip (for tensorflow-metal) or any machine with GPU support

## Installation

1. Clone the repository

```bash
git clone https://github.com/Paul-Clue/brain-tumor-classification.git


cd tumor-classification
```

2. Create and activate a conda environment

```bash
conda create -n tumor-classification python=3.10
conda activate tumor-classification
```

3. Install dependencies
```bash
# For Mac M1/M2 users:
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn jupyter
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt

# For other users:
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root:
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

2. Create necessary directories:
```bash
mkdir -p saliency_maps
mkdir -p models
```

3. Create pre-trained models for streamlit app:
```bash
Running all of the cells will create the models/ directory with the pre-trained models:
xception_model.h5
cnn_model.h5
```

## Running the Application

1. Start the Streamlit app locally:
```bash
streamlit run main.py
```

2. For remote access using ngrok:
```python
from pyngrok import ngrok
public_url = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print("Public URL:", public_url)
```

## Project Structure
```
tumor-classification/
├── main.py                # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── models/               # Pre-trained models
│   ├── xception_model.h5
│   └── cnn_model.h5
├── saliency_maps/        # Generated saliency maps
└── README.md
```

## Model Information
- **Transfer Learning Model**: Fine-tuned Xception architecture
- **Custom CNN**: Custom convolutional neural network
- Input shape: (224, 224, 3) for CNN, (299, 299, 3) for Xception
- Classes: ['glioma', 'meningioma', 'no_tumor', 'pituitary']

## Usage
1. Access the application through your browser
2. Upload a brain MRI scan image
3. Select the model (Transfer Learning or Custom CNN)
4. View the classification results, saliency map, and expert analysis

## Troubleshooting

### Common Issues
1. **TensorFlow Installation**
   - For Mac M1/M2: Use tensorflow-macos and tensorflow-metal
   - For other systems: Standard tensorflow package

2. **Package Conflicts**
   ```bash
   pip uninstall protobuf
   pip install protobuf==3.20.3
   ```

3. **Memory Issues**
   - Reduce batch size in model prediction
   - Close other memory-intensive applications

### Error Messages
- If you see `ModuleNotFoundError`: Check if all requirements are installed
- For CUDA errors: Verify GPU support and drivers
- For API errors: Verify your OpenAI API key in .env

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset: [Brain Tumor MRI Dataset](dataset_link)
- TensorFlow team for the deep learning framework
- OpenAI for GPT-4 Vision API
- Streamlit team for the web framework

## Project Link
 https://github.com/Paul-Clue/tumor-classification

