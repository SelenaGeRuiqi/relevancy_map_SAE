# Relevance Map SAE

A project for implementing and analyzing Sparse Autoencoders (SAE) for relevance mapping in vision-language models.

## 🚀 Features

- **Sparse Autoencoder Implementation**: Complete SAE implementation with various activation functions
- **Model Loading**: Pre-trained MSAE models from Hugging Face Hub
- **Relevance Mapping**: Generate relevance maps for image understanding
- **GPU Support**: Full CUDA support for accelerated computation
- **Modular Design**: Clean, modular code structure

## 📁 Project Structure

```
relevance_map_SAE/
├── models/          # SAE model files and utilities
│   ├── sae.py      # Main SAE implementation
│   ├── utils.py     # Utility functions
│   └── config.py    # Configuration settings
├── data/            # Test images and datasets
├── src/             # Source code files
├── outputs/         # Generated results and outputs
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/relevancy_map_SAE.git
   cd relevancy_map_SAE
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv sae_env
   source sae_env/bin/activate  # On Windows: sae_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- **PyTorch** >= 2.0.0
- **Transformers** >= 4.35.0
- **Hugging Face Hub** >= 0.17.0
- **NumPy** >= 1.24.0
- **Matplotlib** >= 3.7.0
- **OpenCV** >= 4.8.0
- **Pillow** >= 10.0.0

## 🔧 Usage

### Loading SAE Models

```python
from models.sae import SAE
import torch

# Load pre-trained SAE model
model_path = "path/to/model.pth"
sae_model = SAE(model_path)

# Encode input data
input_tensor = torch.randn(1, 768)  # Example input
latents, full_latents = sae_model.encode(input_tensor)
```

### Generating Relevance Maps

```python
# Example code for generating relevance maps
# (Implementation details to be added)
```

## 🎯 Key Components

### SAE Model
- **Autoencoder**: Standard autoencoder with sparse activations
- **TopK Activation**: Keeps only top-k activations for sparsity
- **JumpReLU**: Custom activation with learnable thresholds
- **Matryoshka Autoencoder**: Nested autoencoder architecture

### Utilities
- **Data Processing**: Efficient dataset handling with memory mapping
- **Learning Rate Schedulers**: Cosine warmup and linear decay schedulers
- **Custom Activations**: Rectangle, JumpReLU, and Step functions

## 📊 Model Architecture

The SAE model consists of:
- **Encoder**: Maps input to sparse latent representation
- **Decoder**: Reconstructs input from latent representation
- **Activation Functions**: Various sparse activation functions
- **Normalization**: Mean-centering and scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the MSAE implementation by WolodjaZ
- Uses pre-trained models from Hugging Face Hub
- Inspired by research on sparse autoencoders for interpretability

## 📞 Contact

For questions and support, please open an issue on GitHub. 