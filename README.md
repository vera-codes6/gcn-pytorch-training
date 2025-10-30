# PyTorch GNN Training Template

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0.2-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive and modular PyTorch template for training Graph Neural Networks (GNNs) on node classification tasks. This repository provides a clean, extensible framework for experimenting with various GNN architectures and datasets.

## 🌟 Features

- **Modular Architecture**: Clean separation of data handling, model definition, and training logic
- **Multiple GNN Models**: Support for GCN, GAT, and easily extensible to other architectures
- **Flexible Configuration**: YAML-based configuration system for easy hyperparameter management
- **Reproducibility**: Built-in random seed management for reproducible experiments
- **Early Stopping**: Configurable patience-based early stopping with validation monitoring
- **Multi-Run Support**: Run multiple experiments with different random seeds
- **Comprehensive Logging**: Detailed logging of training metrics and results
- **PyTorch Geometric Integration**: Seamless integration with PyG for graph data handling

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Available Models](#available-models)
- [Usage Examples](#usage-examples)
- [Extending the Framework](#extending-the-framework)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vera-codes6/gcn-pytorch-training.git
cd gcn-pytorch-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

```
torch==1.10.0
torch_geometric==2.0.2
torch_scatter==2.0.9
torch_sparse==0.6.12
numpy==1.20.3
scikit_learn==1.0.2
PyYAML==6.0
tqdm==4.62.3
```

## ⚡ Quick Start

Train a GCN model on the Cora dataset:

```bash
python main.py --model GCN --dataset Cora
```

Train a GAT model on the Cora dataset:

```bash
python main.py --model GAT --dataset Cora
```

## 📁 Project Structure

```
.
├── main.py                      # Main entry point
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
│   └── Cora.yml                # Cora dataset configuration
├── DataHelper/                  # Data handling modules
│   ├── dataset_helper.py       # Dataset utilities
│   └── DatasetLocal.py         # Custom dataset loader
├── datasets/                    # Dataset storage
│   └── Cora/                   # Cora dataset files
│       ├── processed/          # Processed data cache
│       └── raw/                # Raw dataset files
├── model/                       # Model implementations
│   ├── GAT.py                  # Graph Attention Network
│   └── GCN.py                  # Graph Convolutional Network
├── training_procedure/          # Training pipeline
│   ├── __init__.py             # Trainer initialization
│   ├── prepare.py              # Model preparation
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation logic
└── utils/                       # Utility modules
    ├── constants.py            # Global constants
    ├── logger.py               # Logging utilities
    ├── random_seeder.py        # Random seed management
    └── utils.py                # Helper functions
```

## ⚙️ Configuration

Configuration files are stored in `config/{dataset_name}.yml` format. Each configuration file contains hyperparameters for all supported models.

### Configuration File Format

```yaml
dataset: "Cora"
model_name: "GCN"  # Default model to use

# GCN Configuration
GCN:
  epochs: 150                    # Number of training epochs
  multirun: 10                   # Number of runs with different seeds
  dropout: 0.5                   # Dropout rate
  cuda: 0                        # GPU device ID (-1 for CPU)
  feat_norm: True                # Feature normalization
  hidden_dim: 64                 # Hidden layer dimension
  multilabel: False              # Multi-label classification
  patience: 50                   # Early stopping patience
  seed: 1234                     # Random seed
  lr: 0.005                      # Learning rate
  weight_decay: 0.0005           # L2 regularization
  lr_scheduler: False            # Learning rate scheduler
  monitor: "val_acc"             # Metric to monitor (val_acc/val_loss)
  recache: False                 # Force recache dataset
  optimizer: "Adam"              # Optimizer type
  num_layers: 2                  # Number of GNN layers
  activation: "relu"             # Activation function

# GAT Configuration
GAT:
  epochs: 100
  multirun: 10
  dropout: 0.6
  cuda: 0
  feat_norm: True
  hidden_dim: 64
  multilabel: False
  heads: 1                       # Number of attention heads
  patience: 50
  seed: 1234
  lr: 0.005
  weight_decay: 0.0005
  lr_scheduler: False
  monitor: "val_acc"
  recache: False
  num_layers: 2
  optimizer: "Adam"
  activation: "leaky_relu"
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Maximum number of training epochs | 150 |
| `multirun` | Number of runs with different seeds | 10 |
| `dropout` | Dropout probability | 0.5 |
| `hidden_dim` | Hidden layer dimensions | 64 |
| `lr` | Learning rate | 0.005 |
| `weight_decay` | L2 regularization coefficient | 0.0005 |
| `patience` | Early stopping patience | 50 |
| `monitor` | Metric for early stopping | "val_acc" |
| `num_layers` | Number of GNN layers | 2 |

## 🧠 Available Models

### Graph Convolutional Network (GCN)

Implementation of the GCN model from [Kipf & Welling (ICLR 2017)](https://arxiv.org/abs/1609.02907).

**Location**: `model/GCN.py`

**Usage**:
```bash
python main.py --model GCN --dataset Cora
```

### Graph Attention Network (GAT)

Implementation of the GAT model from [Veličković et al. (ICLR 2018)](https://arxiv.org/abs/1710.10903).

**Location**: `model/GAT.py`

**Usage**:
```bash
python main.py --model GAT --dataset Cora
```

## 📚 Usage Examples

### Basic Training

```bash
# Train GCN on Cora
python main.py --model GCN --dataset Cora

# Train GAT on Cora
python main.py --model GAT --dataset Cora
```

### Advanced Options

```bash
# Train without validation set
python main.py --model GCN --dataset Cora --no_dev

# Custom configuration
python main.py --model GCN --dataset Cora --config custom_config.yml
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (GCN, GAT) | Required |
| `--dataset` | Dataset name (Cora) | Required |
| `--config` | Path to config file | `config/{dataset}.yml` |
| `--no_dev` | Skip validation during training | False |

## 🔧 Extending the Framework

### Adding a New Model

1. Create a new model file in `model/`:

```python
# model/YourModel.py
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class YourModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        # Define your model architecture
        
    def forward(self, x, edge_index):
        # Implement forward pass
        return x
```

2. Add model configuration to `config/Cora.yml`:

```yaml
YourModel:
  epochs: 100
  hidden_dim: 64
  # Add other hyperparameters
```

3. Update the model initialization in `training_procedure/prepare.py` if needed.

### Adding a New Dataset

1. Place dataset files in `datasets/{dataset_name}/raw/`
2. Create a configuration file `config/{dataset_name}.yml`
3. Update `DataHelper/dataset_helper.py` if custom preprocessing is needed

## 🙏 Acknowledgements

This project is inspired by:

- [TWIRLS](https://github.com/FFTYYY/TWIRLS) - Graph neural network research framework
- [IFM Lab Program Template](http://www.ifmlab.org/files/template/IFM_Lab_Program_Template_Python3.zip) - Python project template

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🎯 Citation

If you use this template in your research, please cite:

```bibtex
@misc{pytorch-gnn-template,
  author = {Vera Codes},
  title = {PyTorch GNN Training Template},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/vera-codes6/gcn-pytorch-training}
}
```

---

**Happy Training! 🚀**
