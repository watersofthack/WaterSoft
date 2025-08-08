# LSTM-SSI: Streamflow Prediction and Reservoir Management

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Neural Hydrology](https://img.shields.io/badge/framework-neuralhydrology-orange.svg)](https://github.com/neuralhydrology/neuralhydrology)

A comprehensive machine learning framework for hydrological modeling, combining LSTM and Transformer architectures for streamflow prediction with adaptive reservoir management strategies. Developed for the WatersoftHack 2025.

## 🌊 Project Overview

This project implements state-of-the-art deep learning approaches for:
- **Streamflow Prediction**: Using LSTM and Transformer models trained on CAMELS dataset
- **Drought Assessment**: Standardized Streamflow Index (SSI) calculation for drought monitoring
- **Reservoir Operations**: Adaptive decision-making for Taylor Park Reservoir based on SSI and storage conditions

### Key Features

- 🧠 **Dual Model Architecture**: LSTM vs Transformer comparison for streamflow prediction
- 📊 **Multi-Station Analysis**: 22 USGS gauge stations across diverse watersheds
- 🏔️ **Reservoir Management**: SSI-based adaptive operations for Taylor Park Dam
- 📈 **Comprehensive Evaluation**: Multiple performance metrics (NSE, KGE, RMSE, Pearson-r)
- 🔄 **End-to-End Pipeline**: From data preprocessing to operational decision-making

## 🏗️ Repository Structure

```
LSTM-SSI/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory
│   ├── raw/                     # Raw CAMELS data
│   │   ├── attributes/          # Basin attributes (static features)
│   │   └── time_series/         # Streamflow and meteorological data
│   ├── processed/               # Processed data for modeling
│   └── README.md                # Data documentation
│
├── src/                         # Source code
│   ├── preprocessing/           # Data preprocessing modules
│   ├── models/                  # Model implementations
│   │   ├── lstm/                # LSTM model configuration and scripts
│   │   └── transformer/         # Transformer model configuration and scripts
│   ├── postprocessing/          # SSI calculation and reservoir operations
│   └── utils/                   # Utility functions and visualization
│
├── notebooks/                   # Jupyter notebooks for analysis
├── configs/                     # Configuration files
├── results/                     # Model outputs and analysis results
├── docs/                        # Documentation
└── tests/                       # Unit tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LSTM-SSI.git
   cd LSTM-SSI
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate watersoft-ml
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install neuralhydrology framework**
   ```bash
   git clone https://github.com/neuralhydrology/neuralhydrology.git
   cd neuralhydrology
   pip install -e .
   cd ..
   ```

### Basic Usage

1. **Data Preprocessing**
   ```bash
   python src/preprocessing/data_extraction.py
   python src/preprocessing/attribute_processing.py
   ```

2. **Train LSTM Model**
   ```bash
   python -m neuralhydrology.nh_run train --config-file configs/lstm_config.yml
   ```

3. **Train Transformer Model**
   ```bash
   python -m neuralhydrology.nh_run train --config-file configs/transformer_config.yml
   ```

4. **Evaluate Models**
   ```bash
   python -m neuralhydrology.nh_run evaluate --run-dir results/model_outputs/[run_directory]
   ```

5. **Run Reservoir Operations Analysis**
   ```bash
   python src/postprocessing/reservoir_operations.py
   ```

## Dataset Description

### USGS Gauge Stations (22 stations)

The project uses streamflow data from 22 USGS gauge stations, primarily located in the Upper Colorado River Basin:

| Station ID | Location | Drainage Area (km²) | Elevation (m) |
|------------|----------|-------------------|---------------|
| 6221400 | North Fork Shoshone River | 227.88 | 3336.8 |
| 6224000 | South Fork Shoshone River | 484.86 | 3116.82 |
| 9107000 | Taylor River at Taylor Park | 331.62 | 3300.29 |
| ... | ... | ... | ... |

### Static Attributes (49 features)

- **Climatic**: Precipitation, temperature, aridity, seasonality
- **Hydrologic**: Runoff ratio, baseflow index, streamflow elasticity
- **Physiographic**: Elevation, slope, drainage area, land cover
- **Soil**: Depth, porosity, conductivity, texture fractions
- **Geologic**: Permeability, dominant rock types

### Dynamic Inputs

- **Precipitation** (prcp): Daily precipitation [mm/day]
- **Maximum Temperature** (tmax): Daily maximum temperature [°C]

### Target Variable

- **Streamflow**: Daily streamflow [m³/s]

## Model Architectures

### LSTM Model (Entity-Aware LSTM)

- **Architecture**: Entity-Aware LSTM with static attribute integration
- **Hidden Size**: 128 units
- **Sequence Length**: 30 days
- **Features**: 
  - Forget gate bias initialization: 3.0
  - Output dropout: 0.2
  - Gradient clipping: 1.0

### Transformer Model

- **Architecture**: Multi-head attention with positional encoding
- **Attention Heads**: 8
- **Layers**: 4 transformer blocks
- **Feedforward Dimension**: 512
- **Features**:
  - Positional encoding: Summation
  - Dropout: 0.1
  - Static/dynamic embedding layers

## Performance Metrics

Models are evaluated using multiple hydrological performance metrics:

- **Nash-Sutcliffe Efficiency (NSE)**: Model efficiency relative to mean flow
- **Kling-Gupta Efficiency (KGE)**: Decomposed efficiency metric
- **Root Mean Square Error (RMSE)**: Absolute error magnitude
- **Pearson Correlation (r)**: Linear correlation coefficient
- **Mean Square Error (MSE)**: Squared error metric

## Reservoir Operations

### Taylor Park Reservoir

- **Capacity**: 106,250 acre-feet
- **Purpose**: Irrigation water supply and flood control
- **Operation**: Adaptive releases based on SSI and storage conditions

### Standardized Streamflow Index (SSI)

SSI categories for drought assessment:

| SSI Range | Category | Description | Frequency |
|-----------|----------|-------------|-----------|
| ≥ 1.5 | Extremely Wet | Exceptional high flow | ~6.7% |
| 1.0 to 1.49 | Moderately Wet | Above-normal flow | ~9.2% |
| 0.5 to 0.99 | Mildly Wet | Slightly above-normal | ~15.0% |
| -0.49 to 0.49 | Normal | Near-normal conditions | ~38.2% |
| -0.99 to -0.5 | Mildly Dry | Slightly below-normal | ~15.0% |
| -1.49 to -1.0 | Moderately Dry | Below-normal, early drought | ~9.2% |
| -1.99 to -1.5 | Severely Dry | Significant drought | ~4.4% |
| ≤ -2.0 | Extremely Dry | Extreme drought | ~2.3% |

### Adaptive Release Strategy

The reservoir operations use empirically-derived adjustment factors:

```
Adaptive Release = Base Release × Combined Factor

Combined Factor = (w_inflow × SSI_factor) + (w_storage × Storage_factor)
```

Where:
- `w_inflow`: Weight for inflow/SSI conditions
- `w_storage`: Weight for storage conditions
- Factors derived from historical operations data

## 📈 Results

### Model Performance (Test Period: 2020-2023)

| Model | NSE | KGE | RMSE | Pearson-r |
|-------|-----|-----|------|-----------|
| LSTM | 0.XX | 0.XX | X.XX | 0.XX |
| Transformer | 0.XX | 0.XX | X.XX | 0.XX |

*Note: Results to be updated after model training completion*

### Reservoir Operations Impact

- **Drought Response**: X% reduction in releases during severe drought periods
- **Storage Efficiency**: X% improvement in storage utilization
- **Operational Flexibility**: Adaptive responses to varying hydrological conditions

## 🔧 Configuration

### Model Configuration Files

- `configs/lstm_config.yml`: LSTM model hyperparameters and training settings
- `configs/transformer_config.yml`: Transformer model configuration
- `configs/reservoir_params/`: Reservoir operational parameters

### Key Parameters

```yaml
# Training Configuration
train_start_date: '01/01/1980'
train_end_date: '31/12/2010'
validation_start_date: '01/01/2011'
validation_end_date: '31/12/2019'
test_start_date: '01/01/2020'
test_end_date: '31/12/2023'

# Model Parameters
seq_length: 30
batch_size: 256
learning_rate: 1e-3
epochs: 50
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


