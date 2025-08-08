# Changelog

All notable changes to the WaterSoft Hydrological ML project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and repository setup
- Comprehensive README with project overview and usage instructions
- LSTM and Transformer model configurations
- Data preprocessing modules for CAMELS dataset
- SSI-based adaptive reservoir operations framework
- Jupyter notebooks for data analysis and model training
- Complete development environment setup (conda, pip)
- CI/CD pipeline with GitHub Actions
- Testing framework with pytest
- Code quality tools (Black, isort, flake8)
- Documentation structure

### Changed
- Reorganized original folder structure into GitHub-ready format
- Moved notebooks to centralized notebooks/ directory
- Consolidated configuration files in configs/ directory
- Standardized data directory structure

## [1.0.0] - 2025-01-08

### Added
- **Core Framework**: Complete machine learning framework for hydrological modeling
- **Dual Model Architecture**: LSTM and Transformer implementations for streamflow prediction
- **Data Processing**: Comprehensive preprocessing pipeline for CAMELS dataset
- **Reservoir Operations**: SSI-based adaptive management for Taylor Park Reservoir
- **Multi-Station Analysis**: Support for 22 USGS gauge stations
- **Performance Metrics**: NSE, KGE, RMSE, Pearson correlation evaluation
- **Documentation**: Extensive README, contributing guidelines, and API documentation
- **Testing**: Unit test framework and CI/CD pipeline
- **Reproducibility**: Standardized configurations and environment management

### Features
- **Streamflow Prediction**: 
  - Entity-Aware LSTM with static attribute integration
  - Multi-head attention Transformer with positional encoding
  - 44-year training dataset (1980-2023)
  - Multiple performance metrics evaluation

- **Drought Assessment**:
  - Standardized Streamflow Index (SSI) calculation
  - Monthly drought categorization
  - Historical drought analysis and visualization

- **Reservoir Management**:
  - Empirically-derived adjustment factors
  - Adaptive release strategies based on SSI and storage
  - Taylor Park Reservoir case study
  - Operational parameter optimization

- **Data Infrastructure**:
  - CAMELS dataset integration
  - 49 static basin attributes
  - Quality-controlled time series data
  - Automated preprocessing pipeline

### Technical Specifications
- **Python**: 3.8+ compatibility
- **Deep Learning**: PyTorch-based neural networks
- **Hydrology**: neuralhydrology framework integration
- **Data Science**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Geospatial**: geopandas, pygeohydro
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions workflow

### Repository Structure
```
LSTM-SSI/
├── README.md                 # Project overview and documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── setup.py                  # Package installation
├── .gitignore               # Git ignore rules
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # This file
├── data/                    # Data directory
├── src/                     # Source code
├── notebooks/               # Analysis notebooks
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── results/                 # Model outputs
└── .github/                 # GitHub workflows
```

### Acknowledgments
- Neural Hydrology team for the deep learning framework
- CAMELS dataset contributors for comprehensive hydrometeorological data
- USGS for streamflow and basin attribute data
- Water Software Hackathon 2025 for project motivation

---

**Note**: This project was developed for the Water Software Hackathon 2025 and represents a comprehensive approach to machine learning in hydrology with practical applications for water resource management.
