"""
Basic tests to ensure the package structure is working correctly.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_package_imports():
    """Test that main package modules can be imported."""
    try:
        import src
        import src.preprocessing
        import src.postprocessing
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import package modules: {e}")


def test_data_directory_exists():
    """Test that data directory structure exists."""
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists(), "Data directory should exist"
    assert (data_dir / "raw").exists(), "Raw data directory should exist"
    assert (data_dir / "processed").exists(), "Processed data directory should exist"


def test_config_directory_exists():
    """Test that config directory and files exist."""
    config_dir = Path(__file__).parent.parent / "configs"
    assert config_dir.exists(), "Config directory should exist"
    
    # Check for key config files
    lstm_config = config_dir / "lstm_config.yml"
    transformer_config = config_dir / "transformer_config.yml"
    
    assert lstm_config.exists(), "LSTM config file should exist"
    assert transformer_config.exists(), "Transformer config file should exist"


def test_notebooks_directory_exists():
    """Test that notebooks directory exists and contains notebooks."""
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    assert notebooks_dir.exists(), "Notebooks directory should exist"
    
    # Check if there are any notebook files
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    assert len(notebook_files) > 0, "Should contain at least one notebook file"


def test_essential_files_exist():
    """Test that essential project files exist."""
    project_root = Path(__file__).parent.parent
    
    essential_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        "environment.yml",
        ".gitignore",
        "setup.py",
        "CONTRIBUTING.md"
    ]
    
    for file_name in essential_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"{file_name} should exist in project root"


if __name__ == "__main__":
    pytest.main([__file__])
