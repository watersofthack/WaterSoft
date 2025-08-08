"""
Basin attribute processing module for WaterSoft Hydrological ML.

This module provides functions for processing, normalizing, and preparing
basin static attributes for use in neural network models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_basin_attributes(
    attributes_file: str = "data/raw/attributes/attributes.csv"
) -> pd.DataFrame:
    """
    Load basin attributes from CSV file.
    
    Args:
        attributes_file: Path to the attributes CSV file
        
    Returns:
        DataFrame with basin attributes
    """
    try:
        df = pd.read_csv(attributes_file, index_col='basin_id')
        logger.info(f"Loaded attributes for {len(df)} basins with {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading attributes: {str(e)}")
        return pd.DataFrame()


def process_basin_attributes(
    attributes_df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Process basin attributes by handling missing values and categorical variables.
    
    Args:
        attributes_df: DataFrame with raw basin attributes
        categorical_columns: List of categorical column names
        drop_columns: List of columns to drop
        
    Returns:
        Processed DataFrame
    """
    df = attributes_df.copy()
    
    # Default categorical columns
    if categorical_columns is None:
        categorical_columns = ['huc_02', 'dom_land_cover']
    
    # Default columns to drop (if any)
    if drop_columns is None:
        drop_columns = []
    
    logger.info(f"Processing {len(df)} basins with {len(df.columns)} attributes")
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        logger.info(f"Dropped {len(drop_columns)} columns")
    
    # Handle missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found missing values in {(missing_counts > 0).sum()} columns")
        
        # Fill missing values with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {missing_counts[col]} missing values in {col} with median: {median_val:.3f}")
    
    # Handle categorical variables
    for col in categorical_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Convert categorical to numerical using label encoding
                unique_values = df[col].unique()
                value_map = {val: i for i, val in enumerate(unique_values)}
                df[col] = df[col].map(value_map)
                logger.info(f"Encoded categorical column {col}: {len(unique_values)} categories")
    
    logger.info(f"Processed attributes: {len(df)} basins, {len(df.columns)} features")
    return df


def normalize_attributes(
    attributes_df: pd.DataFrame,
    method: str = "standard",
    save_scaler: bool = True,
    scaler_path: str = "data/processed/attribute_scaler.pkl"
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Normalize basin attributes using specified method.
    
    Args:
        attributes_df: DataFrame with basin attributes
        method: Normalization method ('standard' or 'minmax')
        save_scaler: Whether to save the fitted scaler
        scaler_path: Path to save the scaler
        
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    logger.info(f"Normalizing attributes using {method} scaling")
    
    # Initialize scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    normalized_values = scaler.fit_transform(attributes_df.values)
    normalized_df = pd.DataFrame(
        normalized_values,
        index=attributes_df.index,
        columns=attributes_df.columns
    )
    
    # Save scaler if requested
    if save_scaler:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
    
    logger.info(f"Normalized {len(normalized_df)} basins with {len(normalized_df.columns)} features")
    return normalized_df, scaler


def create_attribute_embeddings(
    attributes_df: pd.DataFrame,
    n_components: int = 10,
    method: str = "pca"
) -> Tuple[pd.DataFrame, PCA]:
    """
    Create lower-dimensional embeddings of basin attributes.
    
    Args:
        attributes_df: DataFrame with normalized basin attributes
        n_components: Number of embedding dimensions
        method: Dimensionality reduction method ('pca')
        
    Returns:
        Tuple of (embeddings DataFrame, fitted reducer)
    """
    logger.info(f"Creating {n_components}-dimensional embeddings using {method}")
    
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    # Fit and transform
    embeddings = reducer.fit_transform(attributes_df.values)
    
    # Create DataFrame
    embedding_columns = [f"embed_{i+1}" for i in range(n_components)]
    embeddings_df = pd.DataFrame(
        embeddings,
        index=attributes_df.index,
        columns=embedding_columns
    )
    
    # Log explained variance for PCA
    if method == "pca":
        explained_var = reducer.explained_variance_ratio_.sum()
        logger.info(f"PCA embeddings explain {explained_var:.2%} of total variance")
    
    logger.info(f"Created embeddings: {len(embeddings_df)} basins, {n_components} dimensions")
    return embeddings_df, reducer


def analyze_attribute_correlations(
    attributes_df: pd.DataFrame,
    threshold: float = 0.8,
    save_plot: bool = True,
    plot_path: str = "results/figures/attribute_correlations.png"
) -> pd.DataFrame:
    """
    Analyze correlations between basin attributes.
    
    Args:
        attributes_df: DataFrame with basin attributes
        threshold: Correlation threshold for identifying highly correlated pairs
        save_plot: Whether to save correlation heatmap
        plot_path: Path to save the plot
        
    Returns:
        DataFrame with highly correlated attribute pairs
    """
    logger.info("Analyzing attribute correlations...")
    
    # Calculate correlation matrix
    corr_matrix = attributes_df.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    if len(high_corr_df) > 0:
        logger.warning(f"Found {len(high_corr_df)} highly correlated pairs (|r| >= {threshold})")
        logger.info("Top 5 correlations:")
        top_corr = high_corr_df.reindex(high_corr_df['correlation'].abs().sort_values(ascending=False).index)
        for _, row in top_corr.head().iterrows():
            logger.info(f"  {row['feature_1']} - {row['feature_2']}: {row['correlation']:.3f}")
    else:
        logger.info(f"No highly correlated pairs found (threshold: {threshold})")
    
    # Save correlation heatmap
    if save_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.1)
            plt.title('Basin Attribute Correlations')
            plt.tight_layout()
            
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved correlation heatmap to {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping plot")
    
    return high_corr_df


def create_attribute_summary(
    attributes_df: pd.DataFrame,
    output_file: str = "results/reports/attribute_summary.txt"
) -> Dict:
    """
    Create a summary report of basin attributes.
    
    Args:
        attributes_df: DataFrame with basin attributes
        output_file: Path to save the summary report
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Creating attribute summary...")
    
    summary = {
        'n_basins': len(attributes_df),
        'n_attributes': len(attributes_df.columns),
        'missing_values': attributes_df.isnull().sum().sum(),
        'attribute_stats': attributes_df.describe()
    }
    
    # Create text report
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Basin Attributes Summary Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of basins: {summary['n_basins']}\n")
        f.write(f"Number of attributes: {summary['n_attributes']}\n")
        f.write(f"Missing values: {summary['missing_values']}\n\n")
        
        f.write("Attribute Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(summary['attribute_stats'].to_string())
        f.write("\n\n")
        
        f.write("Attribute List:\n")
        f.write("-" * 15 + "\n")
        for i, col in enumerate(attributes_df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
    
    logger.info(f"Saved attribute summary to {output_file}")
    return summary


def prepare_attributes_for_modeling(
    attributes_file: str = "data/raw/attributes/attributes.csv",
    output_dir: str = "data/processed",
    normalize: bool = True,
    create_embeddings: bool = True,
    embedding_dims: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Complete pipeline for preparing basin attributes for modeling.
    
    Args:
        attributes_file: Path to raw attributes file
        output_dir: Directory to save processed data
        normalize: Whether to normalize attributes
        create_embeddings: Whether to create embeddings
        embedding_dims: Number of embedding dimensions
        
    Returns:
        Dictionary with processed DataFrames
    """
    logger.info("Starting complete attribute processing pipeline...")
    
    # Load raw attributes
    raw_attributes = load_basin_attributes(attributes_file)
    if raw_attributes.empty:
        return {}
    
    # Process attributes
    processed_attributes = process_basin_attributes(raw_attributes)
    
    results = {'raw': raw_attributes, 'processed': processed_attributes}
    
    # Normalize if requested
    if normalize:
        normalized_attributes, scaler = normalize_attributes(
            processed_attributes,
            scaler_path=f"{output_dir}/attribute_scaler.pkl"
        )
        results['normalized'] = normalized_attributes
        
        # Create embeddings if requested
        if create_embeddings:
            embeddings, reducer = create_attribute_embeddings(
                normalized_attributes,
                n_components=embedding_dims
            )
            results['embeddings'] = embeddings
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, df in results.items():
        if name != 'raw':  # Don't save raw data again
            output_file = f"{output_dir}/attributes_{name}.csv"
            df.to_csv(output_file)
            logger.info(f"Saved {name} attributes to {output_file}")
    
    # Create analysis reports
    analyze_attribute_correlations(processed_attributes)
    create_attribute_summary(processed_attributes)
    
    logger.info("Attribute processing pipeline completed!")
    return results


if __name__ == "__main__":
    # Example usage
    results = prepare_attributes_for_modeling(
        attributes_file="data/raw/attributes/attributes.csv",
        output_dir="data/processed",
        normalize=True,
        create_embeddings=True,
        embedding_dims=10
    )
    
    print("Attribute processing completed!")
    for name, df in results.items():
        print(f"{name}: {df.shape}")
