"""
Data handling module for GMSIFN

Includes:
- MetaDataset: Small-sample episode generation
- CategorySplitter: Cross-category task splitting
- get_meta_loaders: Convenience function for creating data loaders
"""

from .meta_dataset import MetaDataset, CategorySplitter, get_meta_loaders

__all__ = ['MetaDataset', 'CategorySplitter', 'get_meta_loaders']
