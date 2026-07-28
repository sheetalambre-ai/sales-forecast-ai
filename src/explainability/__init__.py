"""
Explainability package.

Provides utilities for interpreting trained machine learning models,
including feature importance analysis and SHAP-based explanations.
"""

from .feature_importance import (
    FeatureImportanceAnalyzer,
)

try:
    from .shap_values import (
        SHAPAnalyzer,
    )
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    SHAPAnalyzer = None

__all__ = [
    "FeatureImportanceAnalyzer",
]

if HAS_SHAP:
    __all__.append("SHAPAnalyzer")