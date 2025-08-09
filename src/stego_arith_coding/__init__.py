"""
Arithmetic coding steganography library for hiding information in text.

This library implements steganographic encoding and decoding using arithmetic coding
to hide secret messages within LLM-generated text by influencing token selection.
"""

from .core import ArithmeticSteganography
from .config import ArithmeticSteganographyConfig

__version__ = "0.1.0"
__all__ = ["ArithmeticSteganography", "ArithmeticSteganographyConfig"]