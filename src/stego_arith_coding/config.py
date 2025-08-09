"""
Configuration management for arithmetic steganography.

This module contains the configuration class that holds all parameters
used throughout the arithmetic steganography implementation.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ArithmeticSteganographyConfig:
    """Configuration object for arithmetic steganography parameters."""
    
    # Model configuration
    model: str = "gpt-3.5-turbo-1106"
    
    # API parameters for token probability requests
    api_params: Dict[str, Any] = field(default_factory=lambda: {
        "seed": 42,
        "max_completion_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 2.0,
    })
    
    # Additional temperature scaling for probability flattening
    extra_temperature: float = 1.5
    
    # Cache configuration
    cache_dir: Optional[Path] = None
    
    # API retry configuration
    max_retries: int = 3
    
    # Encoding configuration
    precision: int = 16
    topk_limit: int = 50
    min_prob_threshold_divisor: float = 1.0  # Used in cur_threshold calculation
    min_interval_width: int = 2  # Minimum width for token intervals
    
    def __post_init__(self):
        """Initialize cache directory using tempfile if not provided."""
        if self.cache_dir is None:
            # Create cache directory in temporary directory
            temp_dir = Path(tempfile.gettempdir())
            self.cache_dir = temp_dir / "stego_arith_coding_cache" / "openai"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_api_params_with_model(self) -> Dict[str, Any]:
        """Get API parameters with model included."""
        params = self.api_params.copy()
        params["model"] = self.model
        return params
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArithmeticSteganographyConfig":
        """Create config from dictionary, handling nested structures."""
        # Extract relevant sections from config
        config = cls()
        
        # Update with provided values, handling nested dicts
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config