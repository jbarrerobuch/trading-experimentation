"""
Configuration Module for gamma neutral strategy.

This module provides configuration management functionality.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for the gamma neutral strategy.
    
    Provides default configuration and methods to load/save custom configs.
    """
    
    DEFAULT_CONFIG = {
        "strategy": {
            "target_gamma": 0.0,
            "gamma_tolerance": 0.1,
            "delta_tolerance": 0.05,
            "rebalance_frequency": 3600,  # seconds
        },
        "risk": {
            "max_portfolio_delta": 1.0,
            "max_portfolio_gamma": 0.5,
            "max_position_size": 100.0,
            "max_notional_exposure": 1000000.0,
            "var_confidence": 0.95,
        },
        "trading": {
            "risk_free_rate": 0.0,
            "transaction_cost": 0.0005,
            "min_rebalance_threshold": 0.1,
        },
        "market": {
            "default_volatility": 0.5,
            "volatility_floor": 0.1,
            "volatility_cap": 2.0,
        },
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional configuration dictionary. If not provided,
                        uses default configuration.
        """
        if config_dict is None:
            self.config = self.DEFAULT_CONFIG.copy()
        else:
            self.config = self._merge_configs(self.DEFAULT_CONFIG, config_dict)
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """
        Merge custom config with default config.
        
        Args:
            default: Default configuration
            custom: Custom configuration
        
        Returns:
            Merged configuration
        """
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., "strategy.target_gamma")
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            >>> config.get("strategy.target_gamma")
            0.0
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., "strategy.target_gamma")
            value: Value to set
        
        Example:
            >>> config.set("strategy.target_gamma", 0.1)
        """
        keys = key_path.split(".")
        config_dict = self.config
        
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
        
        config_dict[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
        
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the configuration.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Validate strategy parameters
        if self.get("strategy.gamma_tolerance") <= 0:
            errors.append("strategy.gamma_tolerance must be positive")
        
        if self.get("strategy.delta_tolerance") <= 0:
            errors.append("strategy.delta_tolerance must be positive")
        
        if self.get("strategy.rebalance_frequency") <= 0:
            errors.append("strategy.rebalance_frequency must be positive")
        
        # Validate risk parameters
        if self.get("risk.max_position_size") <= 0:
            errors.append("risk.max_position_size must be positive")
        
        if self.get("risk.max_notional_exposure") <= 0:
            errors.append("risk.max_notional_exposure must be positive")
        
        var_confidence = self.get("risk.var_confidence")
        if not (0 < var_confidence < 1):
            errors.append("risk.var_confidence must be between 0 and 1")
        
        # Validate trading parameters
        if self.get("trading.transaction_cost") < 0:
            errors.append("trading.transaction_cost must be non-negative")
        
        # Validate market parameters
        vol_floor = self.get("market.volatility_floor")
        vol_cap = self.get("market.volatility_cap")
        
        if vol_floor <= 0:
            errors.append("market.volatility_floor must be positive")
        
        if vol_cap <= vol_floor:
            errors.append("market.volatility_cap must be greater than volatility_floor")
        
        # Warnings
        if self.get("trading.transaction_cost") > 0.01:
            warnings.append("High transaction cost (>1%) may impact profitability")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config({json.dumps(self.config, indent=2)})"
