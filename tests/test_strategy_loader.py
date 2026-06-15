"""
Unit tests for strategy_loader module
"""
import pytest
import yaml
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.strategy_loader import load_strategy_config, load_all_strategies, load_strategies_by_name

@pytest.fixture
def temp_strategies_dir(tmp_path):
    d = tmp_path / "strategies"
    d.mkdir()
    return d

def test_load_strategy_config_valid_individual(temp_strategies_dir):
    config = {
        'name': 'Test Strategy',
        'type': 'individual',
        'indicator': 'rsi',
        'params_grid': {'period': [14]}
    }
    f = temp_strategies_dir / 'test_ind.yaml'
    with open(f, 'w') as file:
        yaml.dump(config, file)
        
    loaded = load_strategy_config(str(f))
    assert loaded['name'] == 'Test Strategy'
    assert loaded['type'] == 'individual'

def test_load_strategy_config_valid_combo(temp_strategies_dir):
    config = {
        'name': 'Test Combo',
        'type': 'combo',
        'indicators': [{'name': 'rsi'}],
        'combination_methods': ['AND']
    }
    f = temp_strategies_dir / 'test_combo.yaml'
    with open(f, 'w') as file:
        yaml.dump(config, file)
        
    loaded = load_strategy_config(str(f))
    assert loaded['name'] == 'Test Combo'
    assert loaded['type'] == 'combo'

def test_load_strategy_config_missing_fields(temp_strategies_dir):
    config = {
        'type': 'individual',
        'indicator': 'rsi'
        # missing name and params_grid
    }
    f = temp_strategies_dir / 'test_invalid.yaml'
    with open(f, 'w') as file:
        yaml.dump(config, file)
        
    # Should print error and return None
    loaded = load_strategy_config(str(f))
    assert loaded is None

def test_load_all_strategies(temp_strategies_dir):
    # Valid
    config1 = {
        'name': 'Strat 1', 'type': 'individual', 'indicator': 'rsi', 'params_grid': {}
    }
    # Invalid
    config2 = {
        'type': 'individual' 
    }
    
    with open(temp_strategies_dir / 's1.yaml', 'w') as f:
        yaml.dump(config1, f)
    with open(temp_strategies_dir / 's2.yaml', 'w') as f:
        yaml.dump(config2, f)
        
    strategies = load_all_strategies(str(temp_strategies_dir))
    assert len(strategies) == 1
    assert strategies[0]['name'] == 'Strat 1'

def test_load_strategies_by_name(temp_strategies_dir):
    config = {
        'name': 'Target Strategy', 'type': 'individual', 'indicator': 'rsi', 'params_grid': {}
    }
    # Name the file same as what we search for
    with open(temp_strategies_dir / 'target_strategy.yaml', 'w') as f:
        yaml.dump(config, f)
        
    strategies = load_strategies_by_name(['target_strategy'], str(temp_strategies_dir))
    assert len(strategies) == 1
    assert strategies[0]['name'] == 'Target Strategy'
