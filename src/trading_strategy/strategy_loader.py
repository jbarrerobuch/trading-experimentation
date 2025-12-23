"""
Módulo de carga de configuraciones de estrategias
Carga estrategias desde archivos YAML en la carpeta strategies/
"""

import os
import glob
import yaml


def load_strategy_config(strategy_file):
    """
    Carga una configuración de estrategia desde un archivo YAML
    
    Parameters:
    -----------
    strategy_file : str
        Path al archivo YAML de configuración
    
    Returns:
    --------
    dict : Configuración de la estrategia
    """
    try:
        with open(strategy_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validar campos requeridos
        if 'name' not in config:
            raise ValueError(f"Falta campo 'name' en {strategy_file}")
        
        if 'type' not in config:
            config['type'] = 'individual'  # Default
        
        if config['type'] == 'individual':
            if 'indicator' not in config:
                raise ValueError(f"Estrategia individual debe tener campo 'indicator' en {strategy_file}")
            if 'params_grid' not in config:
                raise ValueError(f"Estrategia individual debe tener campo 'params_grid' en {strategy_file}")
        
        elif config['type'] == 'combo':
            if 'indicators' not in config:
                raise ValueError(f"Estrategia combo debe tener campo 'indicators' en {strategy_file}")
            if 'combination_methods' not in config:
                config['combination_methods'] = ['AND']  # Default
            if 'position_types' not in config:
                config['position_types'] = ['both']  # Default
        
        return config
    
    except yaml.YAMLError as e:
        print(f"❌ Error parseando YAML en {strategy_file}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error cargando {strategy_file}: {e}")
        return None


def load_all_strategies(strategies_dir=None):
    """
    Carga todas las estrategias desde la carpeta strategies/
    
    Parameters:
    -----------
    strategies_dir : str or Path, optional
        Path al directorio con archivos YAML de estrategias
        Default: None (usa get_strategies_dir())
    
    Returns:
    --------
    list : Lista de configuraciones de estrategias
    """
    from .utils.paths import get_strategies_dir, get_project_root
    
    if strategies_dir is None:
        strategies_dir = get_strategies_dir()
    elif not os.path.isabs(strategies_dir):
        # Si es relativo, intentar resolverlo desde el root
        strategies_dir = os.path.join(get_project_root(), strategies_dir)
    
    if not os.path.exists(strategies_dir):
        print(f"⚠️  Directorio de estrategias no existe: {strategies_dir}")
        return []
    
    # Buscar todos los archivos YAML
    yaml_files = glob.glob(os.path.join(strategies_dir, '*.yaml')) + \
                 glob.glob(os.path.join(strategies_dir, '*.yml'))
    
    if not yaml_files:
        print(f"⚠️  No se encontraron archivos YAML en: {strategies_dir}")
        return []
    
    print(f"📂 Cargando estrategias desde: {strategies_dir}")
    
    strategies = []
    for yaml_file in sorted(yaml_files):
        filename = os.path.basename(yaml_file)
        config = load_strategy_config(yaml_file)
        
        if config:
            strategy_type = config.get('type', 'individual')
            print(f"  ✓ {filename} → {config['name']} ({strategy_type})")
            strategies.append(config)
        else:
            print(f"  ✗ {filename} → Error al cargar")
    
    print(f"✓ {len(strategies)} estrategias cargadas")
    return strategies


def load_strategies_by_name(strategy_names, strategies_dir=None):
    """
    Carga estrategias específicas por nombre de archivo (sin extensión)
    
    Parameters:
    -----------
    strategy_names : list of str
        Lista de nombres de archivos (sin .yaml)
        Ejemplo: ['rsi_optimization', 'macd_optimization']
    strategies_dir : str or Path, optional
        Path al directorio con archivos YAML
        Default: None (usa get_strategies_dir())
    
    Returns:
    --------
    list : Lista de configuraciones de estrategias
    """
    from .utils.paths import get_strategies_dir, get_project_root
    
    if strategies_dir is None:
        strategies_dir = get_strategies_dir()
    elif not os.path.isabs(strategies_dir):
        strategies_dir = os.path.join(get_project_root(), strategies_dir)
    
    if not os.path.exists(strategies_dir):
        print(f"⚠️  Directorio de estrategias no existe: {strategies_dir}")
        return []
    
    print(f"📂 Cargando estrategias específicas desde: {strategies_dir}")
    
    strategies = []
    for name in strategy_names:
        # Intentar con .yaml y .yml
        yaml_file = os.path.join(strategies_dir, f"{name}.yaml")
        if not os.path.exists(yaml_file):
            yaml_file = os.path.join(strategies_dir, f"{name}.yml")
        
        if not os.path.exists(yaml_file):
            print(f"  ✗ {name} → Archivo no encontrado")
            continue
        
        config = load_strategy_config(yaml_file)
        if config:
            strategy_type = config.get('type', 'individual')
            print(f"  ✓ {name}.yaml → {config['name']} ({strategy_type})")
            strategies.append(config)
        else:
            print(f"  ✗ {name}.yaml → Error al cargar")
    
    print(f"✓ {len(strategies)} estrategias cargadas")
    return strategies


def get_strategy_info(strategies_dir=None):
    """
    Muestra información sobre las estrategias disponibles
    
    Parameters:
    -----------
    strategies_dir : str or Path, optional
        Path al directorio con archivos YAML
        Default: None (usa get_strategies_dir())
    
    Returns:
    --------
    dict : Información de estrategias disponibles
    """
    strategies = load_all_strategies(strategies_dir)
    
    info = {
        'total': len(strategies),
        'individual': [],
        'combo': []
    }
    
    for strategy in strategies:
        strategy_type = strategy.get('type', 'individual')
        strategy_summary = {
            'name': strategy['name'],
            'file': f"{strategy['name'].lower().replace(' ', '_')}.yaml"
        }
        
        if strategy_type == 'individual':
            strategy_summary['indicator'] = strategy.get('indicator', 'unknown')
            info['individual'].append(strategy_summary)
        else:
            strategy_summary['n_indicators'] = len(strategy.get('indicators', []))
            info['combo'].append(strategy_summary)
    
    return info
