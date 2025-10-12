# GammaNeutral

A gamma neutral trading strategy for crypto markets using options and perpetual futures.

## Overview

GammaNeutral is a sophisticated trading strategy designed for cryptocurrency markets that maintains a gamma-neutral portfolio by combining options and perpetual futures. The strategy aims to profit from volatility while managing directional risk through dynamic hedging.

## Features

- **Options Greeks Calculator**: Calculate delta, gamma, vega, theta, and rho for crypto options
- **Perpetual Futures Hedging**: Dynamic delta hedging using perpetual futures contracts
- **Portfolio Management**: Track and manage multiple options and futures positions
- **Risk Management**: Built-in risk controls and position limits
- **Automatic Rebalancing**: Monitor and rebalance portfolio to maintain gamma neutrality
- **Comprehensive Testing**: Full test suite for core components

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jbarrerobuch/GammaNeutral.git
cd GammaNeutral
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from datetime import datetime, timedelta
from gamma_neutral import GammaNeutralStrategy

# Initialize the strategy
strategy = GammaNeutralStrategy(
    target_gamma=0.0,
    gamma_tolerance=0.1,
    delta_tolerance=0.05
)

# Add an options position
expiry_date = datetime.now() + timedelta(days=30)
strategy.add_option_position(
    quantity=10.0,
    entry_price=2500.0,
    strike_price=52000.0,
    expiry_date=expiry_date,
    option_type="call",
    volatility=0.8
)

# Calculate portfolio Greeks
spot_price = 50000.0
greeks = strategy.calculate_portfolio_greeks(spot_price)
print(f"Portfolio Delta: {greeks['delta']:.4f}")
print(f"Portfolio Gamma: {greeks['gamma']:.4f}")

# Rebalance if needed
if strategy.check_rebalance_needed(spot_price)['needs_any_rebalance']:
    result = strategy.rebalance(spot_price)
    print(f"Rebalanced at price: {result['spot_price']}")
```

## Strategy Components

### 1. Options Greeks Calculator (`gamma_neutral.core.greeks`)

Calculates the Greek values for options positions using the Black-Scholes model:

- **Delta**: Sensitivity to underlying price changes
- **Gamma**: Rate of change of delta
- **Vega**: Sensitivity to volatility changes
- **Theta**: Time decay
- **Rho**: Sensitivity to interest rate changes

### 2. Perpetual Futures Hedger (`gamma_neutral.core.hedging`)

Manages hedging using perpetual futures contracts:

- Dynamic delta hedging
- Transaction cost calculation
- Funding rate consideration
- Position tracking and P&L calculation

### 3. Portfolio Tracker (`gamma_neutral.core.portfolio`)

Tracks all positions in the portfolio:

- Options positions management
- Futures positions management
- Portfolio valuation
- Historical tracking

### 4. Gamma Neutral Strategy (`gamma_neutral.strategies.gamma_neutral`)

Main strategy implementation:

- Maintains gamma neutrality
- Automatic rebalancing
- Risk monitoring
- Portfolio status reporting

### 5. Risk Manager (`gamma_neutral.utils.risk_management`)

Comprehensive risk management:

- Position sizing
- Exposure limits
- VaR calculation
- Expected shortfall
- Maximum drawdown tracking
- Sharpe ratio calculation

### 6. Configuration (`gamma_neutral.utils.config`)

Flexible configuration management:

- Default configurations
- Custom configuration loading
- Configuration validation
- JSON import/export

## Usage Examples

See the `examples/basic_usage.py` file for a complete example demonstrating:

1. Strategy initialization
2. Adding options positions
3. Calculating portfolio Greeks
4. Checking rebalancing needs
5. Performing rebalancing
6. Monitoring portfolio status
7. Handling price movements

Run the example:
```bash
python examples/basic_usage.py
```

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

Run specific test files:
```bash
python -m unittest tests.test_greeks
python -m unittest tests.test_strategy
```

## Project Structure

```
GammaNeutral/
├── gamma_neutral/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── greeks.py          # Options Greeks calculator
│   │   ├── hedging.py         # Perpetual futures hedging
│   │   └── portfolio.py       # Portfolio tracker
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── gamma_neutral.py   # Main strategy implementation
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── risk_management.py # Risk management tools
├── tests/
│   ├── __init__.py
│   ├── test_greeks.py
│   └── test_strategy.py
├── examples/
│   └── basic_usage.py
├── requirements.txt
└── README.md
```

## Strategy Explanation

### Gamma Neutrality

Gamma measures how much an option's delta changes when the underlying asset price moves. A gamma-neutral portfolio has minimal sensitivity to price movements, which is achieved by:

1. **Balancing Options Positions**: Combining calls and puts to offset gamma exposure
2. **Dynamic Delta Hedging**: Using perpetual futures to hedge the delta exposure
3. **Regular Rebalancing**: Adjusting positions as market conditions change

### Benefits

- **Reduced Directional Risk**: Less exposure to price movements
- **Volatility Trading**: Profit from changes in volatility
- **Theta Collection**: Earn from time decay of options
- **Flexibility**: Adapt to different market conditions

### Key Considerations

- **Transaction Costs**: Frequent rebalancing incurs costs
- **Funding Rates**: Perpetual futures have funding costs/benefits
- **Slippage**: Market impact on large positions
- **Volatility Risk**: Changes in implied volatility affect profitability

## Configuration

Create a custom configuration file:

```python
from gamma_neutral.utils.config import Config

config = Config({
    "strategy": {
        "target_gamma": 0.0,
        "gamma_tolerance": 0.15,
        "delta_tolerance": 0.1
    },
    "risk": {
        "max_position_size": 50.0,
        "max_notional_exposure": 500000.0
    }
})

# Save configuration
config.save("my_config.json")

# Load configuration
config = Config.load("my_config.json")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies and derivatives involves substantial risk of loss. Always conduct thorough research and consider seeking advice from financial professionals before trading.
