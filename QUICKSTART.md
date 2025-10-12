# Quick Start Guide

Get started with GammaNeutral in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/jbarrerobuch/GammaNeutral.git
cd GammaNeutral

# Install dependencies
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

## Basic Usage (3 Steps)

### Step 1: Initialize Strategy

```python
from gamma_neutral import GammaNeutralStrategy

strategy = GammaNeutralStrategy(
    target_gamma=0.0,        # Target gamma value
    gamma_tolerance=0.1,     # Rebalance when |gamma| > 0.1
    delta_tolerance=0.05     # Rebalance when |delta| > 0.05
)
```

### Step 2: Add Positions

```python
from datetime import datetime, timedelta

# Add a long call option
expiry = datetime.now() + timedelta(days=30)
strategy.add_option_position(
    quantity=10.0,           # Number of contracts
    entry_price=2500.0,      # Entry price in USD
    strike_price=52000.0,    # Strike price
    expiry_date=expiry,      # Expiration date
    option_type="call",      # "call" or "put"
    volatility=0.8           # 80% annualized volatility
)
```

### Step 3: Monitor & Rebalance

```python
spot_price = 50000.0  # Current BTC price

# Check portfolio Greeks
greeks = strategy.calculate_portfolio_greeks(spot_price)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")

# Rebalance if needed
if strategy.check_rebalance_needed(spot_price)['needs_any_rebalance']:
    result = strategy.rebalance(spot_price)
    print(f"Rebalanced! New delta: {result['greeks_after']['delta']:.4f}")
```

## Run Examples

```bash
# Basic example
python examples/basic_usage.py

# Advanced example with risk management
python examples/advanced_usage.py
```

## Run Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_greeks
```

## Common Use Cases

### 1. ATM Straddle (Long Volatility)

```python
expiry = datetime.now() + timedelta(days=30)

# Long ATM call
strategy.add_option_position(10.0, 3000.0, 50000.0, expiry, "call", 0.8)

# Long ATM put
strategy.add_option_position(10.0, 2800.0, 50000.0, expiry, "put", 0.8)
```

### 2. Iron Condor (Short Volatility)

```python
expiry = datetime.now() + timedelta(days=30)

# Short call spread
strategy.add_option_position(-5.0, 2000.0, 52000.0, expiry, "call", 0.8)
strategy.add_option_position(5.0, 1000.0, 55000.0, expiry, "call", 0.8)

# Short put spread
strategy.add_option_position(-5.0, 1800.0, 48000.0, expiry, "put", 0.8)
strategy.add_option_position(5.0, 900.0, 45000.0, expiry, "put", 0.8)
```

### 3. Calendar Spread (Theta Strategy)

```python
expiry_near = datetime.now() + timedelta(days=7)
expiry_far = datetime.now() + timedelta(days=30)

# Short near-term call
strategy.add_option_position(-10.0, 1500.0, 50000.0, expiry_near, "call", 0.8)

# Long far-term call
strategy.add_option_position(10.0, 2500.0, 50000.0, expiry_far, "call", 0.8)
```

## Configuration

### Using Config File

```python
from gamma_neutral.utils.config import Config

# Load from JSON
config = Config.load("examples/config_example.json")

# Initialize strategy with config
strategy = GammaNeutralStrategy(
    target_gamma=config.get("strategy.target_gamma"),
    gamma_tolerance=config.get("strategy.gamma_tolerance"),
    delta_tolerance=config.get("strategy.delta_tolerance")
)
```

### Custom Configuration

```python
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

# Validate
validation = config.validate()
if validation["valid"]:
    print("✓ Configuration valid")
```

## Risk Management

```python
from gamma_neutral.utils.risk_management import RiskManager

risk_manager = RiskManager(
    max_portfolio_delta=1.0,
    max_portfolio_gamma=0.5,
    max_position_size=100.0
)

# Calculate VaR
var = risk_manager.calculate_var(
    portfolio_delta=greeks["delta"],
    spot_price=50000.0,
    volatility=0.8,
    time_horizon=1.0  # 1 day
)
print(f"1-Day VaR (95%): ${var:,.2f}")

# Check limits
check = risk_manager.check_greeks_limits(
    portfolio_delta=greeks["delta"],
    portfolio_gamma=greeks["gamma"]
)
print(f"Within limits: {check['passes_all_checks']}")
```

## Portfolio Status

```python
# Get comprehensive status
status = strategy.get_portfolio_status(spot_price)

print(f"Total Positions: {status['portfolio_summary']['total_positions']}")
print(f"Portfolio Delta: {status['greeks']['delta']:.4f}")
print(f"Portfolio Gamma: {status['greeks']['gamma']:.4f}")
print(f"Futures Position: {status['futures_position']:.4f}")
```

## Key Concepts

### Greeks Explained

- **Delta (Δ)**: How much option price changes per $1 move in underlying
- **Gamma (Γ)**: How much delta changes per $1 move in underlying
- **Vega (ν)**: How much option price changes per 1% change in volatility
- **Theta (Θ)**: How much option price decays per day

### Rebalancing Triggers

The strategy rebalances when:

1. **Gamma exceeds tolerance**: `|portfolio_gamma - target_gamma| > gamma_tolerance`
2. **Delta exceeds tolerance**: `|portfolio_delta| > delta_tolerance`
3. **Time interval elapsed**: Based on `rebalance_frequency` setting

### Transaction Costs

Every rebalance incurs costs:

```
Cost = |position_change| × spot_price × transaction_cost_rate
```

Minimize by:
- Using wider tolerance thresholds
- Longer rebalance intervals
- Efficient position structuring

## Best Practices

1. **Start Small**: Test with small positions first
2. **Monitor Greeks**: Check portfolio Greeks frequently
3. **Track Costs**: Keep rebalancing costs under control
4. **Risk Limits**: Always use risk management
5. **Backtest**: Test strategies before live trading
6. **Document**: Keep detailed records of all trades

## Troubleshooting

### High Rebalancing Frequency

**Problem**: Strategy rebalances too often, high costs

**Solution**: 
- Increase `gamma_tolerance` and `delta_tolerance`
- Increase `rebalance_frequency`
- Use wider strikes for options

### Large Delta Exposure

**Problem**: Delta is always outside tolerance

**Solution**:
- Add more options to balance delta
- Use options closer to ATM
- Check futures hedge is executing properly

### Negative P&L

**Problem**: Strategy losing money consistently

**Potential Causes**:
- High transaction costs
- Unfavorable funding rates
- Volatility decline (if long gamma)
- Time decay (theta) overwhelming other profits

## Next Steps

1. Read [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) for detailed theory
2. Study the examples in `examples/` directory
3. Review test cases in `tests/` for usage patterns
4. Customize configuration for your needs
5. Implement your own strategies

## Getting Help

- **Documentation**: Read README.md and TECHNICAL_DOCS.md
- **Examples**: Study the example files
- **Tests**: Review test files for usage patterns
- **GitHub Issues**: Open an issue for bugs or questions

## Disclaimer

⚠️ **Important**: This software is for educational purposes only. Trading involves substantial risk of loss. Always test thoroughly before using real capital.

---

**Ready to trade?** Start with `python examples/basic_usage.py` to see it in action!
