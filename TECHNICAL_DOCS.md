# Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Options Greeks](#options-greeks)
3. [Gamma Neutral Strategy Theory](#gamma-neutral-strategy-theory)
4. [Implementation Details](#implementation-details)
5. [Risk Management](#risk-management)
6. [Perpetual Futures Hedging](#perpetual-futures-hedging)
7. [Performance Considerations](#performance-considerations)

## Architecture Overview

The GammaNeutral package is organized into several key modules:

```
gamma_neutral/
├── core/               # Core functionality
│   ├── greeks.py      # Options Greeks calculation
│   ├── hedging.py     # Perpetual futures hedging
│   └── portfolio.py   # Position tracking
├── strategies/         # Strategy implementations
│   └── gamma_neutral.py
└── utils/             # Utility modules
    ├── config.py      # Configuration management
    └── risk_management.py
```

### Design Principles

1. **Modularity**: Each component is independent and can be used separately
2. **Extensibility**: Easy to add new strategies or risk metrics
3. **Type Safety**: Type hints throughout for better IDE support
4. **Testing**: Comprehensive unit tests for all components

## Options Greeks

### Black-Scholes Model

The package uses the Black-Scholes model to calculate options Greeks. The key formulas:

#### Delta (Δ)

Measures the rate of change of option price with respect to the underlying price.

- **Call Delta**: `Δ_call = N(d1)`
- **Put Delta**: `Δ_put = N(d1) - 1`

Where:
```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
```

#### Gamma (Γ)

Measures the rate of change of delta with respect to the underlying price.

```
Γ = N'(d1) / (S × σ × √T)
```

Where `N'(d1)` is the standard normal probability density function.

#### Vega (ν)

Measures sensitivity to volatility changes.

```
ν = S × N'(d1) × √T
```

#### Theta (Θ)

Measures time decay of the option.

**Call Theta**:
```
Θ_call = -S × N'(d1) × σ / (2√T) - r × K × e^(-rT) × N(d2)
```

**Put Theta**:
```
Θ_put = -S × N'(d1) × σ / (2√T) + r × K × e^(-rT) × N(-d2)
```

### Implementation Notes

- Greeks are calculated per option and then aggregated for portfolio-level metrics
- Theta is converted to per-day values (divided by 365)
- Vega is expressed per 1% volatility change (divided by 100)
- Risk-free rate is typically set to 0 for crypto markets

## Gamma Neutral Strategy Theory

### What is Gamma Neutrality?

A gamma-neutral portfolio has minimal sensitivity to changes in the underlying asset's delta. This is achieved by balancing long and short options positions such that:

```
Portfolio_Gamma ≈ 0
```

### Why Gamma Neutral?

1. **Reduced Directional Risk**: Less exposure to large price movements
2. **Volatility Trading**: Profit from changes in implied volatility
3. **Theta Collection**: Benefit from time decay when selling options
4. **Gamma Scalping**: Profit from rebalancing in volatile markets

### Strategy Mechanics

#### 1. Position Construction

Create a portfolio of options such that:
- Long positions provide positive gamma
- Short positions provide negative gamma
- Net gamma is close to zero

Example:
- Long ATM straddle: High positive gamma
- Short OTM strangle: Negative gamma
- Result: Near-neutral gamma with defined risk

#### 2. Delta Hedging

Use perpetual futures to maintain delta neutrality:

```python
Required_Futures_Position = -Portfolio_Delta
```

#### 3. Dynamic Rebalancing

As the underlying price changes, delta changes due to gamma:

```
ΔDelta = Gamma × ΔPrice
```

Rebalance when:
- Delta exceeds tolerance threshold
- Gamma exceeds tolerance threshold
- Time-based interval reached

### P&L Sources

1. **Theta Decay**: Collect premium from short options
2. **Volatility Changes**: Benefit from long vega if volatility increases
3. **Gamma Scalping**: Buy low, sell high during rebalancing
4. **Funding Rates**: Earn/pay on perpetual futures positions

## Implementation Details

### OptionsGreeksCalculator

```python
calculator = OptionsGreeksCalculator(risk_free_rate=0.0)

greeks = calculator.calculate_greeks(
    spot_price=50000.0,
    strike_price=52000.0,
    time_to_expiry=30/365.0,  # 30 days in years
    volatility=0.8,            # 80% annualized
    option_type="call"
)
```

**Key Methods**:
- `calculate_greeks()`: Individual option Greeks
- `calculate_portfolio_gamma()`: Aggregate portfolio gamma
- `calculate_portfolio_delta()`: Aggregate portfolio delta

### PerpetualFuturesHedger

```python
hedger = PerpetualFuturesHedger(
    min_rebalance_threshold=0.1,
    transaction_cost=0.0005
)

hedge = hedger.calculate_gamma_hedge(
    portfolio_gamma=0.05,
    portfolio_delta=1.2,
    spot_price=50000.0,
    target_gamma=0.0
)
```

**Key Features**:
- Calculates required futures position for delta neutrality
- Estimates transaction costs
- Determines if rebalancing is needed
- Tracks current position

### GammaNeutralStrategy

Main strategy orchestrator:

```python
strategy = GammaNeutralStrategy(
    target_gamma=0.0,
    gamma_tolerance=0.1,
    delta_tolerance=0.05
)

# Add positions
strategy.add_option_position(...)

# Check and rebalance
if strategy.check_rebalance_needed(spot_price)['needs_any_rebalance']:
    strategy.rebalance(spot_price)
```

**Rebalancing Logic**:

1. Calculate current portfolio Greeks
2. Check against tolerance thresholds
3. Calculate required hedge
4. Execute futures position adjustment
5. Log rebalance for analysis

## Risk Management

### Position Limits

```python
risk_manager = RiskManager(
    max_portfolio_delta=1.0,
    max_portfolio_gamma=0.5,
    max_position_size=100.0,
    max_notional_exposure=1000000.0
)
```

### Value at Risk (VaR)

Delta-normal VaR calculation:

```
VaR = Portfolio_Value × Volatility × Z_score × √(Time_Horizon)
```

### Expected Shortfall (CVaR)

Conditional VaR using Monte Carlo simulation:

```python
es = risk_manager.calculate_expected_shortfall(
    portfolio_delta=delta,
    spot_price=50000.0,
    volatility=0.8,
    time_horizon=1.0
)
```

### Maximum Drawdown

Tracks peak-to-trough decline:

```python
dd = risk_manager.calculate_maximum_drawdown(equity_curve)
```

### Sharpe Ratio

Risk-adjusted return metric:

```
Sharpe = (Mean_Return - Risk_Free_Rate) / StdDev_Return × √252
```

## Perpetual Futures Hedging

### Why Perpetual Futures?

1. **No Expiration**: Continuous hedging without rolling contracts
2. **High Liquidity**: Deep order books in crypto markets
3. **Funding Rates**: Additional P&L source
4. **Leverage**: Capital efficient hedging

### Funding Rates

Perpetual futures use funding rates to anchor price to spot:

```python
funding_cost = hedger.calculate_funding_cost(
    position_size=-10.0,
    funding_rate=0.0001,  # 0.01% per 8 hours
    time_periods=3         # 24 hours = 3 periods
)
```

**Key Points**:
- Positive rate: Longs pay shorts
- Negative rate: Shorts pay longs
- Typically settled every 8 hours
- Important cost/revenue consideration

### Dynamic Hedge Ratio

Calculate position changes based on gamma:

```python
adjustment = hedger.calculate_dynamic_hedge_ratio(
    portfolio_gamma=0.05,
    spot_price=50000.0,
    price_change=1000.0
)
```

## Performance Considerations

### Transaction Costs

Minimize costs by:
1. Setting appropriate rebalance thresholds
2. Using limit orders when possible
3. Batching adjustments
4. Monitoring bid-ask spreads

### Slippage

Consider in position sizing:
- Larger positions experience more slippage
- Use VWAP or TWAP for large orders
- Split orders across time

### Greeks Accuracy

Factors affecting Greeks accuracy:
1. **Volatility Smile**: Black-Scholes assumes constant volatility
2. **Interest Rates**: Usually minimal for crypto
3. **Dividends**: Not applicable to crypto
4. **Early Exercise**: Use American option model if needed

### Computational Efficiency

Optimizations:
- Cache Greeks calculations
- Batch position updates
- Use vectorized operations (numpy)
- Pre-compute common values

### Latency Considerations

For high-frequency rebalancing:
- Co-locate servers near exchange
- Use WebSocket feeds for real-time data
- Implement asynchronous execution
- Monitor order confirmation latency

## Advanced Topics

### Volatility Surface

Future enhancement: Incorporate volatility smile/skew:

```python
# Instead of single volatility
volatility = get_implied_vol(strike, expiry)
```

### Cross-Asset Hedging

Extend to multiple underlyings:
- Bitcoin + Ethereum portfolio
- Correlated assets
- Beta-weighted delta

### Machine Learning Integration

Potential ML applications:
1. Volatility forecasting
2. Optimal rebalance timing
3. Dynamic threshold adjustment
4. Pattern recognition in P&L

### Backtesting Framework

Future enhancement for strategy validation:
- Historical options data
- Realistic transaction costs
- Slippage modeling
- Performance metrics

## References

1. **Options, Futures, and Other Derivatives** by John C. Hull
2. **Dynamic Hedging** by Nassim Nicholas Taleb
3. **Volatility Trading** by Euan Sinclair
4. Black-Scholes-Merton Option Pricing Model (1973)

## Contributing

When contributing to this project:

1. Maintain backward compatibility
2. Add unit tests for new features
3. Update documentation
4. Follow existing code style
5. Consider performance implications

## Support

For issues, questions, or contributions:
- GitHub Issues: [github.com/jbarrerobuch/GammaNeutral/issues](https://github.com/jbarrerobuch/GammaNeutral/issues)
- Pull Requests: Always welcome!
