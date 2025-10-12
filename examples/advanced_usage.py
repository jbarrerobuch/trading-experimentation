"""
Advanced usage example of the Gamma Neutral Strategy.

This example demonstrates:
1. Loading custom configuration
2. Multiple options positions with different strikes and expirations
3. Risk management integration
4. Performance tracking
5. Dynamic rebalancing over time
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from gamma_neutral import GammaNeutralStrategy
from gamma_neutral.utils.risk_management import RiskManager
from gamma_neutral.utils.config import Config


def main():
    """Run advanced example with risk management and performance tracking."""
    
    print("=" * 70)
    print("Gamma Neutral Strategy - Advanced Example")
    print("=" * 70)
    
    # Load custom configuration
    print("\n1. Loading Configuration...")
    config = Config.load("examples/config_example.json")
    
    # Validate configuration
    validation = config.validate()
    if validation["valid"]:
        print("   ✓ Configuration loaded and validated successfully")
    else:
        print("   ✗ Configuration validation errors:", validation["errors"])
        return
    
    if validation["warnings"]:
        print("   ⚠ Warnings:", validation["warnings"])
    
    # Initialize strategy with custom config
    strategy = GammaNeutralStrategy(
        target_gamma=config.get("strategy.target_gamma"),
        gamma_tolerance=config.get("strategy.gamma_tolerance"),
        delta_tolerance=config.get("strategy.delta_tolerance"),
        rebalance_frequency=config.get("strategy.rebalance_frequency"),
        risk_free_rate=config.get("trading.risk_free_rate"),
        transaction_cost=config.get("trading.transaction_cost")
    )
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_portfolio_delta=config.get("risk.max_portfolio_delta"),
        max_portfolio_gamma=config.get("risk.max_portfolio_gamma"),
        max_position_size=config.get("risk.max_position_size"),
        max_notional_exposure=config.get("risk.max_notional_exposure"),
        var_confidence=config.get("risk.var_confidence")
    )
    
    print(f"\n2. Strategy Configuration:")
    print(f"   - Target Gamma: {config.get('strategy.target_gamma')}")
    print(f"   - Gamma Tolerance: {config.get('strategy.gamma_tolerance')}")
    print(f"   - Delta Tolerance: {config.get('strategy.delta_tolerance')}")
    print(f"   - Rebalance Frequency: {config.get('strategy.rebalance_frequency')}s")
    print(f"   - Max Position Size: {config.get('risk.max_position_size')}")
    print(f"   - Max Notional Exposure: ${config.get('risk.max_notional_exposure'):,.0f}")
    
    # Market parameters
    spot_price = 50000.0
    volatility = config.get("market.default_volatility")
    
    # Create a diversified options portfolio
    print(f"\n3. Building Diversified Options Portfolio...")
    print(f"   - Spot Price: ${spot_price:,.2f}")
    print(f"   - Volatility: {volatility * 100:.1f}%")
    
    # ATM Straddle (long volatility position)
    expiry_30d = datetime.now() + timedelta(days=30)
    
    # Long call at 50000 strike
    strategy.add_option_position(
        quantity=5.0,
        entry_price=3000.0,
        strike_price=50000.0,
        expiry_date=expiry_30d,
        option_type="call",
        volatility=volatility
    )
    print(f"   ✓ Added 5 ATM calls (strike: 50000, 30d expiry)")
    
    # Long put at 50000 strike
    strategy.add_option_position(
        quantity=5.0,
        entry_price=2800.0,
        strike_price=50000.0,
        expiry_date=expiry_30d,
        option_type="put",
        volatility=volatility
    )
    print(f"   ✓ Added 5 ATM puts (strike: 50000, 30d expiry)")
    
    # OTM Call spread (sell upside)
    strategy.add_option_position(
        quantity=-3.0,
        entry_price=1500.0,
        strike_price=55000.0,
        expiry_date=expiry_30d,
        option_type="call",
        volatility=volatility
    )
    print(f"   ✓ Added 3 short OTM calls (strike: 55000, 30d expiry)")
    
    # OTM Put spread (sell downside)
    strategy.add_option_position(
        quantity=-3.0,
        entry_price=1400.0,
        strike_price=45000.0,
        expiry_date=expiry_30d,
        option_type="put",
        volatility=volatility
    )
    print(f"   ✓ Added 3 short OTM puts (strike: 45000, 30d expiry)")
    
    # Longer-dated protective positions
    expiry_60d = datetime.now() + timedelta(days=60)
    
    strategy.add_option_position(
        quantity=2.0,
        entry_price=3500.0,
        strike_price=52000.0,
        expiry_date=expiry_60d,
        option_type="call",
        volatility=volatility
    )
    print(f"   ✓ Added 2 long calls (strike: 52000, 60d expiry)")
    
    strategy.add_option_position(
        quantity=2.0,
        entry_price=3200.0,
        strike_price=48000.0,
        expiry_date=expiry_60d,
        option_type="put",
        volatility=volatility
    )
    print(f"   ✓ Added 2 long puts (strike: 48000, 60d expiry)")
    
    # Calculate portfolio Greeks
    print(f"\n4. Portfolio Greeks Analysis:")
    greeks = strategy.calculate_portfolio_greeks(spot_price)
    print(f"   - Options Delta: {greeks['options_delta']:>8.4f}")
    print(f"   - Portfolio Gamma: {greeks['gamma']:>8.4f}")
    print(f"   - Portfolio Vega: {greeks['vega']:>8.2f}")
    print(f"   - Portfolio Theta: {greeks['theta']:>8.2f} (per day)")
    
    # Risk assessment
    print(f"\n5. Risk Assessment:")
    
    # Check position limits
    status = strategy.get_portfolio_status(spot_price)
    greeks_check = risk_manager.check_greeks_limits(
        portfolio_delta=greeks["options_delta"],
        portfolio_gamma=greeks["gamma"]
    )
    
    print(f"   Greeks Within Limits:")
    print(f"   - Delta: {'✓' if greeks_check['delta_within_limit'] else '✗'} "
          f"(utilization: {greeks_check['delta_utilization']:.1%})")
    print(f"   - Gamma: {'✓' if greeks_check['gamma_within_limit'] else '✗'} "
          f"(utilization: {greeks_check['gamma_utilization']:.1%})")
    
    # Calculate VaR
    var = risk_manager.calculate_var(
        portfolio_delta=greeks["options_delta"],
        spot_price=spot_price,
        volatility=volatility,
        time_horizon=1.0
    )
    print(f"\n   Risk Metrics:")
    print(f"   - 1-Day VaR (95%): ${var:,.2f}")
    
    # Calculate Expected Shortfall
    es = risk_manager.calculate_expected_shortfall(
        portfolio_delta=greeks["options_delta"],
        spot_price=spot_price,
        volatility=volatility,
        time_horizon=1.0
    )
    print(f"   - Expected Shortfall: ${es:,.2f}")
    
    # Rebalancing
    print(f"\n6. Initial Rebalancing:")
    rebalance_check = strategy.check_rebalance_needed(spot_price)
    
    if rebalance_check["needs_any_rebalance"]:
        print(f"   - Rebalancing required:")
        print(f"     • Gamma deviation: {rebalance_check['gamma_deviation']:.4f}")
        print(f"     • Delta deviation: {rebalance_check['delta_deviation']:.4f}")
        
        result = strategy.rebalance(spot_price)
        
        print(f"\n   After Rebalancing:")
        print(f"   - Portfolio Delta: {result['greeks_after']['delta']:>8.4f}")
        print(f"   - Portfolio Gamma: {result['greeks_after']['gamma']:>8.4f}")
        print(f"   - Futures Position: {result['hedge_info']['futures_position']:>8.4f}")
        print(f"   - Execution Cost: ${result['execution']['execution_cost']:>8.2f}")
    
    # Simulate market scenarios
    print(f"\n7. Market Scenario Analysis:")
    
    scenarios = [
        (48000.0, "Down 4%"),
        (49000.0, "Down 2%"),
        (51000.0, "Up 2%"),
        (52000.0, "Up 4%"),
    ]
    
    print(f"\n   {'Price':<10} {'Scenario':<12} {'Delta':<10} {'Gamma':<10} {'Rebalance':<12}")
    print(f"   {'-'*60}")
    
    for price, scenario in scenarios:
        greeks_scenario = strategy.calculate_portfolio_greeks(price)
        rebalance = strategy.check_rebalance_needed(price)
        rebalance_status = "Yes" if rebalance["needs_any_rebalance"] else "No"
        
        print(f"   ${price:<9,.0f} {scenario:<12} {greeks_scenario['delta']:<10.4f} "
              f"{greeks_scenario['gamma']:<10.4f} {rebalance_status:<12}")
    
    # Portfolio summary
    print(f"\n8. Portfolio Summary:")
    final_status = strategy.get_portfolio_status(spot_price)
    summary = final_status["portfolio_summary"]
    
    print(f"   - Total Positions: {summary['total_positions']}")
    print(f"   - Options Positions: {summary['options_count']}")
    print(f"   - Futures Positions: {summary['futures_count']}")
    print(f"   - Options Notional: ${summary['options_notional']:,.2f}")
    print(f"   - Futures Notional: ${summary['futures_notional']:,.2f}")
    print(f"   - Total Notional: ${summary['total_notional']:,.2f}")
    
    # Performance considerations
    print(f"\n9. Strategy Performance Factors:")
    print(f"   - Theta Decay: ${greeks['theta'] * 365:.2f} per year (positive is profit)")
    print(f"   - Gamma Scalping: Portfolio can profit from volatility")
    print(f"   - Funding Costs: Monitor perpetual futures funding rates")
    print(f"   - Rebalancing Costs: Minimize through appropriate thresholds")
    
    print("\n" + "=" * 70)
    print("Advanced example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
