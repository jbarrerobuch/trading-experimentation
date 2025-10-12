"""
Basic usage example of the Gamma Neutral Strategy.

This example demonstrates how to:
1. Initialize the strategy
2. Add options positions
3. Calculate portfolio Greeks
4. Rebalance the portfolio
5. Monitor portfolio status
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from gamma_neutral import GammaNeutralStrategy


def main():
    """Run a basic example of the gamma neutral strategy."""
    
    print("=" * 60)
    print("Gamma Neutral Strategy - Basic Example")
    print("=" * 60)
    
    # Initialize the strategy
    strategy = GammaNeutralStrategy(
        target_gamma=0.0,
        gamma_tolerance=0.1,
        delta_tolerance=0.05,
        rebalance_frequency=3600,
        risk_free_rate=0.0,
        transaction_cost=0.0005
    )
    
    print("\n1. Strategy initialized with:")
    print(f"   - Target Gamma: 0.0")
    print(f"   - Gamma Tolerance: 0.1")
    print(f"   - Delta Tolerance: 0.05")
    
    # Market parameters
    spot_price = 50000.0  # BTC price
    volatility = 0.8  # 80% annualized volatility
    
    # Add some options positions
    print("\n2. Adding options positions...")
    
    # Long call option
    expiry_date_1 = datetime.now() + timedelta(days=30)
    strategy.add_option_position(
        quantity=10.0,
        entry_price=2500.0,
        strike_price=52000.0,
        expiry_date=expiry_date_1,
        option_type="call",
        volatility=volatility
    )
    print(f"   - Added 10 long call options (strike: 52000)")
    
    # Short put option
    expiry_date_2 = datetime.now() + timedelta(days=30)
    strategy.add_option_position(
        quantity=-8.0,
        entry_price=2000.0,
        strike_price=48000.0,
        expiry_date=expiry_date_2,
        option_type="put",
        volatility=volatility
    )
    print(f"   - Added 8 short put options (strike: 48000)")
    
    # Calculate initial portfolio Greeks
    print("\n3. Initial Portfolio Greeks:")
    greeks = strategy.calculate_portfolio_greeks(spot_price)
    print(f"   - Delta: {greeks['delta']:.4f}")
    print(f"   - Gamma: {greeks['gamma']:.4f}")
    print(f"   - Vega: {greeks['vega']:.4f}")
    print(f"   - Theta: {greeks['theta']:.4f}")
    print(f"   - Options Delta: {greeks['options_delta']:.4f}")
    print(f"   - Futures Delta: {greeks['futures_delta']:.4f}")
    
    # Check if rebalancing is needed
    print("\n4. Checking if rebalancing is needed...")
    rebalance_check = strategy.check_rebalance_needed(spot_price)
    print(f"   - Needs Gamma Rebalance: {rebalance_check['needs_gamma_rebalance']}")
    print(f"   - Needs Delta Rebalance: {rebalance_check['needs_delta_rebalance']}")
    print(f"   - Gamma Deviation: {rebalance_check['gamma_deviation']:.4f}")
    print(f"   - Delta Deviation: {rebalance_check['delta_deviation']:.4f}")
    
    # Perform rebalancing
    if rebalance_check['needs_any_rebalance']:
        print("\n5. Performing rebalancing...")
        rebalance_result = strategy.rebalance(spot_price)
        
        print(f"   - Spot Price: {rebalance_result['spot_price']:.2f}")
        print(f"\n   Greeks Before Rebalance:")
        print(f"     - Delta: {rebalance_result['greeks_before']['delta']:.4f}")
        print(f"     - Gamma: {rebalance_result['greeks_before']['gamma']:.4f}")
        print(f"\n   Greeks After Rebalance:")
        print(f"     - Delta: {rebalance_result['greeks_after']['delta']:.4f}")
        print(f"     - Gamma: {rebalance_result['greeks_after']['gamma']:.4f}")
        print(f"\n   Hedge Execution:")
        print(f"     - Futures Position: {rebalance_result['hedge_info']['futures_position']:.4f}")
        print(f"     - Position Change: {rebalance_result['execution']['position_change']:.4f}")
        print(f"     - Execution Cost: ${rebalance_result['execution']['execution_cost']:.2f}")
    
    # Get portfolio status
    print("\n6. Final Portfolio Status:")
    status = strategy.get_portfolio_status(spot_price)
    print(f"   - Total Positions: {status['portfolio_summary']['total_positions']}")
    print(f"   - Options Count: {status['portfolio_summary']['options_count']}")
    print(f"   - Futures Count: {status['portfolio_summary']['futures_count']}")
    print(f"   - Total Notional: ${status['portfolio_summary']['total_notional']:.2f}")
    print(f"   - Futures Position: {status['futures_position']:.4f}")
    
    # Simulate price movement and check dynamic hedging
    print("\n7. Simulating price movement...")
    new_spot_price = 51000.0
    print(f"   - Price moved to: ${new_spot_price:.2f}")
    
    new_greeks = strategy.calculate_portfolio_greeks(new_spot_price)
    print(f"   - New Delta: {new_greeks['delta']:.4f}")
    print(f"   - New Gamma: {new_greeks['gamma']:.4f}")
    
    rebalance_check_2 = strategy.check_rebalance_needed(new_spot_price)
    if rebalance_check_2['needs_any_rebalance']:
        print(f"   - Rebalancing needed due to price change")
        print(f"   - Delta Deviation: {rebalance_check_2['delta_deviation']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
