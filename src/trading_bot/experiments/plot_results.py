"""Load experiment JSON results and generate visualizations.

Bridges the gap between JSON-serialized results and visualization.py,
which expects BacktestResult objects.
"""

import dataclasses
import json
from pathlib import Path
import numpy as np

from trading_bot.data_loader import load_or_generate_data
from trading_bot.strategy import VectorStrategy, GoldenCross, DeathCross
from trading_bot.backtester import Backtester, BacktestResult, Trade
from trading_bot.visualization import (
    plot_convergence, plot_equity_curves, plot_drawdowns, plot_trade_points,
)


def _as_backtest_result(equity_curve, final_cash=None, n_trades=0, win_rate=0.0, trades=None):
    """Construct a BacktestResult from saved equity curve (for visualization)."""
    ec = np.array(equity_curve, dtype=np.float64)
    trades_list = []
    if trades:
        for td in trades:
            trades_list.append(Trade(
                entry_idx=td.get("entry_idx", 0),
                entry_price=td.get("entry_price", 0.0),
                entry_cost=td.get("entry_cost", 0.0),
                exit_idx=td.get("exit_idx"),
                exit_price=td.get("exit_price"),
                shares=td.get("shares", 0.0),
                pnl=td.get("pnl", 0.0),
            ))
    return BacktestResult(
        final_cash=float(final_cash) if final_cash is not None else float(ec[-1]),
        equity_curve=ec,
        trades=trades_list,
        n_trades=n_trades,
        win_rate=win_rate,
    )


def load_and_plot(results_dir="results", plots_dir="results/plots"):
    """Load all experiment JSON files and generate plots."""
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ds = load_or_generate_data('data/btc_daily_2014_2022.csv')

    # --- Load JSONs ---
    exp1 = json.loads((results_dir / "experiment_1_results.json").read_text())
    exp2 = json.loads((results_dir / "experiment_2_results.json").read_text())
    exp3 = json.loads((results_dir / "experiment_3_results.json").read_text())

    # --- Reconstruct BacktestResult for equity/drawdown plots ---
    test_results = {}

    # PSO dual_crossover (from Exp1 best_params)
    dc_params = exp1['results']['dual_crossover']['best_params']
    dc = VectorStrategy(np.array(dc_params), 'dual_crossover')
    sig = dc.generate_signals(ds.test_prices)
    test_results['PSO dual_cross'] = Backtester(ds.test_prices).evaluate(sig)

    # PSO MACD (from Exp1 best_params)
    macd_params = exp1['results']['macd']['best_params']
    macd = VectorStrategy(np.array(macd_params), 'macd')
    sig_macd = macd.generate_signals(ds.test_prices)
    test_results['PSO MACD'] = Backtester(ds.test_prices).evaluate(sig_macd)

    # Buy & Hold baseline
    bh_sig = np.zeros(len(ds.test_prices))
    bh_sig[0] = 1
    test_results['Buy & Hold'] = Backtester(ds.test_prices).evaluate(bh_sig)

    # Golden Cross baseline
    gc = GoldenCross(50, 200)
    test_results['Golden Cross'] = Backtester(ds.test_prices).evaluate(gc.generate_signals(ds.test_prices))

    # Death Cross baseline
    dc_baseline = DeathCross(50, 200)
    test_results['Death Cross'] = Backtester(ds.test_prices).evaluate(dc_baseline.generate_signals(ds.test_prices))

    # GP best tree (from Exp2 equity_curve + trades)
    if "best_equity_curve" in exp2.get("results", {}):
        ec_data = exp2["results"]["best_equity_curve"]
        metrics = exp2["results"].get("best_metrics", {})
        trades_data = exp2["results"].get("best_trades", [])
        test_results['GP best tree'] = _as_backtest_result(
            ec_data,
            final_cash=exp2["results"]["best_test_fitness"],
            n_trades=metrics.get("n_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            trades=trades_data,
        )

    # PSO-refined (from Exp3 best_refined)
    if exp3.get("results", {}).get("best_refined_equity_curve") is not None:
        ec_data = exp3["results"]["best_refined_equity_curve"]
        metrics = exp3["results"].get("best_refined_metrics", {})
        trades_data = exp3["results"].get("best_refined_trades", [])
        test_results['PSO refined'] = _as_backtest_result(
            ec_data,
            final_cash=exp3["results"]["refined_test_fitness"]["mean"],
            n_trades=metrics.get("n_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            trades=trades_data,
        )

    print("=== Reconstructed BacktestResult objects ===")
    for name, bt in test_results.items():
        print(f"  {name:.<25} final_cash=${bt.final_cash:>10,.0f}  trades={bt.n_trades:>4}  win_rate={bt.win_rate:>5.1f}%")

    # --- Equity Curves ---
    fig1 = plot_equity_curves(
        test_results,
        title='Strategy Comparison: Equity Curves (Test Set 2020-2022)',
        save_path=str(plots_dir / 'equity_curves.png'),
    )

    # --- Drawdowns ---
    fig2 = plot_drawdowns(
        test_results,
        title='Strategy Comparison: Drawdowns (Test Set 2020-2022)',
        save_path=str(plots_dir / 'drawdowns.png'),
    )

    # --- Trade Points ---
    fig3 = plot_trade_points(
        ds.test_prices,
        test_results,
        title='Strategy Comparison: Trade Points (Test Set 2020-2022)',
        save_path=str(plots_dir / 'trade_points.png'),
    )

    # --- Convergence (PSO only) ---
    conv_data = {}
    for name in ['dual_crossover', 'macd']:
        if name in exp1['results']:
            conv_data[f'PSO {name}'] = exp1['results'][name]['convergence'][0]

    # GP convergence from Exp2
    if 'convergence' in exp2.get('results', {}):
        gp_conv = exp2['results']['convergence']
        if gp_conv and len(gp_conv) > 0:
            conv_data['GP'] = gp_conv[0]

    fig4 = plot_convergence(
        conv_data,
        title='Optimization Convergence',
        save_path=str(plots_dir / 'convergence.png'),
    )

    # --- Summary table ---
    print(f"\n=== Experiment Summary ===")
    print(f"Exp1 (PSO) dual_cross: test ${exp1['results']['dual_crossover']['best_test_fitness']:,.0f}")
    print(f"Exp2 (GP):             test ${exp2['results']['best_test_fitness']:,.0f}")
    print(f"Exp3 (PSO refine):     GP test=${exp3['results']['gp_test_fitness']:,.0f}")

    imp = exp3['results']['improvement']
    print(f"  Refined test: mean={imp['mean']:,.0f}, pct={imp['mean_pct']:+.1f}%")
    print(f"Baseline buy&hold:     test ${exp1['baselines']['buy_and_hold']['test_fitness']:,.0f}")

    print(f"\nPlots saved to {plots_dir}/")
    print(f"  - equity_curves.png")
    print(f"  - drawdowns.png")
    print(f"  - trade_points.png")
    print(f"  - convergence.png")
    return test_results


if __name__ == "__main__":
    load_and_plot()
