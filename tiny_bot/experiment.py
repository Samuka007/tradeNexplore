"""Single experiment: PSO + GP on BTC train/test split."""

import numpy as np
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO
from tiny_bot.gp import GP


def run_experiment(csv_path: str, seed: int = 42):
    """Run one experiment: load data, train PSO + GP, evaluate on test.

    Returns dict with full metrics for each strategy.
    """
    np.random.seed(seed)
    train, test = load_btc_data(csv_path)

    bh_train = buy_and_hold(train)
    bh_test = buy_and_hold(test)
    print(f"Buy-and-hold  train=${bh_train:,.2f}  test=${bh_test:,.2f}")

    bounds = (
        [(0.0, 1.0)] * 3
        + [(2.0, 200.0)] * 3
        + [(0.01, 0.99)]
        + [(0.0, 1.0)] * 3
        + [(2.0, 200.0)] * 3
        + [(0.01, 0.99)]
    )

    def pso_fit(params):
        s = VectorStrategy(params, "dual_crossover")
        return backtest(train, s.signals(train))["final_cash"]

    pso = PSO(n_particles=30, max_iter=50, seed=seed)
    pso_res = pso.optimize(pso_fit, bounds)
    pso_test = backtest(
        test, VectorStrategy(pso_res["best"], "dual_crossover").signals(test)
    )
    print(
        f"PSO           train=${pso_res['fitness']:,.2f}  "
        f"test=${pso_test['final_cash']:,.2f}  "
        f"sharpe={pso_test['sharpe_ratio']:.2f}  dd={pso_test['max_drawdown']:.2%}"
    )

    gp = GP(pop_size=50, generations=30, seed=seed)

    def gp_fit(tree):
        return backtest(train, gp.evaluate(tree, train))["final_cash"]

    gp_res = gp.optimize(gp_fit)
    gp_test = backtest(test, gp.evaluate(gp_res["best"], test))
    print(
        f"GP            train=${gp_res['fitness']:,.2f}  "
        f"test=${gp_test['final_cash']:,.2f}  "
        f"sharpe={gp_test['sharpe_ratio']:.2f}  dd={gp_test['max_drawdown']:.2%}"
    )
    print(f"Best tree: {gp_res['best']}")

    return {
        "buy_and_hold": {"train": bh_train, "test": bh_test},
        "pso": {
            "train": pso_res["fitness"],
            "test": pso_test["final_cash"],
            "params": pso_res["best"],
            "history": pso_res["history"],
            "metrics": {k: v for k, v in pso_test.items() if k != "equity_curve"},
        },
        "gp": {
            "train": gp_res["fitness"],
            "test": gp_test["final_cash"],
            "tree": gp_res["best"],
            "history": gp_res["history"],
            "metrics": {k: v for k, v in gp_test.items() if k != "equity_curve"},
        },
    }