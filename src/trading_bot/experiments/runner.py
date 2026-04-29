"""Experiment runner for orchestrating optimization and backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from trading_bot.algorithms.base import ContinuousOptimizer, OptResult, StructuralOptimizer
from trading_bot.backtester import Backtester, BacktestResult
from trading_bot.data_loader import Dataset
from trading_bot.strategy import Strategy, VectorStrategy

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Complete result from a single experiment run."""

    algorithm_name: str
    strategy_type: str
    opt_result: OptResult
    backtest_result: BacktestResult
    train_fitness: float
    val_fitness: Optional[float] = None
    test_fitness: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class ExperimentRunner:
    """Run optimization experiments on trading strategies.

    Provides unified paths for:
    - Continuous optimization (PSO, ABC, HS) with fixed strategy structures
    - Structural optimization (GP) with tree-based strategies
    - Hierarchical two-stage: GP structure discovery + parameter refinement
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def run_continuous(
        self,
        optimizer: ContinuousOptimizer,
        bounds: list[tuple[float, float]],
        strategy_type: str = "dual_crossover",
        evaluate_on_test: bool = True,
    ) -> ExperimentResult:
        """Run a continuous optimizer on a fixed strategy structure.

        Args:
            optimizer: PSO, ABC, or HS optimizer instance.
            bounds: Parameter bounds [(min, max), ...] for each dimension.
            strategy_type: Strategy type identifier (e.g. 'dual_crossover', 'macd').
            evaluate_on_test: If True, also evaluate best params on test set.

        Returns:
            ExperimentResult with optimization and backtest results.
        """
        logger.info(
            "Starting continuous optimization: %s on %s",
            optimizer.name,
            strategy_type,
        )

        train_prices = self.dataset.train_prices

        def fitness_fn(params: np.ndarray) -> float:
            strategy = VectorStrategy(params, strategy_type)
            signals = strategy.generate_signals(train_prices)
            backtester = Backtester(train_prices)
            result = backtester.evaluate(signals)
            return result.fitness

        opt_result = optimizer.optimize(fitness_fn, bounds)

        best_strategy = VectorStrategy(opt_result.best, strategy_type)
        train_signals = best_strategy.generate_signals(train_prices)
        train_backtest = Backtester(train_prices).evaluate(train_signals)

        val_fitness = None
        val_backtest = None
        if self.dataset.val_prices is not None and len(self.dataset.val_prices) > 0:
            val_signals = best_strategy.generate_signals(self.dataset.val_prices)
            val_backtest = Backtester(self.dataset.val_prices).evaluate(val_signals)
            val_fitness = val_backtest.fitness

        test_fitness = None
        test_backtest = None
        if evaluate_on_test:
            test_prices = self.dataset.test_prices
            test_signals = best_strategy.generate_signals(test_prices)
            test_backtest = Backtester(test_prices).evaluate(test_signals)
            test_fitness = test_backtest.fitness

        return ExperimentResult(
            algorithm_name=optimizer.name,
            strategy_type=strategy_type,
            opt_result=opt_result,
            backtest_result=train_backtest,
            train_fitness=train_backtest.fitness,
            val_fitness=val_fitness,
            test_fitness=test_fitness,
            metadata={
                "strategy_description": best_strategy.describe(),
                "val_backtest": val_backtest,
                "test_backtest": test_backtest,
            },
        )

    def run_structural(
        self,
        optimizer: StructuralOptimizer,
        strategy_factory: Callable[[object], Strategy],
        evaluate_on_test: bool = True,
    ) -> ExperimentResult:
        """Run a structural optimizer (GP) on tree-based strategies.

        Args:
            optimizer: GP optimizer instance.
            strategy_factory: Required function converting tree to Strategy.
                              Must be provided — no fallback.
            evaluate_on_test: If True, also evaluate best tree on test set.

        Returns:
            ExperimentResult with optimization and backtest results.

        Raises:
            ValueError: If strategy_factory is not provided.
        """
        if strategy_factory is None:
            raise ValueError(
                "strategy_factory is required for structural optimization. "
                "Provide a callable that converts a GP tree to a Strategy."
            )

        logger.info("Starting structural optimization: %s", optimizer.name)

        train_prices = self.dataset.train_prices

        def fitness_fn(tree: object) -> float:
            strategy = strategy_factory(tree)
            signals = strategy.generate_signals(train_prices)
            backtester = Backtester(train_prices)
            result = backtester.evaluate(signals)
            return result.fitness

        opt_result = optimizer.optimize(fitness_fn, n_generations=50)

        best_strategy = strategy_factory(opt_result.best)
        train_signals = best_strategy.generate_signals(train_prices)
        train_backtest = Backtester(train_prices).evaluate(train_signals)

        val_fitness = None
        val_backtest = None
        if self.dataset.val_prices is not None and len(self.dataset.val_prices) > 0:
            val_signals = best_strategy.generate_signals(self.dataset.val_prices)
            val_backtest = Backtester(self.dataset.val_prices).evaluate(val_signals)
            val_fitness = val_backtest.fitness

        test_fitness = None
        test_backtest = None
        if evaluate_on_test:
            test_prices = self.dataset.test_prices
            test_signals = best_strategy.generate_signals(test_prices)
            test_backtest = Backtester(test_prices).evaluate(test_signals)
            test_fitness = test_backtest.fitness

        return ExperimentResult(
            algorithm_name=optimizer.name,
            strategy_type="gp_tree",
            opt_result=opt_result,
            backtest_result=train_backtest,
            train_fitness=train_backtest.fitness,
            val_fitness=val_fitness,
            test_fitness=test_fitness,
            metadata={
                "strategy_description": best_strategy.describe(),
                "val_backtest": val_backtest,
                "test_backtest": test_backtest,
            },
        )

    def run_hierarchical(
        self,
        gp_optimizer: StructuralOptimizer,
        param_optimizer: ContinuousOptimizer,
        strategy_factory: Callable[[object], Strategy],
        template_extractor: Callable[[object, np.ndarray], tuple[VectorStrategy, list[tuple[float, float]]]],
        evaluate_on_test: bool = True,
    ) -> dict:
        """Two-stage hierarchical optimization.

        Stage 1: GP discovers promising structure.
        Stage 2: Extract parametric template from GP tree and refine with PSO/ABC/HS.

        Args:
            gp_optimizer: GP optimizer for structure discovery.
            param_optimizer: Continuous optimizer for parameter refinement.
            strategy_factory: Required function converting tree to Strategy.
            template_extractor: Function that takes (gp_tree, prices) and returns
                                (template_strategy, param_bounds) for Stage 2.
            evaluate_on_test: If True, evaluate on test set.

        Returns:
            Dictionary with GP result, refined result, and improvement metrics.

        Raises:
            ValueError: If strategy_factory or template_extractor not provided.
        """
        if strategy_factory is None:
            raise ValueError("strategy_factory is required for hierarchical optimization")
        if template_extractor is None:
            raise ValueError(
                "template_extractor is required for hierarchical optimization. "
                "It should convert a GP tree into a parametric template with bounds."
            )

        logger.info("Starting hierarchical two-stage optimization")

        gp_result = self.run_structural(
            gp_optimizer,
            strategy_factory=strategy_factory,
            evaluate_on_test=evaluate_on_test,
        )

        gp_tree = gp_result.opt_result.best
        template_strategy, bounds = template_extractor(gp_tree, self.dataset.train_prices)

        def refined_fitness(params: np.ndarray) -> float:
            strategy = VectorStrategy(params, template_strategy.type)
            signals = strategy.generate_signals(self.dataset.train_prices)
            return Backtester(self.dataset.train_prices).evaluate(signals).fitness

        refined_opt_result = param_optimizer.optimize(refined_fitness, bounds)
        refined_strategy = VectorStrategy(refined_opt_result.best, template_strategy.type)
        refined_train_signals = refined_strategy.generate_signals(self.dataset.train_prices)
        refined_train_bt = Backtester(self.dataset.train_prices).evaluate(refined_train_signals)

        refined_test_fitness = None
        refined_test_bt = None
        if evaluate_on_test:
            refined_test_signals = refined_strategy.generate_signals(self.dataset.test_prices)
            refined_test_bt = Backtester(self.dataset.test_prices).evaluate(refined_test_signals)
            refined_test_fitness = refined_test_bt.fitness

        refined_result = ExperimentResult(
            algorithm_name=param_optimizer.name,
            strategy_type=template_strategy.type,
            opt_result=refined_opt_result,
            backtest_result=refined_train_bt,
            train_fitness=refined_train_bt.fitness,
            test_fitness=refined_test_fitness,
            metadata={
                "strategy_description": refined_strategy.describe(),
                "test_backtest": refined_test_bt,
            },
        )

        improvement = refined_result.train_fitness - gp_result.train_fitness

        return {
            "gp_result": gp_result,
            "refined_result": refined_result,
            "improvement": improvement,
            "improvement_pct": (
                (improvement / gp_result.train_fitness * 100)
                if gp_result.train_fitness != 0
                else 0.0
            ),
        }
