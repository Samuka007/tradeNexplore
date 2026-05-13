"""Tiny trading bot: PSO + GP on BTC data."""

from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.pso import PSO
from tiny_bot.gp import GP, GPNode
from tiny_bot.experiment import run_experiment

__all__ = [
    "load_btc_data",
    "backtest",
    "buy_and_hold",
    "PSO",
    "GP",
    "GPNode",
    "run_experiment",
]
