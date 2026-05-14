import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP

SEED = 0
LAMBDAS = [100, 250, 500, 750, 1000, 2000, 5000]
DEPTHS = [5, 7]

df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)

np.random.seed(SEED)
results = []

for lam in LAMBDAS:
    for depth in DEPTHS:
        gp = GP(pop_size=75, generations=20, seed=SEED, parsimony_penalty=lam)
        gp.max_depth = depth
        r = gp.optimize(lambda tree: backtest(train, gp.evaluate(tree, train))['final_cash'])
        best = r['best']
        test_cash = backtest(test, gp.evaluate(best, test))['final_cash']
        results.append({"seed": SEED, "lambda": lam, "depth": depth, "test": test_cash, "tree_size": gp._tree_size(best), "tree": repr(best)})
        print(f"seed={SEED} λ={lam} depth={depth}: test=${test_cash:.0f} tree={gp._tree_size(best)}")

with open(f'results_seed{SEED}.json', 'w') as f:
    json.dump({"bh": bh, "results": results}, f, indent=2)
