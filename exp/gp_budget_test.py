import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
sys.path.insert(0, '.')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP

df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)

configs = [
    ("Low", 50, 30),
    ("Medium", 100, 50),
    ("High", 150, 75),
]

SEED = 42
results = []
for label, pop, gen in configs:
    gp = GP(pop_size=pop, generations=gen, seed=SEED, parsimony_penalty=1000)
    gp.max_depth = 7
    r = gp.optimize(lambda tree: backtest(train, gp.evaluate(tree, train))['final_cash'])
    best = r['best']
    test_cash = backtest(test, gp.evaluate(best, test))['final_cash']
    results.append({
        "label": label, "pop": pop, "gen": gen,
        "train": r['fitness'], "test": test_cash,
        "tree_size": gp._tree_size(best), "tree": repr(best)
    })
    print(f"{label} ({pop}x{gen}): train=${r['fitness']:.0f} test=${test_cash:.0f} tree={gp._tree_size(best)}")

print(f"\nBH: ${bh:.0f}")
for r in results:
    print(f"{r['label']}: test=${r['test']:.0f}")

with open('exp/gp_budget_test.json', 'w') as f:
    json.dump({"bh": bh, "results": results}, f, indent=2)
