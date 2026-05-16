"""
Exp 20: GP population trajectory analysis.

Record train/test fitness and tree size per generation to show
overfitting dynamics.
"""
import sys, json
sys.path.insert(0, '../..')

import numpy as np
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest
from tiny_bot.gp import GP

train, test, df = load_btc_data()

def make_fitness(train, test):
    def fitness(tree):
        gp = GP(pop_size=1, generations=1)
        sig = gp.evaluate(tree, train, continuous=True)
        return backtest(train, sig)['final_cash']
    return fitness

train_fit = make_fitness(train, test)

def run_gp_lambda(lam, seed=42):
    np.random.seed(seed)
    gp = GP(pop_size=75, generations=20, seed=seed, parsimony_penalty=lam)
    
    # Patch GP to record trajectory
    trajectory = []
    original_eval_all = gp.optimize
    
    # We need to monkey-patch or just do the loop manually
    pop = [gp._random_tree() for _ in range(gp.pop_size)]
    
    def _eval_all(population):
        return [train_fit(ind) for ind in population]
    
    raw_fit = _eval_all(pop)
    fit = [f - gp.parsimony_penalty * gp._tree_size(ind) for f, ind in zip(raw_fit, pop)]
    
    for gen in range(gp.generations):
        best_idx = int(np.argmax(fit))
        best = pop[best_idx]
        
        # Test fitness
        test_sig = gp.evaluate(best, test, continuous=True)
        test_ret = backtest(test, test_sig)['final_cash']
        
        trajectory.append({
            'gen': gen,
            'train': float(raw_fit[best_idx]),
            'test': float(test_ret),
            'tree_size': int(gp._tree_size(best)),
            'fitness_with_penalty': float(fit[best_idx]),
        })
        
        # Evolve
        new = []
        new.append(pop[best_idx].copy())
        while len(new) < gp.pop_size:
            r = np.random.rand()
            if r < 0.9:
                p1 = gp._select(pop, fit)
                p2 = gp._select(pop, fit)
                new.append(gp._crossover(p1, p2))
            elif r < 1.0:
                p = gp._select(pop, fit)
                new.append(gp._mutate(p))
            else:
                new.append(gp._select(pop, fit).copy())
        pop = new
        raw_fit = _eval_all(pop)
        fit = [f - gp.parsimony_penalty * gp._tree_size(ind) for f, ind in zip(raw_fit, pop)]
    
    return trajectory

for lam in [0, 500]:
    traj = run_gp_lambda(lam, seed=42)
    with open(f'trajectory_lambda_{lam}.json', 'w') as f:
        json.dump({
            'lambda': lam,
            'trajectory': traj,
        }, f, indent=2)
    peak_test = max(traj, key=lambda x: x['test'])
    final = traj[-1]
    print(f"lambda={lam}: peak_test=${peak_test['test']:.0f} at gen {peak_test['gen']}, final_test=${final['test']:.0f}, final_size={final['tree_size']}")
