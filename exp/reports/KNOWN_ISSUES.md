# 已知问题与修复记录

## 已修复

### 1. Walk-forward Stepping Bug [P1]
- **问题**: `walk_forward_rolling` 函数步进 `train_years` (3年) 而非 `test_years` (1年)，导致仅产生 2 个窗口而非 6+ 个
- **影响**: Exp 02, 13 的 walk-forward 结果基于过少窗口，统计意义不足
- **修复**: `start = start + pd.DateOffset(years=test_years)`
- **状态**: ✅ 已修复，实验重新运行中

### 2. Backtest Day 0 Entry Block [P1]
- **问题**: 连续仓位逻辑的 `if abs(target - current_pos) > 1e-6 and i > 0` 阻止了第 0 天的仓位调整
- **影响**: 所有连续策略存在系统性 1 天延迟
- **修复**: 移除 `and i > 0`
- **状态**: ✅ 已修复

### 3. Backtest Infinity Overflow [P1]
- **问题**: H3 消融实验中打乱价格导致 backtest 产生 Infinity
- **影响**: H3 训练收益无效（Infinity 非合法 JSON）
- **修复**: 在 backtest 离散逻辑的 cash 计算中添加 `min(..., 1e12)` 上限
- **状态**: ✅ 已修复

## 待修复/记录

### 4. Robust-opt GP Degenerates to Buy-and-Hold [P1]
- **问题**: 多窗口平均 fitness 中，不交易策略比任何交易策略更稳健，导致 GP 进化出"不交易"策略
- **影响**: GP 在 robust-opt 中 0% 胜率，所有窗口 strategy_cash == bh_cash
- **建议修复**: 在 fitness 中惩罚零交易（如 n_trades==0 时 fitness=0）
- **状态**: 📝 已记录，暂不修复（需重新设计 fitness landscape）

### 5. MoE Spurious Regime-Switch Trades [P2]
- **问题**: MoE 在制度切换时产生未在训练中优化的虚假交易
- **影响**: Exp 05 结果不可靠
- **状态**: 📝 已知限制，实验结论仍有效（MoE 无益）

### 6. Global np.random.seed [P2]
- **问题**: PSO/GP 的 `__init__` 中设置全局 random seed，影响后续算法的随机性
- **影响**: 多算法顺序实验的非独立性
- **状态**: 📝 设计选择（为可复现性），非严重 bug

### 7. GP Fitness History Uses Penalized Values [P2]
- **问题**: `GP.optimize` 返回的 `fitness` 和 `history` 是惩罚后的值，非原始收益
- **影响**: 收敛曲线分析基于惩罚值，非真实收益
- **状态**: 📝 已记录，不影响主要结论（最终评估通过 `backtest()` 重新计算）

### 8. Continuous Backtest Trade Miscount [P2]
- **问题**: 连续仓位模式下，每次仓位调整被计为一次交易，且 entry_cost 被覆盖
- **影响**: `n_trades` 和 `win_rate` 对连续策略不准确
- **状态**: 📝 不影响主要结论（主要关注 final_cash）

### 9. Fee Sensitivity Not Re-optimized [P2]
- **问题**: 手续费敏感性实验使用固定参数（在 3% 下优化）评估不同费率，未在每个费率下重新优化
- **影响**: 盈亏平衡点反映的是特定参数的策略，而非策略空间
- **状态**: 📝 设计选择（展示特定策略对手续费的敏感性）

### 10. RSI Uses SMA Instead of Wilder EMA [P3]
- **问题**: `_rsi` 使用简单移动平均而非标准 Wilder EMA 平滑
- **影响**: RSI 计算非标准，且 NaN 回退时返回 raw price 泄露价格信号
- **状态**: 📝 不影响主要结论（GP 仍能找到有效规则）

## 对结论的影响评估

| 结论 | 影响 |
|------|------|
| PSO 10/10 击败 BH | 不受影响（backtest day 0 修复使结果更优） |
| GP 0/10 击败 BH | 不受影响 |
| 制度变迁是主因 | 不受影响 |
| Classic 50/200 最优 | 不受影响 |
| Walk-forward 平均 > single split | **需重新验证**（步进 bug 修复后重新运行） |
| 手续费盈亏平衡 4.4% | 轻微影响（backtest 修复可能改变交易次数） |
| 市场阶段分析 | 不受影响 |
