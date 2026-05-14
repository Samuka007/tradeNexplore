# 课程项目提交清单

## 已完成内容

### 核心实验（14 个）

- [x] **01-消融实验**：因果 WMA、制度变迁、交易成本、过拟合（H1-H3）
- [x] **02-Walk-forward**：滚动窗口验证（dual_crossover，修正步进 bug 后 5 窗口）
- [x] **03-鲁棒优化**：多窗口同时优化（PSO 38.5% 胜率，GP 退化）
- [x] **04-结构复杂度**：7 个实验对比（trivial/MACD/position/GP original/extended/penalty）
- [x] **05-MoE**：Mixture-of-Experts（K-means 分割，有害）
- [x] **06-仓位控制**：离散 vs 连续信号
- [x] **07-PSO 粒子/迭代**：4 种配置 trade-off
- [x] **08-PSO 惯性**：3 种惯性策略对比
- [x] **09-GP 惩罚**：5 λ × 3 depth = 15 次运行
- [x] **10-仓位 scale**：8 个 scale 值扫描
- [x] **11-GP 种群/代数**：4 种配置 trade-off
- [x] **12-GP 函数集**：minimal/original/extended 三档对比
- [x] **13-Walk-forward PSO**：position_sma，修正步进后 5 窗口
- [x] **14-GP+PSO Hybrid**：结构 + 参数优化
- [x] **15-GP Warm-start**：初始种群注入人类规则（**完成！+40% 提升**）
- [ ] **16-GP+PSO λ 扫描**：不同惩罚系数下的 hybrid（运行中）

### 补充分析

- [x] **多种子验证**：10 seeds，PSO 10/10 > BH，GP 0/10 < BH
- [x] **手续费敏感性**：0%-5% 费率扫描，盈亏平衡点 4.4%
- [x] **市场阶段分析**：COVID/Bull/Bear 三阶段对比
- [x] **PSO 收敛分析**：Basin A 深+慢 vs B 浅+快
- [x] **Basin 分析**：少交易 = 高收益
- [x] **2D Landscape**：Basin A (120,180) vs B (35,100)
- [x] **Equity Curves**：策略净值曲线对比
- [x] **GP 预算敏感**：50×30 > 100×50 > 150×75

### 代码修复

- [x] Walk-forward 步进 bug（train_years → test_years）
- [x] Backtest 第 0 天阻塞
- [x] Backtest Infinity 溢出保护

### 文档

- [x] FINAL_REPORT.md：完整项目报告
- [x] EXECUTIVE_SUMMARY.md：执行摘要
- [x] RESULTS_TABLE.md：全实验结果汇总表
- [x] NOTES.md：实验洞察记录
- [x] KNOWN_ISSUES.md：已知问题与修复
- [x] INDEX.md：项目索引
- [x] README.md：项目入口
- [x] 14 个实验中文报告

### 代码质量

- [x] tiny_bot/ 包（~8 文件，~600 行）
- [x] 所有实验独立脚本
- [x] 结果 JSON 文件
- [x] Git 版本控制（51+ commits）

## 待完成项

- [ ] exp15 结果（后台运行中）
- [ ] exp16 结果（后台运行中）
- [ ] 更新 RESULTS_TABLE.md 包含 exp15/16
- [ ] 更新 FINAL_REPORT.md 包含 exp15/16
- [ ] 可能需要的额外实验（如用户要求）

## 建议的课程报告结构

1. **引言**：问题背景、核心问题
2. **方法论**：策略设计、优化算法、评估协议
3. **实验结果**：按主题分组（PSO/GP/对比/评估协议）
4. **讨论**：PSO vs GP、交易频率、制度变迁
5. **结论**：核心发现、局限性、未来方向
6. **附录**：实验目录、数据表格

## 可用的写作素材

| 素材 | 位置 |
|------|------|
| 实验洞察 | reports/NOTES.md |
| 结果汇总 | reports/RESULTS_TABLE.md |
| 详细报告 | exp/XX/report_zh.md |
| 数据文件 | exp/XX/results.json, analysis/ |
| 图表数据 | analysis/landscape/*.json |
