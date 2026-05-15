# 学术 Cosplay 问题清单

## 1. "overturning a single-seed claim that favoured lambda = 1,000"

**问题**: 这个 "claim" 不是任何文献中的结论，是我们自己的 exp09 用单种子跑出来的结果。用 "overturn" 暗示我们推翻了一个已发表的错误结论，实际上只是纠正了自己的初步实验。

**应改为**: "correcting our own earlier single-seed result (Exp.~09) that favoured..."

## 2. "This overturns our earlier claim and aligns with..."

**问题**: 第一，"our earlier claim" 并不存在——我们之前没有在正式论文中发表过这个 claim，只是在草稿中写过。第二，"overturns + aligns with" 这种写法营造了一种"我们先提出了一个错误理论，然后推翻了它，最终与经典文献一致"的学术英雄叙事，但实际上我们只是从一组 baseline 实验里学了一点东西。

**应改为**: 诚实地说 "Our initial working hypothesis, that representation dominates algorithm selection, proved too simplistic."

## 3. "Following White and Sullivan et al., we note that the data-snooping problem... warrants Bonferroni correction"

**问题**: White 2000 和 Sullivan 1999 是金融交易规则多重比较检验的顶级方法论文献（SPA/RC 和 bootstrap reality check）。我们引用了它们，然后说"所以我们做了 Bonferroni 校正"。这是借顶级文献的光环来装饰最基础的多重比较校正——就像引用了费曼来支持你用了计算器。

**应改为**: 要么不做 Bonferroni（直接报告原始 p 值并说明局限性），要么诚实地说 "we apply a simple Bonferroni correction as a conservative heuristic, acknowledging that more sophisticated methods such as White's reality check or Sullivan's bootstrap are beyond this course's scope."

## 4. "aligns with Agapitos et al. on data-mining bias"

**问题**: Agapitos 2010 讨论的是如何用 bootstrapped null model 避免 GP 交易规则中的 data-mining bias。我们引用来支持 "single-seed optimisation is unreliable"——但 Agapitos 的主要贡献不是论证 single-seed 不可靠，而是提出了一种避免 data-mining bias 的方法。这是借文献光环支持一个相关但不完全匹配的论点。

**应改为**: 直接说 "single-seed results are unreliable, as the high variance across seeds demonstrates"，不需要引用 Agapitos 来支持这个常识性论断。如果引用 Agapitos，应该说他提出了什么方法以及为什么我们没有采用。

## 5. "This reveals a fundamental distinction: GP's selection is discrete (keep/kill), while PSO's is continuous (weighted velocity update)"

**问题**: "fundamental distinction"（根本性区别）是过誉。Tournament selection 选最好的，velocity update 做加权平均——这确实是操作层面的区别，但说它是 "fundamental" 过度包装了。两页后就在 Conclusion 里把它列为 "three algorithmic insights" 之一，从很有限的实验（一个数据集、一个表示）提炼出"fundamental"规律，是典型的从 n=1 过度推广到普遍理论。

**应改为**: 降级为 "a mechanistic difference" 或 "an operational difference"。

## 6. "representation determines attractor geometry while the algorithm determines exploration reliability"

**问题**: 这句话被包装成一个普遍的理论框架（abstract 和 conclusion 都出现了），但它完全基于一个控制实验：在一个数据集（BTC 2014-2022）上，一个表示（position_sma）上，两个算法（PSO 和 restricted GP）的对比。从一个控制实验提炼出"representation determines... algorithm determines..."这样的普遍命题，是严重的过度推广。

**应改为**: 加上限定语，如 "On this problem, our causal test suggests that..." 或 "For this parametric landscape..."

## 7. "The causal test is the critical evidence"

**问题**: "critical evidence" 暗示没有它整个论证就崩塌了——确实如此——但这也暗示我们之前的所有比较（PSO vs GP unrestricted）在没有控制实验的情况下是站不住脚的。这恰恰暴露了我们报告的一个深层问题：为什么控制实验（Exp 18）是第18个实验，而不是第1个？

**应改为**: 诚实地说 "The causal test (Exp.~18), which we should have run earlier, resolves the confound that plagued all previous comparisons."

## 8. "three algorithmic insights emerge"

**问题**: "insights" 这个词暗示这些是深刻的新发现。但实际上：(1) PSO 在连续参数空间上稳定是教科书上写的东西；(2) GP 需要正则化也是教科书上的东西（Koza 1992 就讨论了 bloat）；(3) walk-forward 对离散选择算法不友好是一个可以从机制直接推导出的推论。把它们包装成 "insights" 是在把我们的 baseline 实验伪装成理论贡献。

**应改为**: "Three patterns emerge" 或 "Three observations recur across experiments"。

## 9. Abstract 中的 "A causal control experiment"

**问题**: "causal control experiment" 这个措辞暗示了我们用随机化或设计来建立了因果关系。但实际上我们的"控制"只是把 GP 限制到同一个表示——这是一个很好的设计，但它没有随机化，没有反事实框架，没有控制混淆变量。把它叫做"causal control experiment"是在用实验设计的术语来包装一个参数限制。

**应改为**: "A control experiment" 或 "A restricted-representation experiment"。

## 10. "principled complexity control, not a hack"

**问题**: "principled" 暗示这个正则化方法是基于理论的（如 MDL、SRM、PAC）。但实际上 lambda=500 是我们从 42-run 网格搜索中调出来的，没有任何理论推导证明 500 是最优值。说它是"principled"是在把经验调参伪装成理论驱动的设计。

**应改为**: "an empirically tuned complexity control" 或 "a regularisation coefficient identified by grid search"。
