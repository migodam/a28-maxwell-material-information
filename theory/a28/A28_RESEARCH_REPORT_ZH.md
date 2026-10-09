# A28 — 从精确 Maxwell 反馈到材料决策保持的推断

## 独立研究报告：科学诊断、条件性理论、文献审计与最小决定性实验

**日期：2026-10-10。研究状态：理论与小矩阵验证已完成；新网络训练、真实几何全波新实验尚未运行。**

## 研究裁决

**暂停以在线 OM、TASK_R32 或完整 feedback correction 为默认核心的新架构。下一阶段应研究：在合法材料结构之间，Maxwell 非线性究竟增加了多少可用判别信息；如何把这些信息变成保持材料决策目标的训练信号，而不是在部署时反复把一张点估计图推向更低的数据残差。**

最值得推进的主线是：

\[
\boxed{\text{结构化材料假设}\ \longrightarrow\ \text{Maxwell 条件似然}\ \longrightarrow\ \text{材料任务的后验期望损失}\ \longrightarrow\ \text{廉价摊销推断}}
\]

这不是建议立即训练一个 Bayesian 网络。首先应做一个无训练的决定性实验，检查上述链条在 A25–A27 的材料分布上是否真的存在增量价值。

本轮最重要的理论结果不是另一个传播误差界，而是三个直接改变设计的关系：

1. **电流预测误差对材料的影响，由“本构残差在跨源总场方向上的投影”决定，而非普通 current-MSE。** 但这只保证指定材料 pullback 的准确性，不保证反问题弱方向可恢复。
2. **后验平均材料与后验平均电流之间存在一个材料—内部场条件协方差源。** 忽略它，可能使最优电流估计经过传统解析材料更新后，不再是最优材料估计。
3. **完整 Maxwell 应优先约束材料假设或联合后验，而不能不加区分地约束所有点估计统计量。** 对平方材料损失，精确后验均值经任何非零确定性修正都会增加条件风险，即使修正后数据一致性更好。

第二、三点并不意味着“后验均值违反 Maxwell”。均值材料仍然是合法材料，并有自己的 Maxwell 解；不成立的是：它的 Maxwell 解必然等于各个材料假设的场解之平均，或必然拟合同一组观测。

**创新边界：** 非线性期望不交换、Bayesian distillation、近场监督和条件生成都已有明确近邻。强 TAP 的潜在贡献必须来自这里的 Maxwell 特定误差结构、条件风险解释、可计算的材料判别准则，以及独立的成像—延迟收益；不能只给既有损失换一个名字。[L4,L8–L13]

---

# Part I — Scientific Diagnosis

## 1. 核对范围与证据等级

本报告核对了三个用户指定的固定版本：

| 项目 | 固定 commit | 本轮直接核对的关键内容 |
|---|---|---|
| A27 | `325cc8a5f99d686ce1b1570296f94b0df1e90141` | 六份指定报告：recommendation、mechanism、observability、post-decoder correction、latency、intermediate supervision |
| A26 | `2603f3bd16c48e48587fedc4363975779def17f5` | 研究报告；`decoder.py`；`encoder.py` 的初始化、observable 构造和实际 `encode`；`training.py` 的真实 loss、update 和 validation |
| A25 | `59d8cc57c9da99ccb7978febc308193c43d6f3e5` | feedback headroom、最终裁决及其中的配对统计、前瞻开发验证与成本限制 |

同时读取了 Chen 专著第 6.4 节的 SOM/TSOM 相关内容，并独立检索原始论文及近期文献。A12/A17 的相关机制在这里重新作条件性推导；本轮没有重新审计它们全部历史代码。仓库大规模数值结果是**已核对报告和部分实现的既有证据**，不是本轮重新运行的结果。[A25–A27,L1]

本轮实际执行的是随附 `A28_theory_checks.py`：小规模复矩阵、实 GN 二次模型、两单元反例。没有 GPU 训练，没有把有限矩阵反例包装成真实几何 Maxwell 成像实验。

## 2. A25 与 A27 并不矛盾，但不支持同一个普遍命题

A25 的前瞻性无网络开发验证，在 weak background、amplitude 1.2、TV 权重乘 0.1、12 次更新的窄 regime 中，F 相对 L 的**逐 parent 相对收益均值**为 10.517%，区间约 [6.594%,15.316%]；比值型均值收益则是 9.418%。不同统计量不能混用。四家族中 smooth 的收益约 23.16%，而 shell、asymmetric、mixed-scale 分别约 6.73%、3.89%、8.29%。它证明了特定条件下完整反馈有真实材料收益，并未证明复杂结构统一获得 10% 以上收益。[A25]

A27 在已经训练好的 N3 后执行的，是**一次 gradient/prox 更新，不是 GN，也不是多步重建的收益上界**。复杂三家族 F 相对 NONE 的材料改善为 1.137%，F 相对 L 的额外改善为 0.160%，而完整数据残差改善约 25.27%。同条件的 edge F1 与 support IoU 没有相应提升。[A27-P,A27-M]

因此目前支持的是：

\[
\text{feedback 的边际价值依赖于当前估计及其剩余误差，不能仅按绝对对比度判断。}
\]

至少需要同时考虑五个因素：材料假设离真实状态多远、剩余错误是否沿数据可区分方向、先验是否已解释主要可观测结构、更新是否能跨越结构/拓扑变化、以及物理更新是否优化了最终评价任务。

A25 与 A27 的初值、先验、步数、接受规则均不同。不能把两者差异唯一归因于“CNN 吃掉了所有反馈信息”，也不能据 A25 的正面 regime 宣告网络必须内置 feedback。

## 3. 当前最有力的机制：残余材料错误与梯度不对齐

A27 的代表性谱诊断得到 N3 约 96.2% 的材料误差能量在局部 weak 空间，而约 99.97% 的数据梯度能量在 strong 空间。与此同时 strong 空间仍含约 62.5% 的真实材料增量能量；误差选择性分别约为 strong 0.059、weak 5.046。因此这个现象不能只解释为“weak 空间维度更大”。[A27-O]

但需要保留以下边界：诊断仅有四个独立 parent；strong 阈值是背景最大奇异值的 5%；只分析局部导数；复材料生成分布具有固定增量相位；背景已知。它不是一个校准后验，也不是全局不可识别性定理。[A27-O]

最合理的机制判断是：**CNN 已恢复相当多的测量敏感组合，剩余的空腔、薄边界、小部件等错误不能被当前数据梯度有效纠正。** 这比“Maxwell 传播算得不够准”更能解释现有结果。但剩余错误到底是先验偏差、优化路径障碍、真正多解性，还是点估计目标差异，仍未被区分。

## 4. 必须修正一个解释：feedback/噪声范数比不是检测阈值

A27 使用的量可写为

\[
\beta=\frac{\|F(\chi)-L(\chi)\|}{\sigma\sqrt M},\qquad M=8\times32=256,
\]

其中复噪声约定为 \(n\sim\mathcal{CN}(0,\sigma^2I)\)。它比较的是反馈向量范数与**整幅噪声的 RMS 范数**。[A27-M；A26 `encoder.py`]

对于两个完全指定的均值假设，正确的白化距离是

\[
 d=\frac{\sqrt2\|F-L\|}{\sigma}=\sqrt{2M}\,\beta.
\]

等先验、共同高斯协方差下，最优二元错误率为 \(\Phi(-d/2)\)。因此 \(\beta=0.168546\) 对应 \(d\approx3.814\)，理论二元错误率约 2.83%，而不是“必然看不见”。

这**不能**直接解释为材料可识别性：F 与 L 是同一个材料假设的两种前向模型，未知材料、背景误差和候选复杂度都会改变问题。它只否定一个过强推断：**β<1 不等于反馈在统计意义上不可检测。** 反过来，β>1 也不证明反馈可以区分两个 materially different 的结构。下一阶段必须改看合法材料假设间的白化测量距离，而不只看 F−L 的总量。

## 5. 不能断言 A26 的 encoder 已经丢掉材料信息

实际 N3 输入中保留了未 box 截断的背景 BP 图，即与 \(J_b^\top y\) 等价的线性数据变换。若实背景 Jacobian 在数值上满行秩，这个从 512 维数据到 1152 维 BP 的映射可以是单射。空间图像化、源汇聚或“不是 raw data”本身不构成信息丢失的证明。[A26-E]

需要区分：数学上的非单射压缩；有限精度下的病态变换；网络无法有效利用某些特征；以及训练先验和目标本身导致的偏差。A26 的 O 特征确有 rank16 截断，但网络还有其他通道，不能将整个 encoder 等同于那个截断。

一个严格的诊断式是：对平方任务损失、\(X=T\chi\)、\(\phi=\phi(Y)\)，

\[
 R^*_{\phi}-R^*_{Y}
 =\mathbb E\big\|\mathbb E[X\mid Y]-\mathbb E[X\mid\phi(Y)]\big\|^2.
\]

这是最优估计器之间的差，不是两个有限 CNN 的误差差。测不出右侧时，就不能声称 encoder 是确定的信息损失位置。

## 6. 成本事实已经足以否定“默认每图付费”

A27 同会话 resident warm 中，BP+CNN 约 40.6 ms，feedback encoder+CNN 约 99.8 ms，删掉无关 β 计算后仍约 86.5 ms，CNN 加一次 full correction 约 259.7 ms。R correction 的约 251 ms 仍包含完整 F 接受审核，不能作为纯 reduced deployment 成本。LU 复用减少分解次数却没有减少总时延。[A27-L]

当前证据支持暂停默认在线反馈，不支持宣告所有在线物理永远无用。在线物理应在以下情形重新考虑：分布外状态确有可纠正的可观测错误；多步结构先验更新建立显著收益；或者应用要求逐图物理验收。每种情形都必须将完整成本计入 raw-to-image，不能把离线 teacher、背景缓存、decoder-only 和在线认证混成一个时间。

---

# Part II — Independent Theoretical Exploration

## 7. 可恢复性应定义在合法材料的有限差分上

采用统一 TM 离散记号，源索引暂时省略：

\[
 c_\chi=H(\chi)e_\chi,\qquad e_\chi=e^{inc}+Gc_\chi,
\]
\[
 L_\chi=I-H(\chi)G,\qquad F(\chi)=Sc_\chi.
\]

\(H\) 是材料的逐点乘法算子。所有质量坐标、实虚 packing 和白化都必须与后端一致。对两个合法材料 \(\chi_0,\chi_1\)，令 \(d=\chi_1-\chi_0\)，假设两个状态算子可逆。直接相减得到

\[
\boxed{c_1-c_0=L_1^{-1}H(d)e_0=L_0^{-1}H(d)e_1.}\tag{1}
\]

这是 source-anchored relative feedback 的有限变化形式，不要求 \(d\) 足够小。再利用 resolvent 恒等式，得到

\[
\boxed{F(\chi_1)-F(\chi_0)-J_0d
=SL_1^{-1}H(d)G L_0^{-1}H(d)e_0.}\tag{2}
\]

右侧精确展示了材料有限差分通过内部反馈改变接收响应的路径。它可能增加、减少或旋转可区分信号，不存在一般的“非线性只会增加信息”定理。

### 7.1 对算法真正有用的稳定性定义

给定明确的材料可行类 \(\mathcal K\)、噪声白化 \(W=\Sigma_n^{-1/2}\)、材料任务 \(T\)，定义

\[
\omega_{\mathcal K,T}(\epsilon)
=\sup_{\substack{\chi_a,\chi_b\in\mathcal K\\
\|W(F(\chi_a)-F(\chi_b))\|\le\epsilon}}
\|T\chi_a-T\chi_b\|.
\tag{3}
\]

若噪声满足 \(\|Wn\|\le\epsilon\)，且估计与真值均在 \(\mathcal K\)、都满足该误差容限，则任务误差不超过 \(\omega(2\epsilon)\)。证明只需对两个数据误差作三角不等式。

反过来，若存在任务差距接近 \(\omega(2\epsilon)\) 的两材料，将观测设为两前向响应的中点，那么二者都与噪声容限一致；任何确定性估计器在其中至少一个材料上的任务误差不小于该差距的一半。这给出确定性 minimax 下界。

若在 \(\mathcal K\) 上有下方割线界

\[
\|W(F(\chi_a)-F(\chi_b))\|\ge\kappa\|T\chi_a-T\chi_b\|,
\]

则误差上界是 \(2\epsilon/\kappa\)。**这才是“结构先验让 weak 方向可恢复”的一种有意义的条件性定理：先验缩小允许比较的材料对，而非凭空增加测量。**

它也是实验中可反驳的量。找到一对结构差异很大、白化响应很近的合法材料，就能否定该类上的过强稳定性主张；没有找到反例则不能证明稳定性。

### 7.2 局部 Jacobian 的 weak 不等于全局不可识别

标量例子 \(F(x)=x^2\) 在 \(x=0\) 的导数为零，但在先验域 \(x\ge0\) 上仍然单射，只是零附近逆映射不满足线性 Lipschitz 稳定。允许正负两个分支后，歧义又出现。由此可见，非线性、先验可行域和有限幅度同时决定结论。

维数同样需要严谨解释：完整复网格的 1152 维开集不可能经连续映射单射到 512 维数据空间。固定复相位但允许任意 576 个实像素的开集也有这个问题。可是实际生成器若只覆盖低维的形状族，则不能用上述维数论否定其可识别性。A28 应明确研究哪一个材料类，而非同时把“完整网格自由度”和“低维标签先验”当作自己的优势。

## 8. 不建议硬分成 physics 部分与 prior 部分

设材料由结构坐标 \(\chi=D(z)\) 表示，局部先验协方差为 \(C_z\)。线性高斯近似中的合适分析矩阵是

\[
 A=\Sigma_n^{-1/2}J_\chi D_z C_z^{1/2},
\]

其 prior-whitened 后验协方差为 \((I+A^\top A)^{-1}\)。软信息份额是 \(\lambda/(1+\lambda)\)，依赖噪声、先验、状态和表示，而非背景奇异值的一个任意阈值。这与 likelihood-informed subspace 和 goal-oriented Bayesian approximation 的思想一致，不是新的命名。[L6,L7]

更本质的表示是

\[
 p(a,b\mid y)=p(a\mid y)\,p(b\mid a,y),
\]

其中 \(a\) 是测量较支持的结构组合，\(b\) 是剩余结构。当似然只依赖 \(a\) 时，正确结果是 \(p(b\mid a,y)=p(b\mid a)\)，不是独立的 \(p(b)\)。

简单反例：先验协方差为 \(\begin{psmallmatrix}1&\rho\\\rho&1\end{psmallmatrix}\)，只测量 \(y=x_1+n\)，\(\operatorname{Var}(n)=\sigma^2\)。虽然 \(x_2\) 属于测量 Jacobian 的零空间，

\[
\operatorname{Var}(x_2\mid y)=1-\frac{\rho^2}{1+\sigma^2}.
\]

取 \(\rho=0.99,\sigma^2=0.01\)，后验方差约 0.0296，而非原先的 1。这个恢复依赖先验相关性，相关性在 OOD 中失效时就不再可靠。

**设计结论：** 保留所有合法的数据支持，不强制 NN 只输出固定 weak-SVD 补空间；使用多尺度结构、形状—材料耦合或条件概率表达。只有针对一个已经校准的局部模型，才使用显式软分解。\(\chi=\chi_{physics}+\chi_{prior}\) 可以是计算参数化，但一般不是统计上唯一或物理上可解释的分解。## 9. 哪些 current error 对材料无害，哪些危险？

这里可以得到一个精确的条件性结论。设真实状态为 \((\chi^*,c^*,e^*)\)，网络预测

\[
\hat c=c^*+\delta c,\qquad \hat e=e^{inc}+G\hat c=e^*+G\delta c.
\]

定义传统的跨源最小二乘材料 pullback

\[
 Q_i(\hat c)=\frac{\sum_s\overline{\hat e_{is}}\hat c_{is}}{D_i},
\qquad D_i=\sum_s|\hat e_{is}|^2>0.
\]

这个 ratio 本身不是新算法；CSI、SOM-Net 及 A27 的设计中已有相关更新。[L2,L3,A27-S]

### 命题 1：材料 pullback 的精确误差分解

令 \(r_s=L_{\chi^*}\delta c_s\)。则

\[
\boxed{Q_i(\hat c)-\chi_i^*
=\frac{\sum_s\overline{\hat e_{is}}\,r_{is}}{D_i}.}\tag{4}
\]

若把第 \(i\) 个像素的跨源向量写为 \(r_i,\hat e_i\)，令 \(P_{\hat e_i}\) 为投影到 \(\hat e_i\) 的复正交投影，则

\[
\boxed{\|r_i\|^2
=D_i|Q_i(\hat c)-\chi_i^*|^2
+\|(I-P_{\hat e_i})r_i\|^2.}\tag{5}
\]

**证明。** 本构关系给出

\[
\hat c-H(\chi^*)\hat e
=\delta c-H(\chi^*)G\delta c=L_{\chi^*}\delta c.
\]

左右乘跨源场共轭并求和、除以 \(D_i\)，得到式 (4)。式 (5) 是 \(r_i\) 沿 \(\hat e_i\) 与其正交补的 Pythagoras 分解。

**物理解释。** 决定材料偏差的不是任意 current error，而是该误差经过真实本构算子后，能够伪装成“所有源共享的材料变化”的那一部分。正交部分对这个 ratio 没有直接偏移，却可能意味着跨源状态不一致。

所以，一个更合理的 task seminorm 是

\[
\|\delta c\|_{mat,\chi^*,\hat e}^2
=\sum_i\left|D_i^{-1}\sum_s\overline{\hat e_{is}}
 (L_{\chi^*}\delta c_s)_i\right|^2,
\]

而不是 \(\|\delta c\|^2\) 或 \(\|S\delta c\|^2\)。这直接给出指定 pullback 的材料误差，但使用了训练真值和预测场，不是可直接部署的可恢复性证书。

### 9.1 两个反例和一个危险点

当 \(G=0\)、跨源场为 \((1,1)\)、电流误差为 \((a,-a)\) 时，current-MSE 可以任意大，材料 ratio 误差却为零。相反，当 \(D_i\) 很小时，少量相关本构误差就会产生很大的材料偏差：

\[
 |Q_i-\chi_i^*|\le\|r_i\|/\sqrt{D_i}.
\]

因此暗场区域不应强行执行无偏 ratio。若使用带锚点的 ridge：

\[
 Q_i^\lambda=\frac{\sum_s\overline{\hat e_{is}}\hat c_{is}+\lambda\chi_{0i}}{D_i+\lambda},
\]

误差分子中会明确出现 \(\lambda(\chi_{0i}-\chi_i^*)\)。这是先验偏差，不是凭一个 denominator floor 就消失的数值细节。

另一个反例与反问题相关：\(J=[1,0]\)，真值 \((0,1)\)，预测 \((0,0)\)。数据残差与 GN 更新误差都可以为零，材料误差却为 1。**即使一个学习目标精确保持某次 material update，也不意味着它恢复了数据未支持的材料结构。**

## 10. A12/A17 应如何进入学习目标，而不是继续选 current modes？

对同一个冻结实坐标二次模型，定义

\[
 Q_F(s)=\tfrac12\|r_F+J_Fs\|^2+\tfrac12s^\top\Lambda s+p^\top s,
\quad H_F=J_F^\top J_F+\Lambda\succ0.
\]

近似模型为 \((r_A,J_A)\)，正则和线性先验项相同。令

\[
 E=J_F-J_A,\quad d=r_F-r_A,\quad \rho_A=r_A+J_As_A.
\]

利用近似解的一阶条件，完整模型在 \(s_A\) 的梯度是

\[
\zeta=J_F^\top(d+Es_A)+E^\top\rho_A.
\]

从而精确得到

\[
\boxed{s_A-s_F=H_F^{-1}\zeta,}\tag{6}
\]
\[
\boxed{Q_F(s_A)-Q_F(s_F)=\tfrac12\zeta^\top H_F^{-1}\zeta.}\tag{7}
\]

对材料任务 \(T\)，有意义的近似目标是 \(\|TH_F^{-1}\zeta\|\)。它要求准确的是残差缺陷、\(Es_A\) 以及 \(E^\top\rho_A\)，并不要求全 current、全 Jacobian 或所有传播方向都同等准确。

把 \(E=O_A L^{-1}I_A\) 或低秩 transfer \(E=Y\Gamma^{-1}N\) 代入，就能沿 A12/A17 的反馈通道生成训练探针与误差权重。在冻结 residual、state、regularization 的低秩修改下，GN 变化落在 \(H_A^{-1}\operatorname{Range}[J_A^\top Y,N^\top]\) 一类低维支撑中。这种支撑适合**离线任务探针**，不意味着又需要构造一个在线 OPM 网络。

但需要两条限制。第一，式 (6) 保持的是指定局部优化问题，不是未知真值；先验不对时，完美复现其更新仍可能得出错误结构。第二，\(H_F^{-1}\) 或相应 adjoint 探针可能昂贵，若没有廉价 teacher 或训练缓存，就会重复 A25 的成本失败。此处的二次缺陷恒等式不应被单独包装成一项新 TAP 算法贡献。

## 11. 最关键的结构关系：后验均值有一个协方差源

内部标签是否有价值，不能只问“网络能否预测 current”，还必须问“这个 current head 在统计上预测的究竟是什么”。

给定测量 \(y\)，定义条件均值

\[
\mu_\chi=\mathbb E[\chi\mid y],\quad
\mu_{c,s}=\mathbb E[c_s\mid y],\quad
\mu_{e,s}=e_s^{inc}+G\mu_{c,s}.
\]

假设条件二阶矩存在。定义**不含复共轭的双线性条件交叉矩**

\[
\kappa_s=\mathbb E\left[H(\chi-\mu_\chi)(e_s-\mu_{e,s})\mid y\right].
\]

这是因为本构关系是 \(\chi e\)，而不是 \(\chi\overline e\)。在实虚展开后，可以用普通实交叉协方差实现。

### 命题 2：Maxwell 条件均值闭合

\[
\boxed{\mu_{c,s}=H(\mu_\chi)\mu_{e,s}+\kappa_s,}\tag{8}
\]
\[
\boxed{L_{\mu_\chi}\mu_{c,s}=H(\mu_\chi)e_s^{inc}+\kappa_s.}\tag{9}
\]

若 \(L_{\mu_\chi}\) 可逆，则

\[
\boxed{F_s(\mu_\chi)-\mathbb E[F_s(\chi)\mid y]
=-SL_{\mu_\chi}^{-1}\kappa_s.}\tag{10}
\]

**证明。** 将 \(\chi=\mu_\chi+\delta\chi\)、\(e=\mu_e+\delta e\) 代入 \(c=H(\chi)e\)。条件均值中的一次中心化项为零，剩下 \(H(\mu_\chi)\mu_e+\mathbb E[H(\delta\chi)\delta e\mid y]\)，得到式 (8)。再用场方程的线性性整理得式 (9)。与均值材料自身的 Maxwell 方程相减，得到式 (10)。

还有一个直接约束 current-based 网络的结论：

\[
\boxed{Q_i(\mu_c)-\mu_{\chi i}
=\frac{\sum_s\overline{\mu_{e,is}}\kappa_{is}}
{\sum_s|\mu_{e,is}|^2}.}\tag{11}
\]

因此，假如 current-MSE head 学到了最优条件均值 \(\mu_c\)，传统解析 ratio 也未必给出材料 MSE 的最优解 \(\mu_\chi\)。**两个分别最优的中间量目标，经非线性物理后处理后并不自动组成一个最优材料估计器。**

这不是对 A27 已验证的因果解释。A26 用的是材料 MSE 加 SSIM，网络有限、训练样本少，还带 box，因此不能把其输出认定为精确后验均值。这里提供的是一个此前“增加共同材料一致性”建议必须面对的条件性反例机制。

### 11.1 更严格的数据一致性为何可能损害材料风险

对平方材料损失，\(\mu_\chi\) 是最优点估计。任意由同一测量决定的修正 \(a(y)\) 满足

\[
\mathbb E[\|\mu_\chi+a-\chi\|^2\mid y]
=\mathbb E[\|\mu_\chi-\chi\|^2\mid y]+\|a\|^2.\tag{12}
\]

交叉项为零即为证明。这里假设决策空间允许输出条件均值；若任务强制输出一个二元结构或先验流形上的单个实体，受约束 Bayes 决策就未必是均值，必须重新按该动作空间求解。先验支持集和估计器允许输出的空间不可混淆。这不是说“所有物理修正都无效”，而是说：**当点估计已接近某个材料任务的 Bayes 最优行为，单纯用数据一致性替代该任务，可能把它推离最优点。**

有噪声时，\(\mathbb E[F(\chi)\mid y]\) 也一般不等于 \(y\)，因为条件噪声均值不必为零。这个区别必须保留。

对 NRMSE、SSIM、edge、IoU，最优决策并不一般是材料均值。故正确设计是“联合物理假设或后验 + 明确的决策损失”，不能把所有指标都交给一张平均图，再同时要求它满足所有物理约束。

### 11.2 已完成的两单元反例

取互易有限 VIE 代数

\[
G=\begin{pmatrix}0&0.8\\0.8&0\end{pmatrix},\qquad
 e^{inc}=(1,1)^\top,\qquad S=(1,1),
\]

两个等先验材料为 \(\chi_A=(0.1,1)\)、\(\chi_B=(1,0.1)\)。结果如下：

| 量 | 数值 |
|---|---:|
| 两个材料的测量 | 均为 1.346153846 |
| 材料条件均值 | (0.55, 0.55) |
| 平均电流 | (0.673076923, 0.673076923) |
| 平均内部场 | (1.538461538, 1.538461538) |
| 协方差源 κ | (−0.173076923, −0.173076923) |
| 均值材料的前向测量 | 1.964285714 |
| 平均电流经普通 ratio 得到的材料 | (0.4375, 0.4375) |
| ratio 材料的测量 | 1.346153846，精确拟合 |
| 材料均值的条件平方风险，按两个像素求和 | 0.405 |
| ratio 材料的条件平方风险 | 0.4303125 |

**数据一致性变成精确，材料平方风险却增加 6.25%。** 两个端点的本构矩阵条件数约 2.413，不是依靠接近奇点制造的数值失效。

这是一个互易有限矩阵反例，尚不是 2D 几何 Green 函数生成的 Maxwell benchmark。它否定普遍推理，但是否主导 A27，必须在真实几何、合法结构候选中另行验证。

### 11.3 真值场加权的材料 loss 也可能改变估计目标

若 \(D^*=\sum_s|e_s^*|^2\)，优化

\[
\mathbb E[D^*|a-\chi^*|^2\mid y]
\]

得到

\[
 a_D=\frac{\mathbb E[D^*\chi^*\mid y]}{\mathbb E[D^*\mid y]},
\]

而不一般等于 \(\mathbb E[\chi^*\mid y]\)。在上述反例中，这个加权最优解为每像素约 0.338235，普通材料平方风险约 0.494689，比条件均值高约 22.15%。

这种加权可以对某种物理任务有用，但不能自动声称改善未加权材料恢复。Liu 等人的近场/电流物理损失已经包含与此非常接近的 teacher-field weighting；A27 的廉价备选不应被当作全新方法。[L4]

相反，在固定平方材料目标下，若正定权重只依赖观测 \(y\) 或冻结的 data-only pilot，并且不随待优化输出变化，那么条件均值仍是最优解。这提供了较安全的 task-weighted 训练原则；不过有限网络的优化效果仍需实验，不由这个统计结论保证。

## 12. 高风险新表示：学习 κ 的材料有效部分，而不是全部 current

式 (8)–(11) 给出一种可能的新中间表示。令

\[
 A_i=\sum_s\overline{\mu_{e,is}}\mu_{c,is},\quad
 B_i=\sum_s|\mu_{e,is}|^2,\quad
 K_i=\sum_s\overline{\mu_{e,is}}\kappa_{is}.
\]

当 \(B_i>0\) 时，

\[
\boxed{\mu_{\chi i}=(A_i-K_i)/B_i.}\tag{13}
\]

这只有每像素五个实统计通道：复 A、实 B、复 K，而非八源复电流和内部场全部在线输出。训练时用物理一致候选的联合条件分布计算，部署时可以只留下统计预测器或直接材料预测器。

但自由输出 K 的网络毫无保证：它总能把任意材料误差吸收掉，式 (13) 就退化为重写输出。必须通过**独立联合候选、后验覆盖、跨源条件矩与留出观测**验证 K，不能仅靠材料标签反推出 K 然后声称学会了协方差物理。

若后验集中且 current 映射二阶光滑，以实随机坐标表示材料扰动，\(\delta\chi\approx\sum_jv_j\xi_j\)、\(\mathbb E[\xi_j\xi_k]=\delta_{jk}\)，则

\[
\kappa_s\approx\sum_jH(v_j)G\,\mathcal K_{\mu,s}v_j,
\qquad \mathcal K_{\mu,s}=L_{\mu}^{-1}B_{\mu,s},\tag{14}
\]

余项在有界二阶导数条件下可按 \(\mathbb E\|\delta\chi\|^3\) 控制。这里的 \(v_j\) 是条件材料协方差方向，不是背景 weak-SVD 向量。式 (14) 把 posterior ambiguity 与反馈传播直接联系起来，适合离线 teacher，但当前 64 parents 未必足以可靠学习这样的条件矩。

这是值得保留的高风险路线，而不是下一步立即上线的架构。一般性的 Jensen gap 已由 diffusion inverse problems 和 PIDDM 等文献讨论；随机介质平均场闭合也早有传统。本报告不主张其一般概念为新。潜在差异是 **inverse-problem 条件后验下的 Maxwell κ、其材料决策偏差，以及低成本任务统计量**。[L10,L17]## 13. 主路线：用 Maxwell 评价材料假设，再蒸馏材料决策

### 13.1 为什么是条件似然，而不是 another current head

对于材料假设 \(\chi_k\)，Maxwell 的任务是计算它能否解释观测。对于一个后验分布，材料重建器的任务则是根据指定损失作决策。二者不应合并为“让输出图同时像标签、像电流、像数据、像所有中间状态”的无区分多任务目标。

设一个明确声明的有限结构先验

\[
 p_{bank}(\chi)=\sum_{k=1}^K w_k\delta_{\chi_k},\quad w_k>0,\quad\sum_kw_k=1.
\]

每个材料候选是合法的；同一已知背景下预先计算全波响应 \(f_k=F(\chi_k)\)。高斯噪声下

\[
 \pi^F_k(y)=\frac{w_k\exp[-\tfrac12\|\Sigma_n^{-1/2}(y-f_k)\|^2]}
 {\sum_jw_j\exp[-\tfrac12\|\Sigma_n^{-1/2}(y-f_j)\|^2]}.
\tag{15}
\]

这是真正使用完整 feedback 的位置：它改变合法结构之间的条件相对支持，而非只是给一张初值多加两个特征。

对材料任务损失 \(\ell(a,\chi)\)，训练廉价学生 \(a_\theta(y)\) 最小化

\[
\boxed{\mathcal L_F(\theta;y)
=\sum_k\pi_k^F(y)\,\ell(a_\theta(y),\chi_k).}\tag{16}
\]

平方损失时可简化为学习后验均值。含 SSIM、结构指标的可微替代项或相对误差时，应保留**期望损失**，不能把“对平均标签算 loss”冒充它。

### 13.2 一个可证明而且会影响训练设计的性质

假设训练中的材料和噪声确实来自所声明的有限先验及正确噪声模型，交换微分与期望合法，则

\[
 \nabla_\theta\mathcal L_F(\theta;y)
 =\mathbb E[\nabla_\theta\ell(a_\theta(Y),\chi)\mid Y=y].
\tag{17}
\]

因此它与抽取单个真值标签的监督梯度具有相同总体期望；由条件方差分解，条件化梯度的协方差不大于原随机梯度的协方差。这是 Rao–Blackwell 化，不是凭空获得额外观测。

**实际意义：** 若两个结构在噪声下确实难区分，不让它们对应的训练样本不断把网络推向互相冲突的唯一标签；由正确物理似然决定应该保留的条件材料信息。若完整 F 与线性 L 对这些结构的相对支持不同，则 F teacher 可能减少错误的歧义合并。

**最重要的限制：** 有限 bank 后验仅对这个有限先验精确。64 parents 不是材料世界的完整分布；补充候选的权重、覆盖与 OOD 偏差可能比物理精度更重要。如果几乎所有 \(\pi_k\) 都是 one-hot，则式 (16) 近乎普通监督，没有足够理由指望独立收益。若候选先验失真，所谓后验蒸馏反而会放大字典偏差。

所以这个方案不是已经证明优于 BP+CNN 的新算法，而是一个**可以用三臂实验清晰证伪、在线物理成本为零的研究候选**。Bayesian distillation 和 simulator-assisted supervision 已有先例；创新必须由 Maxwell 的材料判别机制与独立实验建立。[L11,L12]

### 13.3 为什么这仍可能胜过普通 physics-generated labels

所有使用全波合成训练集的网络都已间接接触 Maxwell；不能把“训练标签来自全波”算作 A28 独有机制。这里新增的干预是：**同一个 noisy observation 对多个合法材料假设的条件权重，以及这些权重如何影响最终材料决策。**

但它在无限数据、无限容量、正确优化的极限下并不打败普通监督的 Bayes 最优风险。潜在收益来自有限样本、难区分结构、减少梯度冲突、正确使用条件分布及更好的泛化归纳偏置。因此必须与同样接触全部候选标签和同样噪声增强的普通监督基线竞争。

## 14. 不损失材料判别力的廉价统计：条件充分性而非固定弱空间

在白化实数据中，两个候选前向均值为 \(f_a,f_b\)。共同单位协方差下，其数据对数似然比为

\[
 \ell_{ab}(y)=\left(y-\frac{f_a+f_b}{2}\right)^\top(f_a-f_b).
\]

将数据投影到 \(P=UU^\top\)，令丢失割线 \(r_{ab}=(I-P)(f_a-f_b)\)，则精确有

\[
\ell_{ab}(y)-\ell^P_{ab}(y)
=\left(y-\frac{f_a+f_b}{2}\right)^\top r_{ab}.\tag{18}
\]

在假设 a 下，这个差是均值 \(\|r_{ab}\|^2/2\)、方差 \(\|r_{ab}\|^2\) 的高斯变量；投影丢失的该对假设 KL 距离是 \(\|r_{ab}\|^2/2\)。若 bank 内所有前向割线均在 U 中，则 \(U^\top(y-f_{ref})\) 对该有限先验是精确充分统计量。

式 (1)–(2) 允许将这些割线的来源解释为材料切向和有限 feedback transfer。它与只保持当前 residual 或前几个 current singular modes 不同；它直接检查一个压缩是否抹去了材料假设的相对证据。

已有 score compression 能在局部保持 Fisher 信息，因此“physics sufficient statistics”本身也不是新概念。[L15] 这里值得研究的是**有限结构假设、非线性响应割线和材料决策误差之间的联系**。

若近似 teacher 的各候选 log-likelihood 误差，在扣除共同常数后满足 \(|\Delta\log L_k|\le\eta\)，则归一化后验的总变差距离不超过 \(\tanh\eta\)。对任务值直径为 \(D_T\) 的 bank，有

\[
\|\tilde\mu_T-\mu_T\|\le D_T\tanh\eta.
\]

对平方任务损失，这导致至多 \(D_T^2\tanh^2\eta\) 的额外条件风险；对取值在 \([0,\ell_{max}]\) 的一般损失，近似后验 Bayes 决策的额外风险可用 \(2\ell_{max}\tanh\eta\) 控制。证明分别来自后验密度比界、总变差和两个决策期望损失的比较。

这是一个条件性压缩验收标准，不是又一个需要立刻开发的 rank selection 项目。**当前仅有 512 个实数据，优先保留完整数据/BP 信息；没有必要为了一个小输入再追逐 rank32。**

## 15. Generative inference：什么时候需要，什么时候不需要？

必须先区分条件估计偏差与真实多模态。合法近等测量材料对可以证明歧义存在，但两点都可行尚不意味着两者都具有显著后验概率；还需要结构先验质量与适当概率权重。局部 profile likelihood、多个合法候选及它们之间的连接障碍，比一张 SVD 图更适合初步检查多模态。

如果主要目标是平方误差，一个独立后验样本的条件 MSE 为 \(2\operatorname{tr}C\)，后验均值的为 \(\operatorname{tr}C\)。随机生成更尖锐的图，并不能据此宣称定量材料更准。评估应分别报告材料决策图、结构/拓扑概率以及样本多样性；禁止 truth-selected best-of-K。

现阶段建议按证据升级：确定性结构先验和有限候选后验先行；证实多模态且候选覆盖不足，再尝试小型 conditional mixture 或 CVAE；只有复杂连续后验确实无法由这些模型表达、数据量与时延预算也支持时，才进入 flow/diffusion。Deep Injective Prior、Back-Projection Diffusion 已经覆盖生成材料先验、后验求解与物理编码后的条件生成，不能把进入 diffusion 本身当作突破。[L8,L9]

对于 PnP/RED、DIP、learned proximal、INR 和 neural operator，判断应基于具体机制：是否增加了与材料结构匹配的先验；是否仍需逐图多次 full-wave/adjoint；是否具有可验证的 likelihood fidelity；是否只是把求解器成本移动到了庞大训练集。ScoreField 等已将 score prior 与 current/contrast 神经表示和全波方程结合，因此“先验 + 电流 + Maxwell loss”已是明确竞争区。[L16]

## 16. 材料表示与观测方式的最小改变

表示上，优先比较普通多尺度实/虚材料图与“共享几何 + 定量材料”参数化，而不是立刻冻结一个低维 shape chart。后者适合分段均匀材料，但若错误地约束复杂混合尺度结构，同样会制造不可修正的先验偏差。所有表示都应通过合法材料割线和 OOD 结构检查，而非只比较训练误差。

观测上，当前最值得考虑的不是重新规划整套阵列，而是**一个新增频率**。原因不是高频必然更好，而是它可能改变原先混淆结构的响应方向。

在一个局部线性、高斯且先验协方差 C 正确的模型中，增加白化测量 A 后

\[
 C_{new}=C-CA^\top(I+ACA^\top)^{-1}AC,
\]

任务方差减少量为

\[
 \operatorname{tr}\left[TCA^\top(I+ACA^\top)^{-1}ACT^\top\right]\ge0.\tag{19}
\]

这里 A 应来自 \(SL^{-1}B\) 作用于实际结构先验坐标，而非 receiver/current 谱单独决定。这是局部预测；最终仍要检查有限结构候选及模型偏差。

还有一个容易忽略的无增益情形：固定材料与频率时，Maxwell 对入射源线性。若新增照明只是已有入射场的线性组合，新增无噪声数据也是已有数据的线性组合，不增加独立信号空间。它可以改变噪声平均或能量分配，但不能伪装成新增材料信息。比较时必须固定总发射能量和采集时间。

多频必须使用一致的物理参数化。若假定实介电常数与电导率不随频率变，则损耗对应的虚部对比随 \(1/\omega\) 改变；不能在所有频率机械保持复 χ 不变，又声称是在测同一材料。色散、背景和标定误差应作为 nuisance 或误差源处理。

---

# Part III — Literature and Novelty Audit

## 17. 最近邻不是单一 SOM-Net，而是四组方法

下表区分已存在的机制与 A28 可能留下的空间。没有全文访问的补充检索结果不承担决定性创新结论；核心判断基于专著章节、可访问论文正文/作者稿及原始摘要。

| 最近邻 | 已经做过什么 | A28 不能再宣称什么；剩余差异是什么 |
|---|---|---|
| SOM、TSOM、FFT-TSOM；Chen 专著 [L1,L18,L19] | 数据可确定 current 与待优化部分；状态方程约束；内部算子第二重限制；逐步扩大子空间；FFT 加速 | 不能把 observable/ambiguous 或内部传播正则当作新思想。材料风险与有限材料割线不等于 current 谱强度，是需要实证的新层次 |
| CSI / MR-CSI [L2] | 电流与材料交替求解、共享本构约束、解析材料最小化、边缘正则 | 无需每一步完整重新求前向、current-to-material ratio、joint state/data loss 都非新概念 |
| DBIM 与 GN/LM [L20] | 在变化材料状态更新物理线性化；耦合多重散射与正则化 | “在 CNN 后用真实 Jacobian 再更新”本身不是创新，也不是全部可能的更新路径 |
| SOM-Net [L3] | SOM 展开；确定性 current/BP 初始化；current、receiver field、材料相关训练约束；物理材料更新 | 新 current head、普通电流监督、多损失融合都不够。需要材料决策误差机制和公平增量 |
| Physics-Guided Loss Functions [L4] | 用内部散射场/诱导电流监督网络；实际包括 truth-total-field 加权材料偏差 | A27 的无 head 场加权备选已有强近邻。可研究该加权改变条件估计目标的具体规律，但不是首次 near-field supervision |
| Two-Step Contrast Source Learning [L5] | 确定 current → 学习 total current → 解析 contrast → U-Net refinement | “current 中间表示比直接材料更物理”不是新架构；须解释哪些误差有害、如何避免统计 pullback 偏差 |
| Learned Primal-Dual、unrolled reconstruction [L13] | 学习数据/图像空间的迭代与先验耦合 | primal-dual 双向交互、几层展开、learned prox 都是既有模板；A28 必须改变目标或可用信息，而非只改 block |
| Null-space networks [L14] | 用网络补足数据约束外的成分，并保持相应线性数据一致性 | 固定 weak complement 接 NN 不是新原则；非线性、先验相关性与 calibrated uncertainty 是尚需处理的问题 |
| Likelihood-informed / goal-oriented Bayesian approximation [L6,L7] | 先验白化的信息坐标、任务相关后验低秩结构 | 背景 SVD 重命名 posterior coordinates 不成立；应采用真实 prior/noise/task 定义 |
| Deep Injective Prior [L8] | 低维生成材料流形、MAP/后验相关推断 | latent material representation 与生成先验已经存在；需要新的材料可识别性或成本机制 |
| Back-Projection Diffusion [L9] | physics-based backprojection 编码加条件 diffusion，利用问题结构实现较快推断 | “低成本物理统计 → posterior”已有非常接近的方案。其研究是 acoustic/wideband 设置，不能直接当作本项目有损 TM 数值基线 |
| PIDDM [L10] | 明确讨论 nonlinear posterior-mean constraints 的 Jensen gap；在蒸馏后的最终生成样本上施加 PDE 约束 | 不能声称第一次发现均值物理不交换，或第一次训练有物理、部署单步。A28 的区别须是 Maxwell 条件协方差及材料决策，而非同一个一般论点 |
| Mining Gold / simulator-assisted inference [L11] | 利用模拟器内部变量、score/likelihood ratio 信息改进统计推断训练 | “内部状态是 privileged information”已有方法论；新的 EM 中间量必须具有具体材料风险含义 |
| Generalized Bayesian Posterior Expectation Distillation [L12] | 蒸馏后验期望，降低推断成本 | 后验均值蒸馏不是新算法类别。式 (16) 的价值要靠 Maxwell 材料歧义与有限训练收益建立 |
| Score compression [L15] | 以局部 score 压缩保持 Fisher 信息 | sufficient physics statistics 不是新术语；有限幅度结构判别与材料决策保持是更合适的区别 |
| ScoreField，2026 preprint [L16] | current/contrast 神经表示与 full-wave 方程、score prior 联合优化 | 普通 score-guided full-wave reconstruction 已是近期竞争方法；逐图优化不能假装成一次前馈推理 |

### 17.1 TSOM 的一个重要术语边界

Chen 专著把理想子空间交集与实际投影近似分开讨论。将内部算子的主要子空间投影到 receiver-ambiguous 空间，通常不等同于精确交集。FFT-TSOM 又以 Fourier 子空间近似内部谱子空间。不能把这些不同层次统一叫成“同一个 O/P/M 分解”，也不能由 source-anchor 的 forcing-P 为零推出 TSOM 的内部反馈作用已经被完全替代。[L1，第 6.4.2 节；L18,L19]

本轮应保留的不是特定 O/P/M 名称，而是三件事：可靠的数据锚定、所有照明共享同一材料、适当限制无数据支持的内部自由度。它们可以存在于结构后验、训练 teacher 或条件 likelihood 中，不必继续存在于一个在线 current basis 中。

### 17.2 哪些结果可能构成论文贡献，哪些只能作背景？

式 (1) 的 resolvent、式 (6) 的固定 GN 缺陷、Rao–Blackwell、局部 Gaussian 后验都属于可由已有数学直接得到的结构，不能单独撑起强算法论文。普通 current loss、physics loss、VAE 和蒸馏也不够。

更有吸引力的联合命题是：**在全波逆散射中，内部物理监督必须匹配材料决策统计量；材料—场条件协方差明确刻画传统 current pullback 的条件偏差；Maxwell finite-secant likelihood 能把真正受测量支持的结构差异传递给低时延学生，而无需在部署时求解反馈。**

这仍是研究命题。只有在真实几何 Maxwell 数据中，κ/割线诊断预测了哪些样本受益，且新的目标在同先验、同架构、同数据预算下赢过强普通监督和近邻物理损失，才值得把它升级为论文主张。本次检索未建立“文献中绝无相同公式”的排他性证明；投稿前仍需针对这一最终主张做进一步引用链检索。# Part IV — Competing Research Directions

## 18. 最终路线排序

### Primary Route：材料决策保持的 Maxwell 条件推断与摊销

主问题不是“怎样把反馈塞进网络”，而是“完整物理是否改变材料候选的正确相对支持，以及这个改变是否值得学习”。研究对象是式 (1)–(3)、(8)–(11)、(15)–(18) 的组合。

在线路径保持简单：测量与已知背景 → 不额外丢信息的 BP/数据输入 → 小型多尺度材料学生。完整 Maxwell、内部 current/field、条件材料候选和任务后验都在训练或离线诊断端。部署不新增状态相关的 full-feedback 求解、LU、TV/prox 或 current auxiliary head。沿用 N3 输入时，其原有固定背景线性算子应用仍须如实计费，不能未经 trace 就宣称整个 encoder 绝对没有 G 应用。

它的吸引力在于可以同时回答三个问题：为何更精确场拟合没有改善材料；哪些结构确实受非线性物理支持；是否能以有限训练改进换取零额外在线物理成本。

它的主要风险也很明确：有限候选 posterior 可能没有代表性，或近似退化成 one-hot；蒸馏只减少训练方差但不改善新材料；收益可能全部来自更好的普通结构先验。因此推进是**有条件 GO**，不是已经确定的 strong-TAP 论文。

### Secondary Route：强普通 BP + 结构先验，让简单方法赢

保留背景缓存和物理输入，换成有竞争力的多尺度 decoder，明确允许 thin structures、空腔和多尺度部件。必要时比较共享几何与定量对比参数，而不是强制 current latent。所有模型使用相同 passivity/box 和训练增强。

若它显著优于 A26，而 Maxwell 条件 teacher 没有额外收益，应接受结论：这一阶段的算法瓶颈主要是先验表达/训练，而不是缺少在线反馈。理论可以用于诊断与风险约束，但不要继续称其为反馈成像突破。

这条路线最稳妥、最便宜，却未必单独达到强 TAP 的新颖性要求。除非能给出更广泛的物理泛化、可识别性结论或实质性的规模化优势。

### High-Risk / High-Reward Route：后验协方差修正的内部物理统计

研究式 (13) 的 A/B/K 表示，或低维条件联合分布 \(p(\chi,c\mid y)\)，使样本级共享材料本构成立，而均值级显式保留 κ。把 A12/A17 的任务探针用于只估计会改变材料决策的 κ/电流分量。

最重要的进入条件是：真实 Maxwell 下 κ 引起的材料偏差足够大，而且可以被有限训练集中的结构规律预测。若 κ 只是无关紧要的小量，或必须依赖高成本后验采样才能获得可靠标签，就不应立刻训练这条路线。

成功时，贡献不只是 uncertainty head，而是“内部物理估计应该以何种条件统计量承载材料信息”的新规律。失败时，也会明确告诉我们 current-first 架构是否值得继续。

### Routes to Stop

停止下一轮 OPM/OM rank 变体；停止把 TASK_R32 在初始状态上的精度直接外推到所有网络状态；停止默认“CNN 后再加一次 full correction”作为论文机制；停止不区分材料估计目标的 current-MSE/teacher-field loss 堆叠；停止把 soft background SVD 当后验；停止用更小参数量、更少模式或更低 residual 代替成像—时延优势。

Diffusion 不列为永久停止，而是**当前不启动**。一次规划十一个网络、随后因成本规则一个也无法完成的执行结构也应停止。

## 19. 何时彻底停止在线 feedback 投入？

若在明确声明的材料类与噪声范围内，正确 F 相比 L 并未显著改变材料决策；若现有点估计已逼近可实现的该任务风险；若多步先验耦合也没有在匹配成本下改善结构；或者在线新增成本明显大于有意义的收益，则不再将在线 feedback 作为主路线。

相反，只要发现新的材料候选在 F 下可分、在 L 下混淆，且当前估计确实漏掉了该差别，就仍有理由研究训练时利用、局部自适应或多步联合推断。停止的是**未经证据支撑的默认在线计算**，不是 Maxwell 物理本身。

---

# Part V — Minimal Decisive Experiments

## 20. 执行总原则与最小产物

三个实验是按结果推进的树，不是同时展开的 benchmark：

\[
 E1\ \longrightarrow\begin{cases}
 E2,&\text{存在可学习的材料判别/决策余量};\\
 E3,&\text{当前观测下结构歧义占主导};\\
 \text{停止该机制},&\text{两类证据均不支持}.
 \end{cases}
\]

不新建大量重复文档。建议只有一个 `a28_config.json`、三个执行入口、结构化 CSV/NPZ 结果和一个持续更新的研究报告。先运行现有代数检查，再进入后端实验。

以下时间和次数是**建议预算上限，不是测得运行时间**。先以冻结样本做成本 microbenchmark；若超预算，在查看科学结果前按 family/parent 分层统一缩减。不要进行到一半才删除失败样本或只保留成功方法。

## E1 — 无训练的“材料假设判别力与均值一致性”实验

**这是下一项最值得做的实验。** 它直接决定后续是学先验、学后验、修更新路径，还是增加测量。

### E1.1 数据与角色隔离

使用已经暴露的 16 个 development parents，两个已知背景，选一个较高幅度，主要分析带噪条件，形成 32 个材料/背景条件。clean fields 可从已有资产复用；噪声重复不增加独立 parent 数。四家族分别报告，复杂三家族作为主要判断组。

必须将两条路径分开：

**真值中心的结构反事实，仅用于可识别性诊断。** 为每个材料生成三种合法编辑，例如空腔填充/打开、边界位移、细小部件增删并允许有限对比补偿。可采用 8–16 个材料/形状坐标作局部优化，但始终保持先验规定的物理约束。任何使用真值几何生成的候选都不得进入“在线重建可实现性”结论。

**CNN 中心的 data-only 候选与多步更新，用于算法余量。** 从已训练 N3 输出构造统一的多尺度/形状扰动，不读取真实 family、真值边界或真实误差方向。F 与 L 使用完全相同的参数化、先验、起点和预算，最多六次迭代；真值只能用于最终评分。与普通 TV 更新作同起点参考。

结构 editing 规则与超参数先在 TRAIN 上定好。不能按 development 真值效果选择哪种编辑最有利。用某个 noisy y 搜索到候选后，要锁定该对候选，再用独立噪声重复检查判别；不能拿用于搜索的同一噪声证明其统计可分性。

### E1.2 必须测什么

对每对合法候选，记录 full F 白化距离、L 白化距离、finite-feedback 割线余项、材料 ΔNRMSE、空腔/支撑/边缘差异，以及固定候选二元模型下的 \(\Phi(-d/2)\)。不只记录 F−L 范数。

在一个明确声明权重的有限候选分布上，计算 \(\mu_\chi,\mu_c,\mu_e,\kappa\)。数值核对式 (8)–(11)，比较：材料均值、mean-current ratio、物理修正后的点估计，以及相应的**后验期望任务损失**。这个分布是诊断模型，不是未经校准的真实后验。

同时测试 N3 实际 encoder：背景 BP 映射的数值秩与条件数、所见材料对在完整输入中的区分程度。只有发现实际非单射/严重数值压缩，才提出 raw-data bypass；否则不要把更换 encoder 当作默认任务。

多步路径报告每步数据、材料、边缘、支撑和接受率，确认 A27 的单步小收益是否来自更新路径限制。不得将这条离线 headroom 轨迹的末端效果当作已经实现的 40 ms 部署性能。

### E1.3 成本与判定

新增 full-wave 材料状态评估建议硬上限 768 次；已有 clean labels 不重复求解。多源 RHS、adjoint/JVP、候选 line search、均值/ratio 状态审核分别计数。没有新训练。具体可执行 cohort 在成本 microbenchmark 后冻结，采用平衡的 prefix，而非结果导向筛选。

建议预先登记以下判断，阈值属于项目决策而非普遍常数：

| E1 观察 | 科研含义 |
|---|---|
| 合法复杂结构差异大，F 下 d≥3，而当前 CNN 经常选错；同先验 data-only F 路径或有限后验决策出现约 ≥10% 材料余量 | 测量确有可用但尚未被当前算法利用的信息；进入 E2 有理由 |
| 一批重要结构对 F 下 d≤1，且先验权重都不可忽略 | 等先验固定二元问题错误率至少约 30.9%；不能只靠更精确梯度解决，优先后验表达或 E3 |
| F 相比 L 的材料决策收益仅约 1% 或更小，且不存在有意义结构增量 | 不再以 feedback-specific 学习作为主张，即使场差异大 |
| κ 的材料有效投影明显解释 mean-current ratio 偏差 | 保留高风险协方差表示；若影响可忽略，则不启动 κ 网络 |
| F 六步明显优于 F 一步，而且结构先验是必要条件 | A27 不能作为多步反例；后续比较摊销 teacher 与廉价有限迭代，而非简单停止非线性 |

“未找到近等测量对”不能写成唯一性；有限候选误差不能写成完整材料类的风险上界。E1 可以给出歧义下界见证和算法余量证据，不能自动给出全局最优恢复证书。

## E2 — 三臂材料决策蒸馏，检验 full physics 是否有独立训练价值

仅当 E1 显示有合适的候选覆盖、非退化 posterior 和可学习余量时运行。

### E2.1 三个模型，禁止再展开十一臂

| 模型 | 输入和架构 | 训练目标 |
|---|---|---|
| M0：强普通监督 | 同一个 BP/数据 encoder，同一个多尺度学生 | 单真值材料损失 |
| ML：背景线性 teacher | 完全相同 | 式 (16)，后验权重来自 L |
| MF：完整 Maxwell teacher | 完全相同 | 式 (16)，后验权重来自 F |

M0 必须看到与 ML/MF 完全相同的新增材料标签、噪声增强、parent 采样和训练预算。否则无法区分 feedback 机制与额外数据/先验收益。旧 A26 N3 是固定参考，不替代新的有竞争力 M0。

建议最小多尺度架构：`12→24→24` 两个 3×3 卷积；平均池化到半分辨率；`24→48→48`；双线性上采样与 24-channel skip 拼接；`72→24→2`。GELU，无 BatchNorm，输出材料残差后执行共同被动性/box。它只用作强简单基线，不算新贡献。若 E1 证明需要数据 bypass，则三臂一起增加，而不是只给 MF。

### E2.2 训练材料、候选 bank 与噪声

复用 64 train parents、256 个 unique clean states 以及逐源电流。可新增最多 128 个 clean states，例如选 32 个平衡 parent，各增加一个结构编辑，再覆盖两背景、两幅度。它们仍属于原 parent，不算 128 个新的独立形状。

teacher 候选池可条件化于**已知背景和采集设置**，不能条件化于真实 parent、真实 family 或真值误差方向。所有候选对同一个观测都有资格参与。parent 总先验质量应平衡；有更多 augmentation 的 parent 不能仅因候选多而获得更大概率。M0 的 parent/amplitude/edit 采样概率必须与 teacher 的 w_k 完全一致，否则式 (17) 的同目标比较不成立。

为避免 truth-normalized SNR 隐含标签信息，主因果实验建议从已有 clean fields 重新廉价加噪：每背景用 TRAIN 统计确定固定、已知 σ，三臂共同使用；例如以 TRAIN 场 RMS 中位数定义标称 20 dB。已有逐真值归一化 20 dB 协议保留作单独评价。若继续使用 heteroscedastic 噪声，则 teacher 必须包含相应候选协方差与 log-determinant，不能假装所有材料有同一个 Gaussian likelihood。

可为每个 clean state 生成 16 份噪声观测；它们是增强，不增加独立 parent 数。teacher posterior 应离线缓存。先检查熵、有效候选数与覆盖；若近乎全部 one-hot，停止蒸馏试验而不是盲目训练。

### E2.3 损失与训练协议

保留同一个训练 RMS 归一化材料 MSE 加 0.05×(1−SSIM)。M0 对采样真值计算；ML/MF 对候选**逐个计算后取后验期望**。平方部分可以解析合并；SSIM 部分不能仅对平均材料计算。可使用保留至少 \(1-10^{-4}\) 后验质量的候选截断，否则计算完整 bank，并记录截断质量。

每臂计划 2000 个真实优化更新，batch=4，family/parent 平衡；AdamW，初始学习率 0.001，weight decay 1e−5，gradient clip 1；三臂相同初始化和样本顺序。每 100 步仅用已暴露 development 选择 checkpoint。先跑 20 次丢弃式成本 microbenchmark；若三臂总预算超过约 2 GPU 小时的建议上限，在查看质量前统一缩减到 1000 或 500 步。不能只缩减某一臂。

第一轮只有一个配对 seed。只有 MF 同时胜过 M0 与 ML 才复现关键比较的第二个 seed。这个 pilot 只解决是否继续；最终论文还需要训练随机性的更充分证据。

### E2.4 物理成本必须分账

| 阶段 | 计入什么 |
|---|---|
| 一次性背景准备 | 原有 G/receiver/background、BP 算子和归一化缓存；按背景/几何变化摊销 |
| 训练前离线 | 最多 128 个新增状态；已有 F 标签复用；teacher likelihood/posterior；新 noisy observations 的共同特征缓存 |
| 学生训练 | 三臂 optimizer；期望 SSIM 等额外算术；无 Maxwell backward |
| 每张图在线 | 原始数据传输、共同 BP/背景线性算子、学生、材料投影、必要同步；MF 相比 M0 不新增 full F、G 应用或 LU |
| 离线科学诊断 | full solver 检查材料图、残差、OOD；不计入“用户每张图必做”，也不能冒充在线认证 |

若实际应用必须逐图 full-wave 验收，则应另报含该验收的端到端延迟。无在线物理 teacher 并不自动给出逐图物理真实性保证。

### E2.5 独立验证、指标与失败标准

所有规则、checkpoint 选择方式和代码 hash 锁定后，才打开真正未暴露的 locked parents。若原 locked 集已在其他阶段暴露，生成新 32 ID parents 加 16 OOD parents；不要把 A25/A26/A27 的 16 个 development parents 再命名为新验证集。OOD 至少包含新组合结构，并将独立变化的损耗/实部作为不同的泛化条件，不与原固定复相位标签混为同分布。

主指标：equal-parent Δχ NRMSE，并同时报告与理论对应的固定尺度材料 MSE。结构指标为 edge F1、support IoU，另报每家族、实/虚部、passivity 和完整数据残差。若输出概率，报告结构事件的 calibration/Brier 或 log score；没有概率输出则不附造 uncertainty 指标。

建议继续门槛：MF 相对 M0 的复杂材料 ΔNRMSE 改善至少约 5%，parent bootstrap 区间下限 >0；相对 ML 至少约 2% 且区间下限 >0；结构指标不能系统性回退，最好至少一个有明确改善；相同架构下 raw-to-image 的 MF/M0 latency 不超过约 1.05。门槛在开发后、测试前冻结，不因结果改变。

**最重要的失败解释：** MF 只赢 ML 不赢 M0，说明 F 比错误的 L teacher 更合适，但没有实际算法优势；M0、ML、MF 都赢旧 N3而彼此接近，说明主要是更好的结构先验；只赢 residual 不赢材料/结构，重复 A27 结论；需要真值 parent 才能算 posterior，则实验无效。

## E3 — 条件启动的一频增量实验

仅当 E1 显示重要结构在当前观测下确实混淆时运行。初始阶段无需新网络。

比较相同总能量和采集时间的两种配置：原频率 \(f_0\) 重复两次；原频率加一个新频率 \(f_1\)。每次分配一半能量，幅度相应乘 \(\sqrt{1/2}\)。\(f_1\) 可先固定为 \(1.2f_0\)，或只在 TRAIN 中从 \(0.8f_0,1.2f_0\) 二选一后冻结，不能在测试结构上调频。

保持同一候选结构先验，按物理 ε/σ 而不是任意固定复 χ 生成多频响应。先用有限候选 posterior 或同一个低维结构优化求解，新增 full-wave 状态建议上限 128。记录重要结构对的判别距离、错误率预测与独立噪声验证、材料期望风险和结构指标。

若一个小测量改动显著优于所有同观测算法改动，瓶颈主要在 acquisition。若收益只来自更多能量或多一次采样，应由重复 \(f_0\) 的控制揭示。若频率变化伴随模型偏差，使材料指标变差，也应保留；局部 Fisher 增量非负并不覆盖错误模型。# Part VI — Final Research Decision

## 21. 十个必须明确回答的问题

### 1. 之前究竟在哪一步损失了真正的材料成像信息？

**目前不能定位为单一 encoder 损失。** A26 的 BP 输入可能保留原数据；没有证据支持“source aggregation 已删除了全部关键材料信息”。更有证据的瓶颈是：有限观测对某些结构弱约束；当前先验/有限训练没有正确解释这些结构；后处理仍沿数据敏感方向优化；不同阶段优化的统计目标不完全一致。

若 E1 发现原测量能清楚区分 CNN 混淆的结构，损失主要发生在表示/先验/学习/更新阶段。若测量根本无法区分，则不是 decoder 把信息弄丢了，而是原测量没有提供该信息。

### 2. 能否不明显增加推理成本而利用 feedback 改善复杂材料？

**可能，最合理的位置首先是训练 teacher，而不是在线层。** 用完整 Maxwell 给合法结构分配正确相对 likelihood，再训练相同成本的材料学生。能否改善尚未建立，必须由 MF 同时胜过 M0、ML 证明。只赢线性 teacher 不够，只训练更慢也不算在线优势。

对于大量固定采集几何的重复成像，离线代价可以摊销；对每张图都有新背景/几何的任务，这个摊销假设可能失效，应另行计费与验证。

### 3. current/field supervision 与 A12/A17 能形成新 learning principle 吗？

**能够形成有约束力的候选原则，但普通 current loss 不够。** 式 (4)–(7) 刻画了与指定材料 pullback 或更新有关的电流/反馈误差；式 (8)–(11) 进一步要求监督目标与材料决策统计量一致。真正可能新的部分，是将 task-relevant feedback 与 posterior constitutive covariance 联系起来，而不是把 current-MSE 改成一个经验权重。

内部标签属于训练时特权信息，不会在测试时增加 \(I(\chi;Y)\)。它能改变有限数据学习效率、归纳偏置和误差传播；不能神奇地唯一确定原先不可区分的材料。

### 4. 是否应显式分离 physics-supported 与 prior-dependent coordinates？

**分析上要分，架构上不宜预设硬分。** 合适定义需要 noise、prior、task、state 和合法材料类。局部可使用 prior-whitened LIS，非局部使用条件材料假设与有限割线。分离一般是条件性的、软耦合的；独立相加的两张材料图没有普遍统计意义。

### 5. VAE、flow、diffusion 是否必要？

**当前没有必要性证据。** 最先比较确定性多尺度先验与有限候选条件决策；若真实后验多峰、其结构概率本身重要且有限候选不足，才升级到小型 mixture/CVAE。训练数据仅 64 parents 时直接启动 diffusion，既不能保证真实 posterior，也容易把先验复现当测量恢复。

后验抽样的价值是表示多解性与任务风险，不是保证单样本 NRMSE 更低。

### 6. OM、TSOM、SOM-Net 应保留与丢弃什么？

保留可靠数据锚定、多照明共享材料、已知背景复用、适当限制内部自由度、逐步增加结构复杂度以及物理/材料联合验证。丢弃必须有显式 OM、固定 TASK_R32、固定 weak-SVD、强制 current latent、默认一层精确修正等架构约束。

TSOM 的 second fold 是内部算子上的结构正则思想，不自动等价于 A23 的 P seed。即使不保留该实现，也应明确由何种材料/后验约束替代其作用，而非声称它从此“冗余”。

### 7. 测量信息有限时，还有多少可实现的图像改善？

**现有结果不支持一个可信的百分比上界。** 96.2% weak error 不是 96.2% 不可恢复；A25 的 10% 也不是 A27 的可实现余量。式 (3) 给出任务稳定性与两点下界；有限候选 Bayes risk 给出特定先验下的可计算诊断。只有把材料类和噪声模型说清楚，才有资格谈上限。

更好的先验可能大幅改善平均表现，但也可能在 OOD 中失败。即使整体 NRMSE 改善有限，某个受观测支持的空腔/部件事件概率仍可能有很大实用改善，应单独定义任务。

### 8. 哪种方法最可能在大规模 Maxwell 成像中兼顾精度、成本和泛化？

我的优先判断是：**以结构先验为主、以物理定义的条件信息为训练约束、推理端轻量摊销、必要时才调用可靠全波审核的混合方案。** 它不需要在每个像素/每张图上预测全部内部电流，也不把低秩在线反馈当作前提。

但大规模泛化不能只靠一个固定 24×24 模型证明。几何变化、3D 向量耦合、独立实虚材料、色散、标定误差和新的空间尺度，都可能需要 geometry-conditioned 算子或局部更新。当前证据只能支持这是一条值得检验的路线，不是已经解决 3D Maxwell 的方法。

### 9. 下一项最值得做的实验是什么？

**E1：合法结构候选的 full-wave 判别力、均值闭合偏差和同先验多步余量。** 它不需要新训练，却能区分四种根本不同的原因：观测没信息、先验没学好、更新路径不合适、点估计物理约束与材料目标冲突。

在这些原因没分清前，新增一个 CNN、一个 current head 或一层 feedback，通常只能产生又一轮难以归因的正/负结果。

### 10. 距离真正有吸引力的强 TAP 算法论文还差什么？

还差一条完整而可反驳的证据链：真实 Maxwell 下的机制量能预测成像失败；针对该机制的新推断/学习方法胜过强普通监督、合适物理损失和 SOM-Net/CSI 类近邻；增量确实来自完整材料条件物理而非更多训练数据；独立 ID/OOD 与不同离散化避免 inverse crime；更大网格或 3D/真实测量展示有意义的规模与物理泛化；端到端时间包含实际必需计算。

E1/E2 的三臂设计是决策实验，不是最终论文 benchmark。若通过，再补最近邻的严格匹配实现和更大规模实验；不能在第一天把全部基线、生成网络和传感设计一起启动。

## 22. 证据等级汇总

| 等级 | 当前可以说什么 |
|---|---|
| 已核对的实证 | A25 窄 regime 的正面收益；A27 复杂材料小收益、明显数据残差下降；局部梯度/材料误差错位；在线额外成本；内部标签可用 |
| 强烈支持的机制 | 继续提高传播精度并非当前复杂结构重建的充分条件；剩余瓶颈与先验/材料相关可观测方向有关 |
| 尚未检验 | A27 的小收益是否主要由后验均值闭合导致；多步结构更新能释放多少余量；MF 蒸馏是否独立获益；κ 是否可廉价学习；新增频率是否最划算 |
| 已被严格反例否定的一般推断 | 更好数据一致性必然更好材料 MSE；最优 current-MSE 经解析 ratio 必然得到最优材料 MSE；局部零/弱导数必然代表先验类上不可识别；β<1 必然统计不可检测 |

## 23. 最终建议

**现在 GO：E1。条件 GO：E2。发现真实结构歧义后再 GO：E3。现在 NO-GO：下一轮在线 OM/full-feedback 网络变体与大型生成模型训练。**

A28 最值得押注的不是某个 block，而是如下可被证明、计算和检验的分工：

\[
\boxed{\text{Maxwell 约束候选世界；先验组织可能结构；决策损失决定最终应该输出什么。}}
\]

如果完整 feedback 无法改善合法材料之间的条件判别，就不值得继续为它付出在线成本。如果它确实提供了差异，优先把差异摊销到训练中。如果材料在当前观测下真实多解，就应输出相应风险或改变最小观测，而不是把更低 residual 当成更可靠图像。

---

# 附录 A — 本轮轻量验证与复现

运行：

```bash
python A28_theory_checks.py
```

只需要 NumPy。脚本旁生成 `A28_theory_checks.json`。随机种子固定为 281010；最大一般复矩阵仅 7×7，另有实 GN 模型与两单元有限 VIE 反例。

14 项代数相等式检查全部通过，最大相对差异约 1.263×10⁻¹³。检查覆盖双向有限割线、完整反馈余项、current-to-material pullback、跨源正交分解、GN 更新缺陷、二次风险差、Gaussian likelihood 投影损失和均值闭合反例。另记录 prior correlation 与 weak-update 反例。

这些检查验证公式实现与反例，不替代符号证明，也不证明新方案在 A27 数据上的有效性。

实现时必须注意：正文所有随机优化的梯度/Hessian 使用实虚展开后的实坐标；复本构协方差 κ 不带共轭；cross-source least squares 带共轭；质量归一化与 cell volume 不可混用；σ=0 单列，不能直接代入 Gaussian whitening；对协方差公式，T 为线性任务算子，对拓扑等非线性任务应使用单独损失/事件概率。

# 附录 B — 固定资产与文献

## B.1 固定研究资产

**[A25]** `migodam/a25-adaptive-om-feedback`，commit `59d8cc57c9da99ccb7978febc308193c43d6f3e5`。

- [A25_FINAL_RESEARCH_DECISION.md](https://github.com/migodam/a25-adaptive-om-feedback/blob/59d8cc57c9da99ccb7978febc308193c43d6f3e5/A25_FINAL_RESEARCH_DECISION.md)
- [FEEDBACK_HEADROOM.md](https://github.com/migodam/a25-adaptive-om-feedback/blob/59d8cc57c9da99ccb7978febc308193c43d6f3e5/FEEDBACK_HEADROOM.md)

**[A26]** `migodam/a26-om-feedback-prior-completion`，commit `2603f3bd16c48e48587fedc4363975779def17f5`。

- [研究报告](https://github.com/migodam/a26-om-feedback-prior-completion/blob/2603f3bd16c48e48587fedc4363975779def17f5/A26_RESEARCH_REPORT_ZH.md)
- **[A26-E]** [encoder.py](https://github.com/migodam/a26-om-feedback-prior-completion/blob/2603f3bd16c48e48587fedc4363975779def17f5/src/a26/encoder.py)
- [decoder.py](https://github.com/migodam/a26-om-feedback-prior-completion/blob/2603f3bd16c48e48587fedc4363975779def17f5/src/a26/decoder.py)
- [training.py](https://github.com/migodam/a26-om-feedback-prior-completion/blob/2603f3bd16c48e48587fedc4363975779def17f5/src/a26/training.py)

**[A27]** `migodam/a27-feedback-failure-attribution`，commit `325cc8a5f99d686ce1b1570296f94b0df1e90141`。

- [A27_RECOMMENDATION_FOR_GPT_PRO.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_RECOMMENDATION_FOR_GPT_PRO.md)
- **[A27-M]** [A27_MECHANISM_DIAGNOSIS.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_MECHANISM_DIAGNOSIS.md)
- **[A27-O]** [A27_OBSERVABILITY_AND_PRIOR.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_OBSERVABILITY_AND_PRIOR.md)
- **[A27-P]** [A27_POST_DECODER_CORRECTION.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_POST_DECODER_CORRECTION.md)
- **[A27-L]** [A27_LATENCY_ATTRIBUTION.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_LATENCY_ATTRIBUTION.md)
- **[A27-S]** [A27_INTERMEDIATE_SUPERVISION_SPEC.md](https://github.com/migodam/a27-feedback-failure-attribution/blob/325cc8a5f99d686ce1b1570296f94b0df1e90141/A27_INTERMEDIATE_SUPERVISION_SPEC.md)

## B.2 核心文献与阅读定位

**[L1]** Xudong Chen. *Computational Methods for Electromagnetic Inverse Scattering*. Wiley–IEEE Press, 2018. ISBN 9781119311980。直接核对第 6.4.1–6.4.2 节及 FFT-TSOM 讨论，纸书约 pp.149–164；来源为用户 Library 中专著。重点是 state/data equation、deterministic/ambiguous current 和内部 second-fold 的实现边界。

**[L2]** P. M. van den Berg and R. E. Kleinman. *A contrast source inversion method*. Inverse Problems, 13, 1997. [DOI: 10.1088/0266-5611/13/6/013](https://doi.org/10.1088/0266-5611/13/6/013)。连同作者 CSI/MR-CSI 后续讨论，用于核对共同材料约束、交替优化与解析材料更新的既有基础。

**[L3]** *SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave Inverse Scattering Problems*. 2022 作者稿。[arXiv:2209.03567](https://arxiv.org/abs/2209.03567)。重点：展开结构、deterministic current、学习 current、解析材料更新和联合监督；不能只按摘要的“physics-informed”概括。

**[L4]** Zicheng Liu, Mayank Roy, Dilip K. Prasad, Krishna Agarwal. *Physics-Guided Loss Functions Improve Deep Learning Performance in Inverse Scattering*. IEEE Transactions on Computational Imaging, 8, 236–245, 2022；2021 preprint。[DOI:10.1109/TCI.2022.3158865](https://doi.org/10.1109/TCI.2022.3158865)，[arXiv:2111.09109](https://arxiv.org/abs/2111.09109)。优先看 near-field/current losses，尤其以真值 total field 乘材料预测的损失。

**[L5]** Anran Si, Miao Wang, Fuping Fang, Da-Hai Dai. *Two-Step Contrast Source Learning Method for Electromagnetic Inverse Scattering Problems*. Sensors, 24(18), 5997, 2024。[出版全文](https://www.mdpi.com/1424-8220/24/18/5997)，DOI:10.3390/s24185997。直接近邻：current-first 再 material refinement。

**[L6]** Tiangang Cui, James Martin, Youssef M. Marzouk, Antti Solonen, Alessio Spantini. *Likelihood-informed dimension reduction for nonlinear inverse problems*. Inverse Problems, 30, 114015, 2014。[arXiv:1403.4680](https://arxiv.org/abs/1403.4680)。用于避免把 unwhitened background SVD 当 posterior subspace。

**[L7]** Alessio Spantini, Tiangang Cui, Karen Willcox, Luis Tenorio, Youssef Marzouk. *Goal-Oriented Optimal Approximations of Bayesian Linear Inverse Problems*. SIAM Journal on Scientific Computing, 39, S167–S196, 2017。[DOI:10.1137/16M1082123](https://doi.org/10.1137/16M1082123)。任务后验误差而非任意状态误差的近邻。

**[L8]** Khorashadizadeh et al. *Deep Injective Prior for Inverse Scattering*. IEEE Transactions on Antennas and Propagation, 2023。[arXiv:2301.03092](https://arxiv.org/abs/2301.03092)，[DOI:10.1109/TAP.2023.3312818](https://doi.org/10.1109/TAP.2023.3312818)。低维材料先验与概率推断的重要基线方向。

**[L9]** Borong Zhang, Martin Guerra, Qin Li, Leonardo Zepeda-Núñez. *Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models*. 2024 preprint，2025 journal version。[arXiv:2408.02866](https://arxiv.org/abs/2408.02866)，[DOI:10.1016/j.cma.2025.118036](https://doi.org/10.1016/j.cma.2025.118036)。物理编码、条件 posterior 与低在线求解成本的最直接近邻之一；注意 acoustic 模型与本项目 TM 设置的区别。

**[L10]** Yi Zhang, Peng Wang, Difan Zou. *Physics-Informed Distillation of Diffusion Models for PDE-Constrained Generation*. 2025 preprint；2026 修订版。[arXiv:2505.22391v2 全文](https://arxiv.org/html/2505.22391v2)。明确讨论 Jensen gap，以及在最终生成样本而非 denoising posterior mean 上施加物理约束。泛化的均值—物理不交换不是 A28 的首次发现。

**[L11]** Johann Brehmer, Gilles Louppe, Juan Pavez, Kyle Cranmer. *Mining gold from implicit models to improve likelihood-free inference*. PNAS, 2020。[arXiv:1805.12244](https://arxiv.org/abs/1805.12244)，[DOI:10.1073/pnas.1915980117](https://doi.org/10.1073/pnas.1915980117)。模拟器内部信息用于 likelihood/score 学习的原始方法论近邻。

**[L12]** Meet P. Vadera, Brian Jalaian, Benjamin M. Marlin. *Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks*. UAI 2020, PMLR 124, 719–728。[出版全文入口](https://proceedings.mlr.press/v124/vadera20a.html)。说明 posterior expectation distillation 本身已有研究。

**[L13]** Jonas Adler, Ozan Öktem. *Learned Primal-Dual Reconstruction*. IEEE Transactions on Medical Imaging, 37(6), 1322–1332, 2018。[arXiv:1707.06474](https://arxiv.org/abs/1707.06474)，[DOI:10.1109/TMI.2018.2799231](https://doi.org/10.1109/TMI.2018.2799231)。用于比较 learned iterative/primal-dual 机制，而非直接移植其成像任务结论。

**[L14]** Johannes Schwab, Stephan Antholzer, Markus Haltmeier. *Big in Japan: Regularizing Networks for Solving Inverse Problems*. Journal of Mathematical Imaging and Vision, 62, 445–455, 2020，2019 online。[DOI:10.1007/s10851-019-00911-1](https://doi.org/10.1007/s10851-019-00911-1)。含 null-space networks 及对弱但非零方向的推广。

**[L15]** Justin Alsing, Benjamin Wandelt. *Generalized massive optimal data compression*. MNRAS Letters, 2018。[arXiv:1712.00012](https://arxiv.org/abs/1712.00012)，[DOI:10.1093/mnrasl/sly029](https://doi.org/10.1093/mnrasl/sly029)。局部 score/Fisher-preserving compression 的近邻。

**[L16]** Wenhan Guo, Yuan Gao, Yu Sun. *ScoreField: Neural Inverse Scattering with Score-Based Generative Priors*. 2026 preprint。[arXiv:2608.02937](https://arxiv.org/abs/2608.02937)。current/contrast 表示、全波约束与 score prior 联合优化；不应自动解释为已校准的 posterior sampler。

**[L17]** Leung Tsang and Jin Au Kong. *Scattering of Electromagnetic Waves: Advanced Topics*. Wiley, 2001。[出版方书目](https://www.wiley-vch.de/en/areas-interest/engineering/scattering-of-electromagnetic-waves-978-0-471-38801-2)。仅作为随机介质平均场与统计闭合历史背景，本轮没有全文审读相应整章，不据此作具体公式优先权断言。

**[L18]** Yu Zhong and Xudong Chen. *Twofold subspace-based optimization method for solving inverse scattering problems*. Inverse Problems, 25, 085003, 2009。原始条目及机制由作者专著和[作者出版列表](https://www.ece.nus.edu.sg/stfpage/elechenx/Publications.htm)交叉核对；本轮未取得完整原论文全文。

**[L19]** Yu Zhong and Xudong Chen. *An FFT Twofold Subspace-Based Optimization Method for Solving Electromagnetic Inverse Scattering Problems*. IEEE Transactions on Antennas and Propagation, 59(3), 914–927, 2011。[DOI:10.1109/TAP.2010.2103027](https://doi.org/10.1109/TAP.2010.2103027)。实现机制以作者专著的 FFT-TSOM 讨论交叉核对。

**[L20]** W. C. Chew and Y. M. Wang. *Reconstruction of Two-Dimensional Permittivity Distribution Using the Distorted Born Iterative Method*. IEEE Transactions on Medical Imaging, 9(2), 218–225, 1990。[作者机构记录](https://experts.illinois.edu/en/publications/reconstruction-of-two-dimensional-permittivity-distribution-using/)，[DOI:10.1109/42.56334](https://doi.org/10.1109/42.56334)。DBIM 的原始基线与物理更新背景。