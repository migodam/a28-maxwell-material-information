# A28 E1 implementation / pre-run plan

来源是冻结 A27/A26/A25；新增独立 `src.a28`，理论原件不覆盖。研究包复制保留在 `theory/a28/`。先审计、代数/后端验证和成本 microbenchmark，再冻结 TRAIN editing rules、family-balanced cohort、随机种子与预算；此时不查看新的科学评分。

四项诊断共享实际 clean current/field 和背景，不重新生成已有 clean 标签。A：真值中心四类固定结构反事实，身份与用途明确为 OFFLINE。B：N3 起点六个共同预算的 gradient/TV outer attempts，只有测量、背景及冻结先验进入更新；普通 full-state adjoint+TV 与 F-TV 数学相同，保留别名。C：有限候选分布，分别标记 truth-centered oracle 与 CNN-centered data-only，不把有限 bank 称为校准后验。D：实际 N3 未截断 BP 的秩、条件数以及候选割线/噪声协方差后的信息。

预计全部16 parents×2 backgrounds×高幅度20dB为32条件。审计/microbenchmark 后才能冻结。主 arm 各至多六个付费候选审核；拒绝步在下一 outer attempt 使用半步 damping，而非增加隐形 line search。实际 accepted steps 与 attempted/query数均报告。相同预算指更新/候选与 full-F 审核额度，F/L 实际 RHS 和 wall-time 并不相等。每种模型的全 wave state、重复、失败都经过持久计数，硬上限768。

计划上界：真值中心四反事实128；CNN中心三反事实96；N3状态32；两臂六审核384；CNN后验均值/ratio审核64，共704。另加每条件最多一次有限均值的测量驱动修正32，primary最多736；审计/microbenchmark预留32（计划28次），总计最多764，硬上限仍768，不借用或增加上限。均值等同端点时仍按实际尝试计数。任何无法认证/缺失保留，不筛成功样本。

为使“经常/一批”可审计，执行约定在结果前固定为至少4个复杂parents，歧义还需覆盖至少两复杂家族；κ的候选用途需至少8条件非退化ESS≥1.5且ratio偏差≥真值材料增量尺度的1%。这是项目screening convention，不是给理论增加一般性证明。E2的10%余量必须来自合法CNN候选或同先验路径，并且parent bootstrap支持正方向；oracle中心决策不参与此门槛。

主判断先在parent内平均两背景，再进行16/12 parent配对bootstrap；噪声重复不增加parent数。原development噪声用于搜索，固定候选用独立seed的新噪声做判别检查，无新材料标签。清晰/模糊的二元阈值是d≥3/d≤1，重要结构尺度固定ΔNRMSE≥.10。E2还需要约10%合法材料余量、候选覆盖与非退化，不能用oracle gain代替；E3需要重要结构歧义见证。κ代数非零与真实材料风险作用分别报告。

本实验不开新网络/训练/E2/E3/locked。完成后保留一个 `E1_DECISION.md` 作为主报告及结构化数据、图和实际网页入口。解析恒等式、数值验证、有限候选机制和在线可实现性分别解释。
