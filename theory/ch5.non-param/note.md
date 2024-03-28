一个样本点$\boldsymbol{x}$落在区域$\mathcal{R}$中的概率为
$$
P = \int_{\mathcal{R}} p(\boldsymbol{x}) d\boldsymbol{x}
$$
在独立抽取的$n$个样本中，落在区域$\mathcal{R}$中的数量
$$
k \sim b(n,\ P)
$$
$k$的数学期望
$$
\mathcal{E}(k) = nP
$$
当区域$\mathcal{R}$取得足够小时，区域$\mathcal{R}$中一点$\boldsymbol{x}$处的概率密度满足
$$
\mathcal{E}(k) = nP = n \int_{\mathcal{R}} p(\tau) d\tau \approx n p(\boldsymbol{x}) V
$$
由此得到$\boldsymbol{x}$处的概率密度的估计
$$
\hat{p}(\boldsymbol{x}) \approx \frac{P}{V} \approx \frac{k/n}{V}
$$
为了保证估计方法的收敛性，分别需要

* 保证区域$\mathcal{R}$收缩得足够小
$$
\lim_{n \to \infty} V_{n} =  0
$$
* 在概率密度不为0的前提下保证频数足够多
$$
\lim_{n \to \infty} k_{n} = \infty
$$
* 在区域收缩得足够小时保证频率也足够小
$$
\lim_{n \to \infty} k_{n} / n = 0
$$

分别可以通过固定区域大小序列$V_{n}$或频数序列$k_{n}$的方式来获取满足以上条件的估计序列

# Parzen 窗方法

固定一个合法的超立方体区域序列
$$
V_{n} = h_{n}^{d}
$$
定义窗函数（在标准超立方体内为1，否则为0）
$$
\varphi(\boldsymbol{u}) = \mathbb{I}\left( |u_{j}| \le \frac{1}{2} \right);\ j = 1,\ 2,\ \cdots,\ d
$$
落在超立方体区域中的样本频数
$$
k_{n} = \sum_{i = 1}^{n} \varphi \left( \frac{\boldsymbol{x} - \boldsymbol{x}_{i}}{h_{n}} \right)
$$
估计得到的概率密度
$$
p_{n}(\boldsymbol{x}) = \frac{1}{nV_{n}} \sum_{i = 1}^{n} \varphi \left( \frac{\boldsymbol{x} - \boldsymbol{x}_{i}}{h_{n}} \right)
$$
合法的概率密度要求
$$
\begin{gather*}
p_{n}(\boldsymbol{x}) \ge 0\ \Leftrightarrow\ \varphi(\boldsymbol{u}) \ge 0 \\ \\
\int_{\mathbb{R}^d} p_{n}(\boldsymbol{x}) d\boldsymbol{x} = 1\ \Leftrightarrow\ \int_{\mathbb{R}^d} \varphi(\boldsymbol{u}) d\boldsymbol{u} = 1
\end{gather*}
$$
满足上述要求的窗函数同样可以用于概率密度的估计，例如常用的高斯窗
$$
\varphi(\boldsymbol{u}) = \varphi \left( \frac{\boldsymbol{x} - \boldsymbol{x}_{i}}{h_{n}} \right) = N(0,\ \boldsymbol{\Sigma})
$$
**影响因子**
$$
\delta_{n}(\boldsymbol{x}) = \frac{1}{V_{n}} \varphi \left( \frac{\boldsymbol{x}}{h_{n}} \right)
$$
估计概率密度可以重写为
$$
p_{n}(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^{n} \delta_{n}(\boldsymbol{x} - \boldsymbol{x}_{i})
$$
在窗宽$h_{n}$过大时，影响因子过于平滑，导致估计产生散焦；在窗宽$h_{n}$过小时，影响因子过于陡峭，导致估计充满噪声

## Parzen 窗估计的收敛性

为了证明 Parzen 窗方法能够在样本数量趋于无穷时收敛到真实值，等价于证明
$$
\begin{gather*}
\lim_{n \to \infty} \mathcal{E} p_{n}(\boldsymbol{x}) = p(\boldsymbol{x}) \\ \\
\lim_{n \to \infty} \mathcal{D} p_{n}(\boldsymbol{x}) = 0
\end{gather*}
$$
考虑估计得到的概率密度$p_{n}(\boldsymbol{x})$的在训练集样本$\boldsymbol{x}_{i}$上的期望
$$
\begin{align*}
\mathcal{E} p_{n}(\boldsymbol{x}) &= \frac{1}{nV_{n}} \sum_{i = 1}^{n} \mathcal{E} \left[ \varphi 
\left( \frac{\boldsymbol{x} - \boldsymbol{x}_{i}}{h_{n}} \right) \right] \\ \\
&= \int \frac{1}{V_{n}} \varphi \left( \frac{\boldsymbol{x} - \boldsymbol{v}}{h_{n}} \right) p(\boldsymbol{v}) d\boldsymbol{v} \\ \\
&= \int \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) p(\boldsymbol{v}) d\boldsymbol{v} = (\delta_{n} \ast p)(\boldsymbol{x})
\end{align*}
$$
在样本数量趋于无穷时，估计值的期望需要满足
$$
\begin{gather*}
\lim_{n \to \infty} \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) = \delta(\boldsymbol{x} - \boldsymbol{v}) \\
\Downarrow \\
\begin{align*}
\lim_{n \to \infty} \mathcal{E} p_{n}(\boldsymbol{x}) &= 
\lim_{n \to \infty} \int \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) p(\boldsymbol{v}) d\boldsymbol{v} \\ \\
&= \int \lim_{n \to \infty} \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) p(\boldsymbol{v}) d\boldsymbol{v} \\ \\
&= \int \delta(\boldsymbol{x} - \boldsymbol{v}) p(\boldsymbol{v}) d\boldsymbol{v} = p(\boldsymbol{x})
\end{align*}
\end{gather*}
$$
即需要满足条件
$$
\begin{gather*}
\lim_{n \to \infty} V_{n} = 0 \\ \\
\lim_{||\boldsymbol{u}|| \to \infty} \varphi(\boldsymbol{u}) \prod_{i = 1}^{d} u_{i} = 0
\end{gather*}
$$
在上述条件的基础上，可以证明影响因子的极限满足狄拉克函数$\delta(\boldsymbol{x} - \boldsymbol{v})$的性质
$$
\begin{gather*}
\int_{\mathbb{R}^d} \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) d\boldsymbol{x} = \int_{\mathbb{R}^d} \varphi(\boldsymbol{u}) d\boldsymbol{u} = 1 \\ \\
\begin{align*}
\lim_{n \to \infty} \delta_{n}(\boldsymbol{x} - \boldsymbol{v}) &= \lim_{n \to \infty} \frac{1}{V_{n}} 
\varphi(\frac{\boldsymbol{x} - \boldsymbol{v}}{h_{n}}) \\ \\
&= \lim_{n \to \infty} \frac{1}{h_{n}^{d}} \varphi(\frac{x_{1} - v_{1}}{h_{n}},\ \cdots,\ \frac{x_{d} - v_{d}}{h_{n}}) \\ \\
&\overset{\boldsymbol{x} \ne \boldsymbol{v}}{\longrightarrow} 
\lim_{||\boldsymbol{u}|| \to \infty} \varphi(\boldsymbol{u}) \prod_{i = 1}^{d} u_{i} \bigg/ \prod_{i = 1}^{d} (x_{i} - v_{i}) = 0
\end{align*}
\end{gather*}
$$
再考虑估计概率密度在训练集上的方差
$$
\begin{align*}
\mathcal{D} p_{n}(\boldsymbol{x}) &= \mathcal{E} p_{n}^{2} (\mathcal{x}) - \mathcal{E}^{2} p_{n}(\boldsymbol{x}) \\ \\
&\le \sum_{i = 1}^{n} \mathcal{E} \left[ \frac{1}{n^{2} V_{n}^{2}} \varphi^{2} 
\left( \frac{\boldsymbol{x} - \boldsymbol{x}_{i}}{h_{n}} \right) \right] \\ \\
&= \frac{1}{nV_{n}} \int \frac{1}{V_{n}} \varphi^{2} \left( \frac{\boldsymbol{x} - \boldsymbol{v}}{h_{n}} \right) d\boldsymbol{v} \\ \\
&\le \frac{\max \varphi(\cdot) \mathcal{E} p_{n}(\boldsymbol{x})}{nV_{n}}
\end{align*}
$$
样本数量趋于无穷时，估计值的方差需要满足
$$
\lim_{n \to \infty} \mathcal{D} p_{n}(\boldsymbol{x}) \le
\lim_{n \to \infty} \frac{\max \varphi(\cdot) \mathcal{E} p_{n}(\boldsymbol{x})}{nV_{n}}
= \frac{\max \varphi(\cdot) p(\mathcal{x})}{\lim_{n \to \infty} nV_{n}} = 0
$$
即需要满足条件
$$
\begin{gather*}
\max \varphi(\cdot) < \infty \\ \\
\lim_{n \to \infty} nV_{n} = \infty
\end{gather*}
$$

# Kn 近邻方法

固定一个合法的频数 $k_{n}$ 序列，以 $\boldsymbol{x}$ 为中心让体积不断扩张直至包含到 $k_{n}$ 个样本点，估计概率密度
$$
p_{n}(\boldsymbol{x}) = \frac{k_{n} / n}{V_{n}}
$$
与 Parzen 窗方法不同的是，$k_{n}$近邻可以动态调整估计窗口大小，同时保证窗口内总是包含样本点。但$k_{n}$近邻在一阶微分上并不连续，会产生崎岖的尖峰，同时$k_{n}$近邻不能保证估计概率密度的归一性
$$
\int_{\mathbb{R}^{d}} p_{n}(\boldsymbol{x}) d\boldsymbol{x} \overset{?}{=} 1
$$
利用$k_{n}$近邻还可以直接估计后验概率
$$
\begin{gather*}
p_{n}(\boldsymbol{x},\ \omega_{i}) = \frac{k_{i} / n}{V_{n}} \\ \\
p_{n}(\boldsymbol{x}) = \frac{k_{n} / n}{V_{n}} \\ \\
p_{n}(\omega_{i} \mid \boldsymbol{x}) = \frac{k_{i}}{k_{n}}
\end{gather*}
$$

# 最近邻规则

将未见样本$\boldsymbol{x}$的类别预测为训练集$D_{n}$中与它距离最近的样本点$\tilde{\boldsymbol{x}}$的类别
$$
\tilde{\boldsymbol{x}} = \argmin_{\boldsymbol{x}_{i}} ||\boldsymbol{x}_{i} - \boldsymbol{x}||
$$
在样本数量趋于无穷时，可以认为**最近邻样本点**$\tilde{\boldsymbol{x}}$与未见样本$\boldsymbol{x}$足够近
$$
||\tilde{\boldsymbol{x}} - \boldsymbol{x}|| \approx 0 \longrightarrow 
p(\omega_{i} \mid \tilde{\boldsymbol{x}_{i}}) \approx p(\omega_{i} \mid \boldsymbol{x}_{i})
$$
为了衡量最近邻规则的性能，考虑样本数量趋于无穷时的平均误差率$p(error)$
$$
p(error) = \lim_{n \to \infty} \mathcal{E} p_{n}(error \mid \boldsymbol{x}) =
\lim_{n \to \infty} \int p_{n}(error \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x}
$$
理论最优的贝叶斯分类器的平均误差率$\wp(error)$
$$
\begin{gather*}
\wp(error) = \mathcal{E} \wp(error \mid \boldsymbol{x}) = \int \wp(error \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} \\ \\
\wp(error \mid \boldsymbol{x}) = 1 - \max_{i} p(\omega_{i} \mid \boldsymbol{x})
\end{gather*}
$$
对于最近邻规则，随机训练集$D_{n}$对应着一个未见样本$\boldsymbol{x}$的随机最近邻向量$\tilde{\boldsymbol{x}}$，条件误差率$p_{n}(error \mid \boldsymbol{x})$可以写作
$$
\begin{align*}
p_{n}(error \mid \boldsymbol{x}) &= \mathcal{E} \bigg( p_{n}(error \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}})\ \bigg|\ \boldsymbol{x} \bigg) \\ \\
&= \int p_{n}(error \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}}) p_{n}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) d\tilde{\boldsymbol{x}}
\end{align*}
$$
考虑一个样本$\tilde{\boldsymbol{x}}$落在未见样本$\boldsymbol{x}$为中心的超球体$\mathcal{S}$中的概率
$$
P_{n} = \int_{\mathcal{S}} p_{n}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) d\tilde{\boldsymbol{x}}
$$
令样本数量趋于无穷，同时令超球体的半径趋于零，所有样本均落在超球体外部的概率
$$
\lim_{n \to \infty} (1 - P_{n})^{n} = 0
$$
也就是说，最近邻向量的后验概率分布在样本数量趋于无穷时，满足狄拉克函数$\delta(\tilde{\boldsymbol{x}} - \boldsymbol{x})$的性质
$$
\begin{gather*}
\int p_{n}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) d\tilde{\boldsymbol{x}} = 1 \\ \\
\lim_{n \to \infty} p_{n}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) \overset{\tilde{\boldsymbol{x}} \ne \boldsymbol{x}}{\longrightarrow} 0
\end{gather*}
$$
由于训练集和未见样本相互独立且均服从分布$p(\omega_{i},\ \boldsymbol{x})$和$p(\boldsymbol{x})$，有
$$
p(\omega_{i},\ \omega_{j} \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}}) = 
\frac{p(\omega_{i},\ \boldsymbol{x};\ \omega_{j},\ \tilde{\boldsymbol{x}})}{p(\boldsymbol{x},\ \tilde{\boldsymbol{x}})} =
\frac{p(\omega_{i},\ \boldsymbol{x}) p(\omega_{j},\ \tilde{\boldsymbol{x}})}{p(\boldsymbol{x}) p(\tilde{\boldsymbol{x}})} =
p(\omega_{i} \mid \boldsymbol{x}) p(\omega_{j} \mid \tilde{\boldsymbol{x}})
$$
根据上式，条件误差率$p_{n}(error \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}})$可以写为
$$
p_{n}(error \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}}) = 1 - \sum_{i = 1}^{c} 
p(\omega_{i},\ \omega_{i} \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}})
= 1 - \sum_{i = 1}^{c} p(\omega_{i} \mid \boldsymbol{x}) p(\omega_{i} \mid \tilde{\boldsymbol{x}})
$$
由此可得，样本数量趋于无穷时的条件误差率$p_{n}(error \mid \boldsymbol{x})$
$$
\begin{align*}
\lim_{n \to \infty} p_{n}(error \mid \boldsymbol{x}) &= \lim_{n \to \infty} \int p_{n}(error \mid \boldsymbol{x},\ \tilde{\boldsymbol{x}}) 
p_{n}(\tilde{\boldsymbol{x}} \mid \boldsymbol{x}) d\tilde{\boldsymbol{x}} \\ \\
&= \int \left[ 1 - \sum_{i = 1}^{c} p(\omega_{i} \mid \boldsymbol{x}) p(\omega_{i} \mid \tilde{\boldsymbol{x}}) \right]
\delta(\tilde{\boldsymbol{x}} - \boldsymbol{x}) d\tilde{\boldsymbol{x}} \\ \\
&= 1 - \sum_{i = 1}^{c} p^{2}(\omega_{i} \mid \boldsymbol{x})
\end{align*}
$$
进而可得，样本数量趋于无穷时的平均误差率
$$
\begin{align*}
p(error) &= \lim_{n \to \infty} \int p_{n}(error \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} \\ \\
&= \int \left[ 1 - \sum_{i = 1}^{c} p^{2}(\omega_{i} \mid \boldsymbol{x}) \right] p(\boldsymbol{x}) d\boldsymbol{x}
\end{align*}
$$
为了与贝叶斯误差 $\wp(error)$ 作比较，考虑在贝叶斯误差固定时，寻找最近邻误差的误差界

贝叶斯误差固定等价于最大后验概率 $p(\omega_{m} \mid \boldsymbol{x}) = \max_{i} p(\omega_{i} \mid \boldsymbol{x})$ 固定，首先有
$$
\sum_{i = 1}^{c} p^{2}(\omega_{i} \mid \boldsymbol{x}) = p^{2}(\omega_{m} \mid \boldsymbol{x}) + \sum_{i \ne m} p^{2}(\omega_{i} \mid \boldsymbol{x})
$$
在上述条件的约束下寻找最近邻误差的上界
$$
\begin{gather*}
\max p(error) \Longleftrightarrow \min \sum_{i \ne m} p^{2}(\omega_{i} \mid \boldsymbol{x}) \\ \\
s.t.\quad p(\omega_{i} \mid \boldsymbol{x}) \ge 0 \quad 
\sum_{i \ne m} p(\omega_{i} \mid \boldsymbol{x}) = 1 - p(\omega_{m} \mid \boldsymbol{x}) = \wp(error \mid \boldsymbol{x})
\end{gather*}
$$
根据 **Cauchy-Schwarz 不等式**
$$
\sum_{i \ne m} p^{2}(\omega_{i} \mid \boldsymbol{x}) \ge \frac{1}{c - 1} \left[ \sum_{i \ne m} p(\omega_{i} \mid \boldsymbol{x}) \right]^{2}
= \frac{1}{c - 1} \wp^{2}(error \mid \boldsymbol{x})
$$
上式等号当且仅当
$$
p(\omega_{i} \mid \boldsymbol{x}) = \frac{\wp(error \mid \boldsymbol{x})}{c - 1},\quad i \ne m
$$
进而可得
$$
1 - \sum_{i = 1}^{c} p^{2}(\omega_{i} \mid \boldsymbol{x}) \le 2\wp(error \mid \boldsymbol{x}) - \frac{c}{c - 1} \wp^{2}(error \mid \boldsymbol{x})
$$
同时考虑$\wp^{2}(error \mid \boldsymbol{x})$和$\wp(error)$之间的关系
$$
\begin{align*}
\mathcal{D} \wp(error \mid \boldsymbol{x}) &= \mathcal{E} \wp^{2}(error \mid \boldsymbol{x}) - \wp^{2}(error) \\ \\
&= \int \wp^{2}(error \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} - \wp^{2}(error) \ge 0
\end{align*}
$$
代入到最近邻误差中即可得到
$$
p(error) = \int \left[ 1 - \sum_{i = 1}^{c} p^{2}(\omega_{i} \mid \boldsymbol{x}) \right] p(\boldsymbol{x}) d\boldsymbol{x}
\le 2\wp(error) - \frac{c}{c - 1} \wp^{2}(error)
$$
最终的最近邻误差范围
$$
\wp(error) \le p(error) \le \left( 2 - \frac{c}{c - 1} \wp(error) \right) \wp(error)
$$