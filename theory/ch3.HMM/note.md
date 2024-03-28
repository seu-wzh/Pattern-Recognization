# 一阶马尔可夫模型
在$t$时刻系统可能处在$n$个状态
$$
\omega_{1},\ \omega_{2},\ \cdots,\ \omega_{n}
$$
将系统处在这$n$个状态的概率组合为状态概率向量
$$
\pi_{\omega}^{(t)} = (p(\omega_{1}),\ p(\omega_{2}),\ \cdots,\ p(\omega_{n}))^{(t)}
$$
**转移概率**；系统从状态$\omega_{i}$在下一时刻转移到状态$\omega_{j}$的概率
$$
a_{ij} = p(\omega_{j}^{(t + 1)} \mid \omega_{i}^{(t)})
$$
**转移概率矩阵**
$$
\mathbf{A} =
\begin{pmatrix}
a_{11}  &   a_{12}  &   \cdots  &   a_{1n} \\[4mm]
a_{21}  &   a_{22}  &   \cdots  &   a_{2n} \\[4mm]
\vdots  &   \vdots  &   \ddots  &   \vdots \\[4mm]
a_{n1}  &   a_{n2}  &   \cdots  &   a_{nn}
\end{pmatrix}
$$
系统下一时刻的状态概率向量只和当前时刻的状态概率向量有关
$$
\pi_{\omega}^{(t + 1)} = \pi_{\omega}^{(t)} \mathbf{A}
$$

# 一阶隐马尔可夫模型
相较于可见马尔可夫模型，隐马尔可夫模型的状态不可见

但可以观测通过由系统隐藏状态激发出的可能的$m$个可见状态
$$
v_{1},\ v_{2},\ \cdots,\ v_{m}
$$
将系统被观测到这$m$个可见状态的概率组合为可见状态概率向量
$$ 
\pi_{v}^{(t)} = (p(v_{1}),\ p(v_{2}),\ \cdots,\ p(v_{n}))^{(t)}
$$
**发射概率**；隐状态$\omega_{i}$激发出可见状态$v_{j}$的概率
$$
b_{ij} = p(v_{j}^{(t)} \mid \omega_{i}^{(t)})
$$
**发射概率矩阵**
$$
\mathbf{B} =
\begin{pmatrix}
b_{11}  &   b_{12}  &   \cdots  &   b_{1m} \\[4mm]
b_{21}  &   b_{22}  &   \cdots  &   b_{2m} \\[4mm]
\vdots  &   \vdots  &   \ddots  &   \vdots \\[4mm]
b_{n1}  &   b_{n2}  &   \cdots  &   b_{nm}
\end{pmatrix}
$$
系统当前的可见状态概率向量
$$
\pi_{v}^{(t)} = \pi_{\omega}^{(t)} \mathbf{B}
$$

## 估值问题

已知模型的状态转移概率矩阵 $\mathbf{A}$、发射概率矩阵 $\mathbf{B}$ 以及初始分布 $\pi$，计算模型在初始隐状态 $\omega^{(0)}$ 下产生某一观测序列 $\mathbf{V}^\mathrm{T}$ 的概率。

### 暴力枚举方法

枚举所有可能的隐状态序列$\mathbf{\Omega}_{r}^{\mathrm{T}}$，利用全概率公式求出该观测序列产生的概率
$$
\begin{align*}
p(\mathbf{V}^\mathrm{T}) &= \sum_{r} p(\mathbf{V}^\mathrm{T} \mid \mathbf{\Omega}_{r}^{\mathrm{T}}) p(\mathbf{\Omega}_{r}^{\mathrm{T}}) \\[5mm]
&= \sum_{\omega^{(\mathrm{T})}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}}
\left[ \prod_{t = 1}^{\mathrm{T}} p(\omega^{(t)} \mid \omega^{(t - 1)}) \right] 
\left[ \prod_{t = 1}^{\mathrm{T}} p(v^{(t)} \mid \omega^{(t)}) \right] \\[5mm]
&= \sum_{\omega^{(\mathrm{T})}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}}
\prod_{t = 1}^{\mathrm{T}} p(\omega^{(t)} \mid \omega^{(t - 1)}) p(v^{(t)} \mid \omega^{(t)}) \\[5mm]
&= \sum_{\omega^{(\mathrm{T})}} \left[ \sum_{\omega^{(\mathrm{T} - 1)}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}}
\prod_{t = 1}^{\mathrm{T}} p(\omega^{(t)} \mid \omega^{(t - 1)}) p(v^{(t)} \mid \omega^{(t)}) \right] \\[5mm]
&= \sum_{i = 1}^{c} \alpha_{i}(\mathrm{T})
\end{align*}
$$
其中 $\alpha_{i}(t)$ 代表在产生 $t$ 时刻及之前所有可见状态序列的基础上，系统处在隐状态 $\omega_{i}$ 的概率，同时 $\alpha_{i}(t)$ 满足以下迭代方程：
$$
\begin{align*}
\alpha_{i}(t) &= \sum_{\omega^{(t - 1)}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}} \prod_{\tau = 1}^{t} p(\omega^{(\tau)} \mid \omega^{(\tau - 1)}) p(v^{(\tau)} \mid \omega^{(\tau)})\ \Rightarrow s.t.\ \omega^{(t)} = \omega_{i} \\[5mm]
&= \sum_{j = 1}^{c} p(\omega_{i} \mid \omega_{j}) p(v^{(t)} \mid \omega_{i}) 
\left[ \sum_{\omega^{(t - 2)}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}}
\prod_{\tau = 1}^{t - 1} p(\omega^{(\tau)} \mid \omega^{(\tau - 1)}) p(v^{(\tau)} \mid \omega^{(\tau)})\ \Rightarrow s.t.\ \omega^{(t - 1)} = \omega_{j} \right] \\[5mm]
&= p(v^{(t)} \mid \omega_{i}) \sum_{j = 1}^{c} p(\omega_{i} \mid \omega_{j}) \alpha_{j}(t - 1)
\end{align*}
$$

### 前向算法
将上述条件概率$\alpha_{i}(t)$组合为条件概率向量
$$
\alpha(t) = (\alpha_{1}(t),\ \alpha_{2}(t),\ \cdots,\ \alpha_{n}(t))
$$
迭代方程重写为
$$
\alpha(t) = (b_{1k},\ b_{2k},\ \cdots,\ b_{nk}) \odot \left[ \alpha(t - 1) \mathbf{A} \right]
\ \Rightarrow s.t.\ v^{(t)} = v_{k}
$$
前向算法
$$
\begin{align*}
&variable \Rightarrow \mathbf{A},\ \mathbf{B},\ \mathbf{V}^{\mathrm{T}},\ \alpha(0 \sim \mathrm{T}) \\[5mm]
&initialize \Rightarrow \alpha(0) \\[5mm]
&for\ t\ in\ 1 \sim \mathrm{T} \Rightarrow \alpha(t) = (b_{1k},\ b_{2k},\ \cdots,\ b_{nk}) \odot 
\left[ \alpha(t - 1) \mathbf{A} \right] \\[5mm]
&p(\mathbf{V}^{\mathrm{T}}) = sum(\alpha(\mathrm{T}))
\end{align*}
$$

### 后向算法
调整求和以及连乘顺序，本质上是对前向算法做时间反演
$$
\sum_{\omega^{(\mathrm{T})}} \cdots \sum_{\omega^{(2)}} \sum_{\omega^{(1)}} \prod_{t = 1}^{\mathrm{T}} 
p(\omega^{(t)} \mid \omega^{(t - 1)}) p(v^{(t)} \mid \omega^{(t)})
\Rightarrow
\sum_{\omega^{(1)}} \cdots \sum_{\omega^{(\mathrm{T} - 1)}} \sum_{\omega^{(\mathrm{T})}} \prod_{t = \mathrm{T} - 1}^{0}
p(\omega^{(t + 1)} \mid \omega^{(t)}) p(v^{(t)} \mid \omega^{(t)})
$$
相应地，迭代变量和迭代方程设为
$$
\begin{align*}
\beta_{i}(t) &= \sum_{\omega^{(t + 1)}} \cdots \sum_{\omega^{(\mathrm{T} - 1)}} \sum_{\omega^{(\mathrm{T})}}
\prod_{\tau = \mathrm{T} - 1}^{t} p(\omega^{(\tau + 1)} \mid \omega^{(\tau)}) p(v^{(\tau)} \mid \omega^{(\tau)})
\ \Rightarrow s.t.\ \omega^{(t)} = \omega_{i} \\[5mm]
&= \sum_{j = 1}^{c} p(\omega_{i} \mid \omega_{j}) p(v^{(t)} \mid \omega_{i})
\left[ \sum_{\omega^{(t + 2)}} \cdots \sum_{\omega^{(\mathrm{T} - 1)}} \sum_{\omega^{(\mathrm{T})}}
\prod_{\tau = \mathrm{T} - 1}^{t + 1} p(\omega^{(\tau)} \mid \omega^{(\tau - 1)}) p(v^{(\tau)} \mid \omega^{(\tau)})
\ \Rightarrow s.t.\ \omega^{(t + 1)} = \omega_{j} \right] \\[5mm]
&= p(v^{(t)} \mid \omega_{i}) \sum_{j = 1}^{c} p(\omega_{i} \mid \omega_{j}) \beta_{j}(t + 1)
\end{align*}
$$
将上述条件概率$\beta_{i}(t)$组合为条件概率向量
$$
\beta(t) = (\beta_{1}(t),\ \beta_{2}(t),\ \cdots,\ \beta_{n}(t))
$$
迭代方程重写为
$$
\beta(t) = (b_{1k},\ b_{2k},\ \cdots,\ b_{nk}) \odot \left[ \beta(t + 1) \mathbf{A} \right]
\ \Rightarrow s.t.\ v^{(t)} = v_{k}
$$
后向算法
$$
\begin{align*}
&variable \Rightarrow \mathbf{A},\ \mathbf{B},\ \mathbf{V}^{\mathrm{T}},\ \beta(\mathrm{T} \sim 0) \\[5mm]
&initialize \Rightarrow \beta(\mathrm{T}) \\[5mm]
&for\ t\ in\ \mathrm{T} - 1 \sim 0 \Rightarrow \beta(t) = (b_{1k},\ b_{2k},\ \cdots,\ b_{nk}) \odot
\left[ \beta(t + 1) \mathbf{A} \right] \\[5mm]
&p(\mathbf{V}^{\mathrm{T}}) = sum(\beta(0))
\end{align*}
$$

## 解码问题
已知模型的的状态转移概率矩阵 $\mathbf{A}$、激发概率矩阵 $\mathbf{B}$ 和初始分布 $\pi$，计算最可能产生某一观测序列 $\mathbf{V}^{\mathrm{T}}$ 的隐状态序列 $\mathbf{\Omega}^{\mathrm{T}}$
$$
\begin{align*}
\hat{\mathbf{\Omega}}^{\mathrm{T}} &= \argmax_{\mathbf{\Omega}^{\mathrm{T}}} p(\mathbf{\Omega}^{\mathrm{T}} \mid \mathbf{V}^{\mathrm{T}}) \\[5mm]
&= \argmax_{\mathbf{\Omega}^{\mathrm{T}}} p(\mathbf{V}^{\mathrm{T}} \mid \mathbf{\Omega}^{\mathrm{T}}) p(\mathbf{\Omega}^{\mathrm{T}}) \\[5mm]
&= \argmax_{\mathbf{\Omega}^{\mathrm{T}}} \prod_{t = 1}^{\mathrm{T}} p(\omega^{(t)} \mid \omega^{(t - 1)}) p(v^{(t)} \mid \omega^{(t)})
\end{align*}
$$
### 贪心算法
每次转移选择最大激发概率的路径
$$
\hat{\omega}^{(t)} = \argmax_{\omega_{i}} \alpha_{i}(t)
$$
贪心算法 &rArr; 基于前向算法
$$
\begin{align*}
&variable \Rightarrow \mathbf{A},\ \mathbf{B},\ \mathbf{V}^{\mathrm{T}},\ \alpha(0 \sim \mathrm{T}), \mathrm{Path}\{\} \\[5mm]
&initialize \Rightarrow \alpha(0) \\[5mm]
&for\ t\ in\ 1 \sim \mathrm{T} \Rightarrow \\[5mm]
&\quad \quad \alpha(t) = (b_{1k},\ b_{2k},\ \cdots,\ b_{nk}) \odot 
\left[ \alpha(t - 1) \mathbf{A} \right] \\[5mm]
&\quad \quad \mathrm{Path}(t) = \argmax_{\omega_{i}} \alpha_{i}(t)
\end{align*}
$$
贪心算法选择的是局部最优路径，但有可能会产生非法路径

例如为了选择最大激发概率而转移到某个特殊的状态，该状态无法转出，并且对于剩余的序列激发概率为0

### 维特比算法
定义到$t$时刻为止，转移到隐状态$\omega_{i}$的所有部分路径的最大概率（部分最长“路径”）$\delta_{i}(t)$

这样可以过比较滤掉肯定不是最大路径的子路径，即对暴力搜索树进行剪枝操作

$\delta_{i}(t)$满足迭代方程
$$
\begin{align*}
\delta_{j}(t) &= \max_{\omega_{i}} \left[ p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1) p(v^{(t)} \mid \omega_{j}) \right] \\[5mm]
&= p(v^{(t)} \mid \omega_{j}) \max_{\omega_{i}} \left[ p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1) \right]
\end{align*}
$$
相应的，记录最大概率部分路径的前驱节点
$$
\begin{align*}
\Psi_{j}(t) &= \argmax_{\omega_{i}} p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1) p(v^{(t)} \mid \omega_{j}) \\[5mm]
&= \argmax_{\omega_{i}} p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1)
\end{align*}
$$
维特比算法 &rArr; 前向搜索 && 反向回溯
$$
\begin{align*}
&variable \Rightarrow \mathbf{A},\ \mathbf{B},\ \mathbf{V}^{\mathrm{T}},\ \delta(0 \sim \mathrm{T}),\ \Psi(1 \sim \mathrm{T}),\ \mathrm{Path}\{\} \\[5mm]
&initialize \Rightarrow \delta(0),\ \Psi(1) \\[5mm]
&for\ t\ in\ 1 \sim \mathrm{T} \Rightarrow\\[5mm]
&\quad \quad for\ j\ in\ 1 \sim c \Rightarrow \\[5mm]
&\quad \quad \quad \quad \delta_{j}(t) = p(v^{(t)} \mid \omega_{j}) 
\max_{\omega_{i}} \left[ p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1) \right] \\[5mm]
&\quad \quad \quad \quad \Psi_{j}(t) = \argmax_{\omega_{i}} p(\omega_{j} \mid \omega_{i}) \delta_{i}(t - 1) \\[5mm]
&\mathrm{Path}(\mathrm{T}) = \argmax_{\omega_{i}} \delta_{i}(\mathrm{T}) \\[5mm]
&for\ t\ in\ \mathrm{T} \sim 1 \Rightarrow \mathrm{Path}(t - 1) = \Psi_{i}(t) \ s.t.\ \mathrm{Path}(t) = \omega_{i}
\end{align*}
$$