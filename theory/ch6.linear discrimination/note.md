# 线性判别

**线性判别函数**
$$
g(\boldsymbol{x}) = \boldsymbol{w}^{\mathrm{T}} \boldsymbol{x} + w_{0} \Rightarrow \mathbb{R} = \mathbb{R}^{d} \cdot \mathbb{R}^{d} + \mathbb{R}
$$
其中$\boldsymbol{w}$为**权向量**，$w_{0}$为**偏置**，通过对特征向量$\boldsymbol{x}$的特征的线性组合形成判别函数

**广义线性判别函数**
$$
g(\boldsymbol{x}) = \boldsymbol{w}^{\mathrm{T}} \varphi(\boldsymbol{x}) + w_{0} = \boldsymbol{a}^{\mathrm{T}} \boldsymbol{y}
$$
先将样本映射到另一特征空间$\mathbb{R}^{\hat{d}}$再进行线性组合，其中权向量和映射特征向量均为增广形式
$$
\boldsymbol{a} = \begin{pmatrix}
    w_{0} \\[3mm]
    w_{1} \\[3mm]
    \vdots \\[3mm]
    w_{\hat{d}}
\end{pmatrix}
\qquad
\boldsymbol{y} = \begin{pmatrix}
    1 \\[3mm]
    \varphi_{1}(\boldsymbol{x}) \\[3mm]
    \vdots \\[3mm]
    \varphi_{\hat{d}}(\boldsymbol{x})
\end{pmatrix}
$$
在二分类中可以将样本进行符号归一化，对负类样本取反，相应的分类器的目标可以表示为
$$
\forall\ \boldsymbol{y} \in D \quad g(\boldsymbol{x}) = \boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} > 0
$$
为了提高分类器的鲁棒性，可以额外设置**边沿裕量$b$**
$$
\forall\ \boldsymbol{y} \in D \quad g(\boldsymbol{x}) = \boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} \ge b > 0
$$
在寻找能够满足上述条件的解向量时可以定义一个**准则函数$J(\boldsymbol{a})$**，并以解向量作为极小值点
$$
\hat{\boldsymbol{a}} = \argmin_{\boldsymbol{a}} J(\boldsymbol{a})
$$
如果准则函数可微，可以采取梯度优化的方式求解（梯度下降法、牛顿下降法。。。）

## 感知器算法
**感知器准则函数**
$$
J_{p}(\boldsymbol{a}) = -\sum_{\boldsymbol{y} \in \mathcal{Y}} \boldsymbol{a}^{\mathrm{T}} \boldsymbol{y}
$$
其中$\mathcal{Y}$代表被感知器$\boldsymbol{a}$错分的样本集合，错分的样本满足
$$
\forall\ \boldsymbol{y} \in \mathcal{Y} \quad \boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} \le 0
$$
当感知器能够正确分类所有样本时，准则函数$J_{p} = 0 = \min J_{p}$

准则函数$J_{p}$对权向量$\boldsymbol{a}$的梯度
$$
\frac{\partial J_{p}}{\partial \boldsymbol{a}} = -\sum_{\boldsymbol{y} \in \mathcal{Y}} \boldsymbol{y}
$$

### 校正方法
* 批处理校正：每次权重更新使用所有的错误样本
$$
\boldsymbol{a} \leftarrow \boldsymbol{a} + \sum_{\boldsymbol{y} \in \mathcal{Y}} \eta \boldsymbol{y}
$$
* 单样本校正：每次权重更新只使用一个错误样本
$$
\boldsymbol{a} \leftarrow \boldsymbol{a} + \eta \boldsymbol{y}
$$
* 带裕量校正：每次权重更新时使用错误样本 + 边沿裕量内的样本
$$
\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} \le b \Rightarrow \boldsymbol{y} \in \mathcal{Y}
$$
* 固定增量校正：权重更新的学习率固定
$$
\eta(k) = \eta_{0}
$$
* 变增量校正：权重更新的学习率可变
$$
\eta(k) = \frac{1}{k},\ k,\ \cdots
$$
* 松弛校正：为了兼顾梯度的连续和样本模长带来的影响，重写准则函数
$$
\begin{gather*}
J_{r}(\boldsymbol{a}) = \frac{1}{2} \sum_{\boldsymbol{y} \in \mathcal{Y}} \frac{(\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} - b)^{2}}{|| \boldsymbol{y} ||^{2}} \\ \\
\frac{\partial J_{r}}{\partial \boldsymbol{a}} = \sum_{\boldsymbol{y} \in \mathcal{Y}} \frac{\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} - b}{|| \boldsymbol{y} ||^{2}} \boldsymbol{y}
\end{gather*}
$$

通过对以上算法的组合可以形成各种感知器算法

## 最小平方误差方法（MSE）
线性不可分问题可以通过一个无解的超定线性方程来描述
$$
\mathbf{Y} \boldsymbol{a} = \boldsymbol{b} \Rightarrow \mathbb{R}^{n \times (d + 1)} \mathbb{R}^{(d + 1)} = \mathbb{R}^{n}
$$
其中$\mathbf{Y}$为样本构成的矩阵，$\boldsymbol{b}$为人为设定的每个样本的裕量（例如$\mathbf{1}^{n}$）

定义平方误差准则函数
$$
J_{s}(\boldsymbol{a}) = || \mathbf{Y} \boldsymbol{a} - \boldsymbol{b} ||^{2} = \sum_{i = 1}^{n} (\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y}_{i} - b_{i})^{2}
$$
极小值点满足方程
$$
\frac{\partial J_{s}}{\partial \boldsymbol{a}} = \sum_{i = 1}^{n} 2 (\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y} - b_{i}) \boldsymbol{y}_{i} = 2 \mathbf{Y}^{\mathrm{T}} (\mathbf{Y} \boldsymbol{a} - \boldsymbol{b}) = 0
$$
原超定线性方程转化为
$$
\mathbf{Y}^{\mathrm{T}} \mathbf{Y} \boldsymbol{a} = \mathbf{Y}^{\mathrm{T}} \boldsymbol{b}
$$
如果方阵$\mathbf{Y}^{\mathrm{T}} \mathbf{Y}$可逆，可以得到唯一解
$$
\hat{\boldsymbol{a}} = (\mathbf{Y}^{\mathrm{T}} \mathbf{Y})^{-1} \mathbf{Y}^{\mathrm{T}} \boldsymbol{b} = \mathbf{Y}^{\dagger} \boldsymbol{b}
$$
其中$\mathbf{Y}^{\dagger} = (\mathbf{Y}^{\mathrm{T}} \mathbf{Y})^{-1} \mathbf{Y}^{\mathrm{T}}$被称为$\mathbf{Y}$的伪逆矩阵，考虑到$\mathbf{Y}^{\mathrm{T}} \mathbf{Y}$不可逆的情况，更一般的定义为
$$
\mathbf{Y}^{\dagger} = \lim_{\epsilon \to 0}(\mathbf{Y}^{\mathrm{T}} \mathbf{Y} + \epsilon \mathbf{I}_{(d + 1)})^{-1} \mathbf{Y}^{\mathrm{T}}
$$
并且以上极限存在，且$\boldsymbol{a} = \mathbf{Y}^{\dagger} \boldsymbol{b}$是原方程的一个$\mathrm{MSE}$解

同时，利用准则函数的梯度也可以进行梯度优化求出权向量的数值解

## 最小均方算法（LMS）| Widrow-Hoff算法
考虑到训练集样本数量可能较为庞大，将$\mathrm{MSE}$的梯度优化算法改写为单样本校正
$$
\boldsymbol{a} \leftarrow \boldsymbol{a} - \eta(k) (\boldsymbol{a}^{\mathrm{T}} \boldsymbol{y}^{(k)} - b^{(k)}) \boldsymbol{y}^{(k)}
$$
与松弛校正方法采取的误分类样本校正不同的是，$\mathrm{LMS}$会持续进行更新，因此学习率$\eta(k)$需要随$k$递减