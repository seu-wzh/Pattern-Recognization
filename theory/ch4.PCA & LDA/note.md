# 主成分分析（PCA）

为了降低样本的特征维度，将原特征空间$\mathbb{R}^{m}$中的样本通过线性变换降维到特征空间$\mathbb{R}^{p}$
$$
\boldsymbol{y} = \mathbf{W}^{\mathrm{T}} \boldsymbol{x} \Rightarrow \mathbb{R}^{\beta} = \mathbb{R}^{\beta \times \alpha} \mathbb{R}^{\alpha}
$$
降维的同时要求保留原特征空间的尺度，设置线性变换矩阵$\mathbf{W}$的$p$个列向量满足单位模长
$$
\mathbf{W}_{i}^{\mathrm{T}} \mathbf{W}_{i} = 1
$$
样本包含的信息量与样本的分散程度正相关，一般通过样本的**散布矩阵**来度量样本的散布情况
$$
\mathbf{S}_{t}(\boldsymbol{x}) = \sum_{i = 1}^{n} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})^{\mathrm{T}}
$$
为了最大化降维后的样本的每个维度包含的信息量，需要最大化降维样本的每个维度的特征方差和
$$
\begin{gather*}
\max_{\mathbf{W}} tr(\mathbf{S}_{t}(\boldsymbol{y})) =  tr\left[ \sum_{i = 1}^{n} (\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})(\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})^{\mathrm{T}} \right] \\ \\
s.t.\quad \mathbf{W}_{i}^{\mathrm{T}} \mathbf{W}_{i} = 1
\end{gather*}
$$
构造拉格朗日函数
$$
\begin{align*}
\mathcal{L}(\mathbf{W},\ \mathbf{\Lambda}) &= tr(\mathbf{S}_{t}(\boldsymbol{y})) - tr\bigg[\mathbf{\Lambda}(\mathbf{W}^{\mathrm{T}} \mathbf{W} - \mathbf{I}_{p})\bigg] \\ \\
&= tr\left[ \sum_{i = 1}^{n} (\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})(\boldsymbol{y}_{i} - \boldsymbol{\mu}_{y})^{\mathrm{T}} \right] - tr\bigg[\mathbf{\Lambda}(\mathbf{W}^{\mathrm{T}} \mathbf{W} - \mathbf{I}_{p})\bigg] \\ \\
&= tr\left[ \sum_{i = 1}^{n} \mathbf{W}^{\mathrm{T}} (\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{x})^{\mathrm{T}} \mathbf{W} \right] - tr\bigg[\mathbf{\Lambda}(\mathbf{W}^{\mathrm{T}} \mathbf{W} - \mathbf{I}_{p})\bigg] \\ \\
&= tr\left[ \mathbf{W}^{\mathrm{T}} \mathbf{S}_{t}(\boldsymbol{x}) \mathbf{W} \right] - tr\bigg[\mathbf{\Lambda}(\mathbf{W}^{\mathrm{T}} \mathbf{W} - \mathbf{I}_{p})\bigg]
\end{align*}
$$
其中拉格朗日乘子为对角阵（只对模长施加限制）
$$
\mathbf{\Lambda} = diag(\lambda_{1},\ \lambda_{2},\ \cdots,\ \lambda_{p})
$$
极大值点需要满足方程
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = 2 \mathbf{S}_{t}(\boldsymbol{x}) \mathbf{W} - 2 \mathbf{W} \mathbf{\Lambda} = \mathbf{0}^{\alpha \times \beta}
$$
即$\mathbf{W}$的所有列向量都是散布矩阵$\mathbf{S}_{t}(\boldsymbol{x})$的特征向量，由于散布矩阵为实对称矩阵，这些向量相互正交。代入原拉格朗日函数
$$
\mathcal{L}(\mathbf{\Lambda}) = tr(\mathbf{\Lambda} (\mathbf{W}^{\mathrm{T}} \mathbf{W})) - tr\bigg[\mathbf{\Lambda}(\mathbf{W}^{\mathrm{T}} \mathbf{W} - \mathbf{I}_{p})\bigg] = tr(\mathbf{\Lambda})
$$
优化问题转化为
$$
\max_{\mathbf{\Lambda}} \mathcal{L}(\mathbf{\Lambda}) = \max_{\Lambda} tr(\mathbf{\Lambda})
$$
即选取特征值最大的$\beta$个单位特征向量作为基向量进行投影

需要注意的是，$\mathrm{PCA}$降维手段只适用于类正态分布的样本，这相当于是一个超椭球在做投影，选取较长的主轴投影可以使得降维样本尽可能的分散。为了衡量降维后样本整体的散布程度，上述推导中使用了散布矩阵的迹作为衡量标准；散布矩阵的行列式也可以起到相同的作用，但得到的解基向量不唯一。

# 线性判别分析（LDA）
与$\mathrm{PCA}$相似，使用线性变换来对样本进行进行降维
$$
\boldsymbol{y} = \mathbf{W}^{\mathrm{T}} \boldsymbol{x} \Rightarrow \mathbb{R}^{\beta} = \mathbb{R}^{\beta \times \alpha} \mathbb{R}^{\alpha}
$$
为了在降维的同时使得样本更加可分，需要减小每类样本的散布程度并增大类间样本的散布程度

通过**类内散布矩阵**$\mathbf{S}_{w}$来衡量类内样本的散布程度
$$
\mathbf{S}_{w}(\boldsymbol{x}) = \sum_{i = 1}^{c} \mathbf{S}_{i} = \sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})(\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}}
$$
直接衡量类间样本的散布程度较为困难，先考虑**全局散布矩阵**$\mathbf{S}_{t}$
$$
\begin{align*}
\mathbf{S}_{t}(\boldsymbol{x}) &= \sum_{\boldsymbol{x}} (\boldsymbol{x} - \boldsymbol{\mu}) (\boldsymbol{x} - \boldsymbol{\mu})^{\mathrm{T}} \\ \\
&= \sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i} + \boldsymbol{\mu}_{i} - \boldsymbol{\mu}) (\boldsymbol{x} - \boldsymbol{\mu}_{i} + \boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}} \\ \\
&= \sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})(\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} + \sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu}) (\boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}} \\ \\
&= \mathbf{S}_{w}(\boldsymbol{x}) + \sum_{i = 1}^{c} n_{i} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu}) (\boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}}
\end{align*}
$$
其中交叉项部分
$$
\begin{gather*}
\sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu}) (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} = \sum_{i = 1}^{c} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu}) \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} = 0 \\ \\
\sum_{i = 1}^{c} \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) (\boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}} = \sum_{\boldsymbol{x} \in \mathcal{X}_{i}} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) \sum_{i = 1}^{c} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}} = 0
\end{gather*}
$$
由于全局散布程度是由类内散布程度和类间散布程度共同贡献的，合理地将**类间散布矩阵**$\mathbf{S}_{b}$定义为
$$
\mathbf{S}_{b}(\boldsymbol{x}) = \sum_{i = 1}^{c} n_{i} (\boldsymbol{\mu}_{i} - \boldsymbol{\mu}) (\boldsymbol{\mu}_{i} - \boldsymbol{\mu})^{\mathrm{T}}
$$
以上关系可以表示为
$$
\mathbf{S}_{t}(\boldsymbol{x}) = \mathbf{S}_{w}(\boldsymbol{x}) + \mathbf{S}_{b}(\boldsymbol{x})
$$
为了在最大化类间散布程度的同时最小化类内散布程度，定义**广义瑞利商**
$$
J(\mathbf{W}) = \sum_{i = 1}^{\beta} \frac{\mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{b} \mathbf{W}_{i}}{\mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{w} \mathbf{W}_{i}}
$$
其中$\mathbf{W}_{i}$为$\mathbf{W}$的列向量。为了便于优化，可以固定$\mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{w} \mathbf{W}_{i} = 1$，优化问题为
$$
\begin{gather*}
\max_{\mathbf{W}} J(\mathbf{W}) = \max_{\mathbf{W}} \sum_{i = 1}^{\beta} \mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{b} \mathbf{W}_{i} \\ \\
s.t.\quad \mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{w} \mathbf{W}_{i} = 1
\end{gather*}
$$
构造拉格朗日函数
$$
\mathcal{L}(\mathbf{W},\ \lambda) = \sum_{i = 1}^{\beta} \mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{b} \mathbf{W}_{i} - \sum_{i = 1}^{\beta} \lambda_{i} (\mathbf{W}_{i}^{\mathrm{T}} \mathbf{S}_{w} \mathbf{W}_{i} - 1)
$$
极大值点需要满足方程
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = 2 \mathbf{S}_{b} \mathbf{W} - 2 \mathbf{S}_{w} \mathbf{W} \mathbf{\Lambda} = 0
$$
其中$\mathbf{\Lambda} = diag(\lambda_{1},\ \lambda_{2},\ \cdots,\ \lambda_{\beta})$。如果类内散布矩阵$\mathbf{S}_{w}$可逆，$\mathbf{W}$满足
$$
\mathbf{S}_{w}^{-1} \mathbf{S}_{b} \mathbf{W} = \mathbf{W} \mathbf{\Lambda}
$$
即$\mathbf{W}$所有的列向量均为$\mathbf{S}_{w}^{-1} \mathbf{S}_{b}$的特征向量，由于$\mathbf{S}_{w}^{-1} \mathbf{S}_{b}$为实对称矩阵，这些向量相互正交。在$\mathbf{S}_{w}$不可逆时，也可以将$\mathbf{S}_{w}$替换为
$$
\mathbf{S}_{w} \leftarrow \mathbf{S}_{w} + \epsilon \mathbf{I}_{\beta},\quad \epsilon > 0
$$
或者通过特征多项式方程解出特征值
$$
|\mathbf{S}_{b} - \lambda_{i} \mathbf{S}_{w}| = 0
$$
进而通过线性方程解出对应的特征向量
$$
(\mathbf{S}_{b} - \lambda_{i} \mathbf{S}_{w}) \mathbf{W}_{i} = 0
$$
将方程代入拉格朗日函数
$$
\mathcal{L}(\mathbf{\Lambda}) = \sum_{i = 1}^{\beta} \lambda_{i} = tr(\mathbf{\Lambda})
$$
优化问题转换为
$$
\max_{\mathbf{\Lambda}} \mathcal{L}(\mathbf{\Lambda}) = \max_{\mathbf{\Lambda}} tr(\mathbf{\Lambda})
$$
即选取最大的$\beta$个特征值对应的特征向量作为投影方向