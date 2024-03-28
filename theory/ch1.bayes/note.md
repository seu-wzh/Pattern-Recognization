# 贝叶斯分类器

**类别后验概率**； 在样本特征$X$的观测值$x$时 判断样本属于类别$\omega_{i}$的概率
$$
p(\omega_{i} \mid x) = \frac{p(x \mid \omega_{i}) p(\omega_{i})}{p(x)}
$$
**类别条件概率密度**； 在类别$\omega_{i}$下样本特征$X$的概率密度
$$
p(x \mid \omega_{i})
$$
**类别先验概率**； 类别$\omega_{i}$占样本空间的比例
$$
p(\omega_{i})
$$
**证据因子**； 样本特征$X$的概率密度
$$
p(x) = \sum_{i = 1}^{c}p(x \mid \omega_{i}) p(\omega_{i})
$$
**条件误分类概率**； 在样本特征$X$的观测值为$x$时 将样本判断为类别$\omega_{i}$出错的概率
$$
p(error \mid x) = 1 - p(\omega_{i} \mid x)
$$
**误分类概率**； 条件误分类概率$p(error \mid x)$的期望
$$
p(error) = \int_{-\infty}^{\infty} p(error \mid x) p(x)dx
$$
**贝叶斯分类器**； 使分类产生的误分类概率$p(error)$最小的决策方法
$$
\hat{\omega} = \argmin_{\omega}p(error) = \argmin_{\omega}p(error \mid x) = \argmax_{\omega}p(\omega_{i} \mid x)
$$
贝叶斯分类器将后验概率最大的类别作为预测值，拥有理论上的最优性能

# 贝叶斯决策规则

**类别集**
$$
\omega_{1}, \omega_{2}, \dots, \omega_{c}
$$
**行为集**
$$
\alpha_{1}, \alpha_{2}, \dots, \alpha_{a}
$$
**风险函数**；真实类别为$\omega_{j}$时采取行动$\alpha_{i}$的风险
$$
\lambda(\alpha_{i} \mid \omega_{j})
$$
**特征向量**
$$
\mathbf{X} = 
\begin{pmatrix}
x_{1} &
x_{2} &
\cdots &
x_{d}
\end{pmatrix}^{\mathrm{T}}
$$
**条件风险**；在样本特征$\mathbf{X}$的观测值为$\boldsymbol{x}$时 采取行为$\alpha_{i}$的风险
$$
R(\alpha_{i} \mid \boldsymbol{x}) = \sum_{j = 1}^{c} \lambda(\alpha_{i} \mid \omega_{j}) p(\omega_{j} \mid \boldsymbol{x})
$$
**行为函数**；样本特征$\mathbf{X}$被观测为$\boldsymbol{x}$时采取的行为
$$
{\alpha}(\boldsymbol{x}),\quad {\alpha} : \mathbb{R}^{d} \to \left \{ \alpha_{1}, \alpha_{2}, \dots, \alpha_{a} \right \}
$$
**总风险**；在行为函数${\alpha}(\boldsymbol{x})$下的条件风险$R({\alpha}(\boldsymbol{x}) \mid \boldsymbol{x})$的期望
$$
\mathfrak{R} = \int_{\mathbb{R}^{d}} R(\boldsymbol{\alpha}(\boldsymbol{x}) \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x}
$$
**贝叶斯决策规则**；使行为产生的总风险$\mathfrak{R}$最小的决策方法
$$
\boldsymbol{\alpha}(\boldsymbol{x}) = \argmin_{\alpha_{i}} \mathfrak{R} = \argmin_{\alpha_{i}} R(\alpha_{i} \mid \boldsymbol{x}) = \argmin_{\alpha_{i}} \sum_{j = 1}^{c} \lambda(\alpha_{i} \mid \omega_{j}) p(\omega_{j} \mid \boldsymbol{x})
$$
**最小误差率分类**；贝叶斯决策规则角度下的贝叶斯分类器
$$
\begin{gather*}
\lambda(\alpha_{i} \mid \omega_{j}) = 
\left\{
\begin{matrix}
0,\quad i = j \\[3mm]
1, \quad i \ne j
\end{matrix}
\right. \\ \\
\alpha(\boldsymbol{x}) = \argmin_{\alpha_{i}} R(\alpha_{i} \mid \boldsymbol{x}) = \argmax_{\alpha_{i}} p(\omega_{i} \mid \boldsymbol{x})
\end{gather*}
$$
调整条件风险即可在贝叶斯分类器的基础上实现不同类别的判决区域的扩张或收缩

# 极小极大化准则

在二分类中确定了行为$\alpha_{1}$和$\alpha_{2}$的区域$\mathcal{R}_{1}$和$\mathcal{R}_{2}$后总风险可以表示为
$$
\begin{align*}
\mathfrak{R} &= \int_{\mathcal{R}_{1}} R(\alpha_{1} \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} + \int_{\mathcal{R}_{2}} R(\alpha_{2} \mid \boldsymbol{x}) p(\boldsymbol{x}) d\boldsymbol{x} \\ \\
&= \int_{\mathcal{R}_{1}} (\lambda_{11} p(\omega_{1} \mid \boldsymbol{x}) + \lambda_{12} p(\omega_{2} \mid \boldsymbol{x})) p(\boldsymbol{x}) d\boldsymbol{x} + \int_{\mathcal{R}_{2}} (\lambda_{21} p(\omega_{1} \mid \boldsymbol{x}) + \lambda_{22} p(\omega_{2} \mid \boldsymbol{x})) p(\boldsymbol{x}) d\boldsymbol{x} \\ \\
&= \int_{\mathcal{R}_{1}} (\lambda_{11} p(\boldsymbol{x} \mid \omega_{1}) p(\omega_{1}) + \lambda_{12} p(\boldsymbol{x} \mid \omega_{2})p(\omega_{2}))
d\boldsymbol{x} + \int_{\mathcal{R}_{2}} (\lambda_{21} p(\boldsymbol{x} \mid \omega_{1}) p(\omega_{1}) + \lambda_{22} p(\boldsymbol{x} \mid \omega_{2})p(\omega_{2})) d\boldsymbol{x} \\ \\
&= p(\omega_{1}) (\lambda_{11} \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega_{1}) d\boldsymbol{x} + \lambda_{21} \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega_{1}) d\boldsymbol{x}) + p(\omega_{2}) (\lambda_{12} \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega_{2}) d\boldsymbol{x} + \lambda_{22} \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega_{2})d\boldsymbol{x}) \\
\end{align*}
$$
通过关系
$$
p(\omega_{1}) + p(\omega_{2}) = 1,\quad \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega) d\boldsymbol{x} + \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega) d\boldsymbol{x} = 1
$$
将总风险化为
$$
\mathfrak{R}(p(\omega_{1})) = \mathfrak{R}_{mm} + \mu p(\omega_{1})
$$
其中
$$
\begin{gather*}
\begin{align*}
\mathfrak{R}_{mm} &= \lambda_{22} + (\lambda_{12} - \lambda_{22}) \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega_{2}) d\boldsymbol{x} \\ \\
&= \lambda_{11} + (\lambda_{21} - \lambda_{11}) \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega_{1}) d\boldsymbol{x}
\end{align*} \\ \\
\begin{align*}
\mu &= (\lambda_{11} - \lambda_{21}) \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega_{1}) d\boldsymbol{x} - (\lambda_{12} - \lambda_{22}) \int_{\mathcal{R}_{1}} p(\boldsymbol{x} \mid \omega_{2}) d\boldsymbol{x} + (\lambda_{21} - \lambda_{22}) \\ \\
&= (\lambda_{22} - \lambda_{12}) \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega_{2}) d\boldsymbol{x} - (\lambda_{21} - \lambda_{11}) \int_{\mathcal{R}_{2}} p(\boldsymbol{x} \mid \omega_{1}) d\boldsymbol{x} + (\lambda_{12} - \lambda_{11})
\end{align*}
\end{gather*}
$$
实际情况中类别条件概率密度$p(\boldsymbol{x} \mid \omega)$相对固定，但类别先验概率$p(\omega)$可能会出现较大的变化，选取合适的行为区域$\mathcal{R}$使得式中的系数$\boldsymbol{\mu} = 0$时
$$
\mathfrak{R} = \mathfrak{R}_{mm}
$$
使得总风险与先验概率无关，避免类别先验概率产生较大改变时出现极大的风险，也就是最小化可能的最大风险

# 贝叶斯分类器的判别函数

一般的分类器可以通过$c$个判别函数$g(\boldsymbol{x})$来表述
$$
\hat{\omega} = \argmax_{\omega_{i}} g_{i}(\boldsymbol{x})
$$
同时，判别函数的单调递增映射仍然是判决函数。贝叶斯分类器可以表述为
$$
\begin{gather*}
g_{i}(\boldsymbol{x}) = -R(\alpha_{i} \mid \boldsymbol{x}) \\ \\
g_{i}(\boldsymbol{x}) = p(\omega_{i} \mid \boldsymbol{x}) = \frac{p(x \mid \omega_{i}) p(\omega_{i})}{p(x)} \\ \\
g_{i}(\boldsymbol{x}) = p(x \mid \omega_{i}) p(\omega_{i}) \\ \\
g_{i}(\boldsymbol{x}) = \ln{p(x \mid \omega_{i})} + \ln{p(\omega_{i})}
\end{gather*}
$$
对于二分类而言，可以采用单个判别函数
$$
\begin{gather*}
g(\boldsymbol{x}) = g_{1}(\boldsymbol{x}) - g_{2}(\boldsymbol{x}) \\ \\
\hat{\omega} = 
\left\{
\begin{matrix}
\omega_{1},\quad g(\boldsymbol{x}) >   0 \\ \\
\omega_{2},\quad g(\boldsymbol{x}) \le 0
\end{matrix}
\right.
\end{gather*}
$$
假设在每一类中样本特征$\mathbf{X} \sim N(\boldsymbol{\mu},\ \boldsymbol{\Sigma})$，即
$$
p(\boldsymbol{x}) = \frac{1}{(2\pi)^{\frac{d}{2}} \left| \boldsymbol{\Sigma} \right|^{\frac{1}{2}} }
\exp \left[ -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right]
$$
取判别函数为
$$
\begin{align*}
g_{i}(\boldsymbol{x}) &= \ln{p(\boldsymbol{x} \mid \omega_{i})} + \ln{p(\omega_{i})} \\ \\
&= -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \boldsymbol{\Sigma}_{i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) -
\frac{d}{2} \ln{2\pi} - \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_{i} \right| + \ln{p(\omega_{i})} \\ \\
&\Rightarrow -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \boldsymbol{\Sigma}_{i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) -
\frac{1}{2} \ln \left| \boldsymbol{\Sigma}_{i} \right| + \ln{p(\omega_{i})}
\end{align*}
$$

考虑以下三种情况
* $\boldsymbol{\Sigma}_{i} = \boldsymbol{\Sigma} = \sigma^{2} \boldsymbol{I}$
$$
\begin{align*}
g_{i}(\boldsymbol{x}) &= -\frac{1}{2\sigma^{2}} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) -
\frac{1}{2} \ln \left| \boldsymbol{\Sigma} \right| + \ln{p(\omega_{i})} \\ \\
&\Rightarrow -\frac{1}{2\sigma^{2}} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) ^{\mathrm{T}} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) + \ln{p(\omega_{i})} \\ \\
&= -\frac{1}{2\sigma^{2}} (\boldsymbol{x} ^{\mathrm{T}} \boldsymbol{x} - 2\boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{x} + \boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{\mu}_{i}) + \ln{p(\omega_{i})} \\ \\
&\Rightarrow -\frac{1}{2\sigma^{2}} (-2\boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{x} + \boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{\mu}_{i}) + \ln{p(\omega_{i})} \\ \\
&= \frac{1}{\sigma^2} \boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{x} - \frac{1}{2\sigma^2} \boldsymbol{\mu}_{i} ^{\mathrm{T}} \boldsymbol{\mu}_{i} + \ln{p(\omega_{i})}
\end{align*}
$$
* $\boldsymbol{\Sigma}_{i} = \boldsymbol{\Sigma} = any$
$$
\begin{align*}
g_{i}(\boldsymbol{x}) &= -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) -
\frac{1}{2} \ln \left| \boldsymbol{\Sigma} \right| + \ln{p(\omega_{i})} \\ \\
&\Rightarrow -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) + \ln{p(\omega_{i})} \\ \\
&= -\frac{1}{2} (\boldsymbol{x}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} + \boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{i} - 
2 \boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x}) + \ln{p(\omega_{i})} \\ \\
&\Rightarrow -\frac{1}{2} (\boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{i} - 2\boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x}) + \ln{p(\omega_{i})} \\ \\
&= \boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} - \frac{1}{2} \boldsymbol{\mu}_{i}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{i} + \ln{p(\omega_{i})}
\end{align*}
$$
* $\boldsymbol{\Sigma}_{i} = any$
$$
g_{i}(\boldsymbol{x}) = -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_{i})^{\mathrm{T}} \boldsymbol{\Sigma}_{i}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_{i}) -
\frac{1}{2} \ln \left| \boldsymbol{\Sigma}_{i} \right| + \ln{p(\omega_{i})}
$$

在二分类问题中，上述判决函数可以表述为
* $\boldsymbol{\Sigma}_{i} = \boldsymbol{\Sigma} = \sigma^{2} \boldsymbol{I}$
$$
\begin{align*}
g(\boldsymbol{x}) &= g_{1}(\boldsymbol{x}) - g_{2}(\boldsymbol{x}) \\ \\
&\Rightarrow (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}) ^{\mathrm{T}} \left[ \boldsymbol{x} - \frac{\boldsymbol{\mu}_{1} + \boldsymbol{\mu}_{2}}{2} +
\sigma^{2} \frac{\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}}{|| \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2} ||^{2}}\ln{\frac{p(\omega_{1})}{p(\omega_{2})}}\right] \\ \\
&= \boldsymbol{w} ^{\mathrm{T}} (\boldsymbol{x} - \boldsymbol{x}_{0})
\end{align*}
$$
其中
$$
\begin{gather*}
\boldsymbol{w} = \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2} \\ \\
\boldsymbol{x}_{0} = \frac{\boldsymbol{\mu}_{1} + \boldsymbol{\mu}_{2}}{2} - \sigma^{2} \frac{\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}}{||\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}||^{2}}\ln{\frac{p(\omega_{1})}{p(\omega_{2})}}
\end{gather*}
$$
判决边界$g(\boldsymbol{x}) = 0$是一个超平面，且垂直于均值向量的连线，并且会偏向先验概率较小的那个类别
<br />
* $\boldsymbol{\Sigma}_{i} = \boldsymbol{\Sigma} = any$
$$
\begin{align*}
g(\boldsymbol{x}) &= g_{1}(\boldsymbol{x}) - g_{2}(\boldsymbol{x}) \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} -
\frac{1}{2} (\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2}) +
\ln{\frac{p(\omega_{1})}{p(\omega_{2})}} \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} -
\frac{1}{2} \left[\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2} +
\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2} - \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2}\right] +
\ln{\frac{p(\omega_{1})}{p(\omega_{2})}} \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} -
\frac{1}{2} \left[\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}) +
(\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2}\right] +
\ln{\frac{p(\omega_{1})}{p(\omega_{2})}} \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} -
\frac{1}{2} \left[(\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{1} +
(\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_{2}\right] +
\ln{\frac{p(\omega_{1})}{p(\omega_{2})}} \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \boldsymbol{x} -
\frac{1}{2} (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_{1} +\boldsymbol{\mu}_{2}) + \ln{\frac{p(\omega_{1})}{p(\omega_{2})}} \\ \\
&= (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} \left[ \boldsymbol{x} - \frac{\boldsymbol{\mu}_{1} + \boldsymbol{\mu}_{2}}{2} +
\frac{\boldsymbol{\Sigma}(\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})}{\mid\mid \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2} \mid\mid^{2}}\ln{\frac{p(\omega_{1})}{p(\omega_{2})}}\right]
= \boldsymbol{w} ^{\mathrm{T}} (\boldsymbol{x} - \boldsymbol{x}_{0})
\end{align*}
$$
其中
$$
\begin{gather*}
\boldsymbol{w} = \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}) \\ \\
\boldsymbol{x}_{0} = \frac{\boldsymbol{\mu}_{1} + \boldsymbol{\mu}_{2}}{2} -
\frac{\boldsymbol{\Sigma}(\boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2})}{\mid\mid \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2} \mid\mid^{2}}\ln{\frac{p(\omega_{1})}{p(\omega_{2})}}
\end{gather*}
$$
判决边界$g(\boldsymbol{x}) = 0$是一个超平面，且不再垂直于均值向量的连线，但仍会偏向先验概率较小的那个类别
<br/>
* $\boldsymbol{\Sigma}_{i} = any$
$$
\begin{align*}
g(\boldsymbol{x}) &= g_{1}(\boldsymbol{x}) - g_{2}(\boldsymbol{x}) \\ \\
&= -\frac{1}{2} \boldsymbol{x}^{\mathrm{T}} (\boldsymbol{\Sigma}_{1}^{-1} - \boldsymbol{\Sigma}_{2}^{-1}) \boldsymbol{x} + (\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}_{1}^{-1} - \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}_{2}^{-1}) \boldsymbol{x} + \ln{\frac{p(\omega_{1})}{p(\omega_{2})}} - \frac{1}{2} (\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}_{1}^{-1} \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}_{2}^{-1} \boldsymbol{\mu}_{2}) \\ \\
&= \boldsymbol{x}^{\mathrm{T}} \mathbf{W} \boldsymbol{x} + \boldsymbol{w} ^{\mathrm{T}} \boldsymbol{x} + b
\end{align*}
$$
其中
$$
\begin{gather*}
\mathbf{W} = -\frac{1}{2} (\boldsymbol{\Sigma}_{1}^{-1} - \boldsymbol{\Sigma}_{2}^{-1}) \\ \\
\boldsymbol{w} = \boldsymbol{\Sigma}_{1}^{-1} \boldsymbol{\mu}_{1} - \boldsymbol{\Sigma}_{2}^{-1} \boldsymbol{\mu}_{2} \\ \\
b = \ln{\frac{p(\omega_{1})}{p(\omega_{2})}} - \frac{1}{2} (\boldsymbol{\mu}_{1}^{\mathrm{T}} \boldsymbol{\Sigma}_{1}^{-1} \boldsymbol{\mu}_{1} - \boldsymbol{\mu}_{2}^{\mathrm{T}} \boldsymbol{\Sigma}_{2}^{-1} \boldsymbol{\mu}_{2})
\end{gather*}
$$
判决边界$g(\boldsymbol{x}) = 0$转变为为二次超曲面