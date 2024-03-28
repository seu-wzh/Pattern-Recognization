在同一概率分布下独立地观测$n$个样本构成样本集，样本之间在参数分布上**条件独立**
$$
D = \left \{ \boldsymbol{x}_{1},\ \boldsymbol{x}_{2},\ \cdots,\ \boldsymbol{x}_{n} \right \}
$$
假设含有未知参数$\theta$（假设参数是一个固定值）的样本分布为
$$
p(\boldsymbol{x} \mid \theta)
$$
**似然函数**；在未知参数$\theta$下观测到这个样本集的概率密度
$$
p(D \mid \theta) = \prod_{k = 1}^{n} p(\boldsymbol{x}_{k} \mid \theta)
$$

# 最大似然估计

假设分布中的未知参数$\theta$是客观存在的固定的值，且在观测前没有任何与参数相关的先验知识。参数$\theta$应当最大化地符合观测结果，即使观测结果出现的概率最大
$$
\hat{\theta} = \argmax_{\theta} p(D \mid \theta)
$$
**对数似然函数**
$$
\ell(\theta) = \ln p(D \mid \theta) = \sum_{k = 1}^{n} \ln p(\boldsymbol{x}_{k} \mid \theta)
$$
问题转换为
$$
\begin{gather*}
\hat{\theta} = \argmax_{\theta} \ell(\theta) \\ \\
\nabla_{\theta} \ell(\hat{\theta}) = \sum_{k = 1}^{n} \nabla_{\theta} \ln p(\boldsymbol{x}_{k} \mid \hat{\theta}) = 0
\end{gather*}
$$

### 正态分布参数的最大似然估计

对数似然函数
$$
\begin{align*}
\ell(\mu;\ \boldsymbol{\Sigma}) &= \sum_{k = 1}^{n} \ln p(\boldsymbol{x}_{k} \mid \theta) \\ \\
&= \sum_{k = 1}^{n} \ln \left[ \frac{1}{(2\pi)^{\frac{d}{2}} \mid\boldsymbol{\Sigma}\mid^{\frac{1}{2}}}
\exp[-\frac{1}{2} (\boldsymbol{x}_{k} - \mu)^\mathrm{T} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x}_{k} - \mu)] \right] \\ \\
&= \sum_{k = 1}^{n} \left[  -\frac{1}{2} (\boldsymbol{x}_{k} - \mu)^\mathrm{T} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x}_{k} - \mu) - \frac{d}{2} \ln(2\pi) - \frac{1}{2} \ln\mid\boldsymbol{\Sigma}\mid \right] \\ \\
&\Rightarrow \sum_{k = 1}^{n} \left[ -\frac{1}{2} (\boldsymbol{x}_{k} - \mu)^\mathrm{T} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x}_{k} - \mu) - \frac{1}{2} \ln\mid\boldsymbol{\Sigma}\mid \right]
\end{align*}
$$
参数估计
$$
\begin{gather*}
\nabla_{\mu} \ell(\hat{\mu};\ \hat{\boldsymbol{\Sigma}}) = \hat{\boldsymbol{\Sigma}}^{-1} \sum_{k = 1}^{n} (\boldsymbol{x}_{k} - \hat{\mu}) = 0 \\ \\
\nabla_{\boldsymbol{\Sigma}^{-1}} \ell(\hat{\mu};\ \hat{\boldsymbol{\Sigma}}) = \sum_{k = 1}^{n} \left[
-\frac{1}{2} (\boldsymbol{x}_{k} - \hat{\mu}) (\boldsymbol{x}_{k} - \hat{\mu})^\mathrm{T} + \frac{1}{2} \hat{\boldsymbol{\Sigma}} \right] = 0 \\ \\
\hat{\mu} = \frac{1}{n} \sum_{k = 1}^{n} \boldsymbol{x}_{k} \\ \\
\hat{\Sigma} = \frac{1}{n} \sum_{k = 1}^{n} (\boldsymbol{x}_{k} - \hat{\mu}) (\boldsymbol{x}_{k} - \hat{\mu})^\mathrm{T}
\end{gather*}
$$
估计的偏差
$$
\begin{gather*}
\tilde{\mu} = \frac{1}{n} \sum_{k = 1}^{n} \boldsymbol{x}_{k} = \hat{\mu} \\ \\
\tilde{\Sigma} = \frac{1}{n - 1} \sum_{k = 1}^{n} (\boldsymbol{x}_{k} - \hat{\mu}) (\boldsymbol{x}_{k} - \hat{\mu})^\mathrm{T}
= \frac{n}{n - 1} \hat{\Sigma} \ne \hat{\Sigma} \\ \\
\hat{\Sigma} \overset{n \to \infty}{\Longrightarrow} \tilde{\Sigma}
\end{gather*}
$$
对于协方差矩阵而言这是一种渐进式估计，在样本足够多的情况下近似为**无偏估计**

# EM 算法

可见随机变量$x$和隐藏随机变量$z$满足分布
$$
p(x,\ z \mid \theta)
$$
通过对可见变量$x$的观测和最大似然估计可知
$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} \prod_{i = 1}^{n} p(x_{i} \mid \theta) \\ \\
&= \argmax_{\theta} \prod_{i = 1}^{n} \int_{z} p(x_{i},\ z \mid \theta) dz \\ \\
&= \argmax_{\theta} \sum_{i = 1}^{n} \ln \int_{z} p(x_{i},\ z \mid \theta) dz \\ \\
&= \argmax_{\theta} \sum_{i = 1}^{n} \ln \int_{z} q_{i}(z) \frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} dz
\end{align*}
$$
其中，$q_{i}(z)$是$z$的任意一个概率分布

### Jasen 不等式

凸函数$f(x)$满足性质
$$
f(\lambda x_{1} + (1 - \lambda) x_{2}) \ge \lambda f(x_{1}) + (1 - \lambda) f(x_{2}),\quad \lambda \in [0,\ 1]
$$
利用数学归纳法将以上性质进行拓展
$$
\begin{align*}
f(\sum_{i = 1}^{n} \lambda_{i} x_{i}) &= f(\lambda_{1} x_{1} + (1 - \lambda_{1}) \sum_{i = 2}^{n} \frac{\lambda_{i}}{1 - \lambda_{1}} x_{i}) \\ \\
&\ge \lambda_{1} f(x_{1}) + (1 - \lambda_{1}) f(\sum_{i = 2}^{n} \frac{\lambda_{i}}{1 - \lambda_{1}} x_{i}) \\ \\
&\ge \lambda_{1} f(x_{1}) + (1 - \lambda_{1}) \left[ \sum_{i = 2}^{n} \frac{\lambda_{i}}{1 - \lambda_{1}} f(x_{i}) \right] \\ \\
&= \sum_{i = 1}^{n} \lambda_{i} f(x_{i}) \\ \\
s.t.\quad &\sum_{i = 1}^{n} \lambda_{i} = 1,\quad \lambda_{i} \in [0,\ 1]
\end{align*}
$$
假设随机变量$x$服从分布$p(x)$，根据凸函数的上述性质可得 **$Jasen$不等式**
$$
\begin{gather*}
\int_{-\infty}^{+\infty} p(x) dx = 1,\quad p(x) \ge 0 \\
\Downarrow \\
f(\int_{-\infty}^{+\infty} x p(x) dx) \ge \int_{-\infty}^{+\infty} f(x) p(x) dx \\
\Downarrow \\
f(\mathcal{E}_{x} x) \ge \mathcal{E}_{x} f(x)
\end{gather*}
$$
利用$Jasen$不等式将对数似然函数化为
$$
\ell(\theta) = \sum_{i = 1}^{n} \ln \int_{z} q_{i}(z) \frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} dz 
\ge \sum_{i = 1}^{n} \int_{z} q_{i}(z) \ln \left[ \frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} \right] dz
$$
当$q_{i}(z) = p(z \mid x_{i},\ \theta)$时不等式取得等号
$$
\begin{gather*}
\frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} = \frac{p(x_{i},\ z \mid \theta)}{p(z \mid x_{i},\ \theta)} = p(x,\ \theta) \\ \\
\ell(\theta) = \sum_{i = 1}^{n} \ln \int_{z} q_{i}(z) \frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} dz
= \sum_{i = 1}^{n} \int_{z} q_{i}(z) \ln \left[ \frac{p(x_{i},\ z \mid \theta)}{q_{i}(z)} \right] dz = \sum_{i = 1}^{n} \ln p(x_{i},\ \theta)
\end{gather*}
$$
令$q_{i}(z) = p(z \mid x_{i},\ \tilde{\theta})$，其中$\tilde{\theta}$是参数$\theta$的任一取值，可得
$$
\ell(\theta) \ge \sum_{i = 1}^{n} \mathcal{E}_{q_{i}(z)} \ln \left[ \frac{p(x_{i},\ z \mid \theta)}{p(z \mid x_{i},\ \tilde{\theta})} \right]
$$
当$\theta = \tilde{\theta}$时上式取得等号，即$\mathcal{E}_{q(z)} Q(\theta,\ \tilde{\theta})$是$\ell(\theta)$的紧下界，为了找到比$\ell(\tilde{\theta})$更大的似然函数值，令
$$
\begin{gather*}
\begin{align*}
\tilde{\theta} &\gets \argmax_{\theta} \sum_{i = 1}^{n} \mathcal{E}_{q_{i}(z)} \ln 
\left[ \frac{p(x_{i},\ z \mid \theta)}{p(z \mid x_{i},\ \tilde{\theta})} \right] \\ \\
&= \argmax_{\theta} \sum_{i = 1}^{n} \mathcal{E}_{q_{i}(z)} \ln p(x_{i},\ z \mid \theta) \\ \\
&= \argmax_{\theta} \sum_{i = 1}^{n} \mathcal{E}_{q_{i}(z)} Q_{i}(\theta,\ \tilde{\theta})
= \argmax_{\theta} Q(\theta,\ \tilde{\theta})
\end{align*} \\ \\
q_{i}(z) = p(z \mid x_{i},\ \tilde{\theta}) = \frac{p(z,\ x_{i} \mid \tilde{\theta})}{p(x_{i} \mid \tilde{\theta})},\quad
Q_{i}(\theta,\ \tilde{\theta}) = \ln p(x_{i},\ z \mid \theta)
\end{gather*}
$$
反复进行上述的优化方法即可不断找到更大的似然函数值直至算法收敛

# 贝叶斯估计

**先验知识**；假设分布中的未知参数$\theta$是随机变量并且服从某个特定的分布
$$
p(\theta)
$$
**证据因子**；观测到样本集$D$的概率（不依赖参数$\theta$）
$$
p(D) = \int_{\Theta} p(D \mid \theta) p(\theta) d\theta
$$
**后验分布**；在观测到样本集$D$时未知参数$\theta$的概率密度
$$
p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)}
$$
贝叶斯估计假设未见样本$\boldsymbol{x}$与样本集$D$的观测对于$\theta$条件独立
$$
p(\boldsymbol{x},\ D \mid \theta) = p(\boldsymbol{x} \mid \theta) p(D \mid \theta)
$$
将上式变形得到
$$
\frac{p(\boldsymbol{x},\ D,\ \theta)}{p(\theta)} = \frac{p(\boldsymbol{x},\ \theta)}{p(\theta)} \frac{p(D,\ \theta)}{p(\theta)} \Rightarrow \frac{p(\boldsymbol{x},\ D,\ \theta)}{p(D,\ \theta)} = \frac{p(\boldsymbol{x},\ \theta)}{p(\theta)}
$$
即
$$
p(\boldsymbol{x} \mid \theta,\ D) = p(\boldsymbol{x} \mid \theta)
$$
**未见样本分布**；通过已观测到的样本集以及参数先验知识的基础上观测一个未见样本$\boldsymbol{x}$的概率密度
$$
\begin{align*}
	p(\boldsymbol{x} \mid D) &= \int_{\Theta} p(\boldsymbol{x},\ \theta \mid D) d\theta \\ \\
	&= \int_{\Theta} p(\boldsymbol{x} \mid \theta,\ D) p(\theta \mid D) d\theta \\ \\
	&= \int_{\Theta} p(\boldsymbol{x} \mid \theta) p(\theta \mid D) d\theta
\end{align*}
$$

### 正态分布参数的贝叶斯估计

* 单变量（$\sigma^2$已知）

先验知识 &rArr; $\mu \sim N(\mu_{0},\ \sigma_{0}^2)$
$$
p(\mu) = \frac{1}{\sqrt{2\pi} \sigma_{0}} \exp\left[ -\frac{1}{2} \frac{(\mu - \mu_{0})^2}{\sigma_{0}^2} \right]
$$
似然函数
$$
p(D \mid \mu) = \prod_{k = 1}^{n} \frac{1}{\sqrt{2\pi} \sigma} \exp\left[ -\frac{1}{2} \frac{(x - \mu)^2}{\sigma^2} \right]
$$
后验分布
$$
\begin{align*}
p(\mu \mid D) &= \frac{1}{p(D)} p(D \mid \mu) p(\mu) \\ \\
&= \frac{1}{p(D)} \prod_{k = 1}^{n} \frac{1}{\sqrt{2\pi} \sigma} 
\exp\left[ -\frac{1}{2} \frac{(x_{k} - \mu)^2}{\sigma^2} \right]
\frac{1}{\sqrt{2\pi} \sigma_{0}} 
\exp\left[ -\frac{1}{2} \frac{(\mu - \mu_{0})^2}{\sigma_{0}^2} \right] \\ \\
&= \alpha \exp\left[ -\frac{1}{2} \sum_{k = 1}^{n} \frac{(x_{k} - \mu)^2}{\sigma^2} -
\frac{1}{2} \frac{(\mu - \mu_{0})^2}{\sigma_{0}^2} \right] \\ \\
&= \alpha \exp\left[ (-\frac{n}{2\sigma^2} - \frac{1}{2\sigma_{0}^2}) \mu^2 +
(\frac{n\bar{x}}{\sigma^2} + \frac{\mu_{0}}{\sigma_{0}^2}) \mu + \cdots \right] \\ \\
&= \alpha \exp\left[ (-\frac{n}{2\sigma^2} - \frac{1}{2\sigma_{0}^2})
(\mu - \frac{n\bar{x} \sigma_{0}^2 + \mu_{0} \sigma^2}{n\sigma_{0}^2 + \sigma^2})^2 + \cdots \right] \\ \\
&= {\alpha}' \exp\left[ -\frac{1}{2} (\mu - \frac{n\bar{x} \sigma_{0}^2 + \mu_{0} \sigma^2}{n\sigma_{0}^2 + \sigma^2})^2 \bigg/
\frac{\sigma^2 \sigma_{0}^2}{n\sigma_{0}^2 + \sigma^2} \right]
\end{align*}
$$
经过样本集修正后的后验分布依然是一个正态分布
$$
\begin{gather*}
p(\mu \mid D) \Rightarrow N(\mu_{D},\ \sigma_{D}^2) \\ \\
\frac{1}{\sigma_{D}^2} \mu_{D} = \frac{n}{\sigma^2} \bar{x} + \frac{1}{\sigma_{0}^2} \mu_{0} \\ \\
\frac{1}{\sigma_{D}^2} = \frac{n}{\sigma^2} + \frac{1}{\sigma_{0}^2}
\end{gather*}
$$
考虑以下三种情况
* $n \to \infty$（观测足够多的样本时，贝叶斯估计与最大似然估计的结果一致）
$$
\begin{gather*}
\mu_{D} \to \bar{x} \\ \\
\sigma_{D}^2 \to 0 \\ \\
\Longrightarrow p(\mu = \bar{x} \mid D) \to 1
\end{gather*}
$$
* $\sigma_{0}^2 \ll \sigma^2$（先验知识非常强，后验分布与先验知识一致）
$$
\begin{gather*}
\mu_{D} \to \mu_{0} \\ \\
\sigma_{D}^2 \to \sigma_{0}^2 \\ \\
\Longrightarrow p(\mu \mid D) \to p(\mu)
\end{gather*}
$$
* $\sigma_{0}^2 \gg \sigma^2$（先验知识非常弱，贝叶斯估计退化为最大似然估计）
$$
\begin{gather*}
\mu_{D} \to \bar{x} \\ \\
\sigma_{D}^2 \to \frac{\sigma_{0}^2}{n} \\ \\
\Longrightarrow p(\mu \mid D) \to N(\bar{x},\ \frac{\sigma_{0}^2}{n})
\end{gather*}
$$

未见样本概率密度
$$
\begin{align*}
p(x \mid D) &= \int_{-\infty}^{+\infty} p(x \mid \mu) p(\mu \mid D) d\mu \\ \\
&= \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi} \sigma} \exp\left[ -\frac{1}{2} \frac{(x - \mu)^2}{\sigma^2} \right]
\frac{1}{\sqrt{2\pi} \sigma_{D}} \exp\left[ -\frac{1}{2}
\frac{(\mu - \mu_{D})^2}{\sigma_{D}^2} \right] d\mu \\ \\
&= \frac{1}{2\pi \sigma \sigma_{D}} \int_{-\infty}^{+\infty}
\exp\left[ -\frac{1}{2} \left[ \frac{(x - \mu)^2}{\sigma^2} +
\frac{(\mu - \mu_{D})^2}{\sigma_{D}^2} \right] \right] d\mu \\ \\
&= \frac{1}{2\pi \sigma \sigma_{D}} \int_{-\infty}^{+\infty}
\exp\left[ -\frac{1}{2} \left[ (\frac{1}{\sigma^2} + \frac{1}{\sigma_{D}^2})\mu^2 -
(\frac{2x}{\sigma^2} + \frac{2\mu_{D}}{\sigma_{D}^2})\mu +
(\frac{x^2}{\sigma^2} + \frac{\mu_{D}^2}{\sigma_{D}^2}) \right] \right] d\mu \\ \\
&= \frac{1}{2\pi \sigma \sigma_{D}} \int_{-\infty}^{+\infty}
\exp\left[ -\frac{1}{2} (\frac{1}{\sigma^2} + \frac{1}{\sigma_{D}^2})
(\mu - \frac{\sigma_{D}^2 x + \sigma^2 \mu_{D}}{\sigma^2 + \sigma_{D}^2})^2 -
\frac{1}{2} \frac{(x - \mu_{D})^2}{\sigma^2 + \sigma_{D}^2} \right] d\mu \\ \\
&= \exp\left[ -\frac{1}{2} \frac{(x - \mu_{D})^2}{\sigma^2 + \sigma_{D}^2} \right]
\frac{1}{2\pi \sigma \sigma_{D}} \int_{-\infty}^{+\infty}
\exp\left[ -\frac{1}{2} (\frac{1}{\sigma^2} + \frac{1}{\sigma_{D}^2})
(\mu - \frac{\sigma_{D}^2 x + \sigma^2 \mu_{D}}{\sigma^2 + \sigma_{D}^2})^2\right] d\mu \\ \\
&= \frac{1}{\sqrt{2\pi} \sqrt{\sigma^2 + \sigma_{D}^2}}
\exp\left[ -\frac{1}{2} \frac{(x - \mu_{D})^2}{\sigma^2 + \sigma_{D}^2} \right]
\sim N(\mu_{D},\ \sigma^2 + \sigma_{D}^2)
\end{align*}
$$
从效果来看$\mu$被看作$\mu_{D}$，而$\mu$的不确定性$\sigma_{D}^2$传递到未见样本分布上

* 多变量情况（$\boldsymbol{\Sigma}$已知）

类似于单变量，此处暂时省略推导过程，直接给出后验概率和未见样本分布
$$
\begin{gather*}
p(\mu \mid D) \sim N(\mu_{D},\ \boldsymbol{\Sigma_{D}}) \\ \\
\boldsymbol{\Sigma_{D}}^{-1} = n\boldsymbol{\Sigma}^{-1} + \boldsymbol{\Sigma_{0}}^{-1} \\ \\
\boldsymbol{\Sigma_{D}}^{-1} \mu = n\boldsymbol{\Sigma}^{-1}\bar{\boldsymbol{x}} + \boldsymbol{\Sigma_{0}}^{-1} \mu_{0} \\ \\
p(\boldsymbol{x} \mid D) \sim N(\mu_{D},\ \boldsymbol{\Sigma} + \boldsymbol{\Sigma_{D}})
\end{gather*}
$$

### 贝叶斯增量式学习（递归学习）

样本个数为$n$时的样本集
$$
D^n = \left \{ \boldsymbol{x}_{1},\ \boldsymbol{x}_{2},\ \cdots,\ \boldsymbol{x}_{n} \right\}
$$
考虑到样本集的样本条件独立地被观测
$$
p(D^n \mid \theta) = p(\mathcal{x}_{n} \mid \theta) p(D^{n - 1} \mid \theta)
$$
被新增样本$\boldsymbol{x}_{n}$更新后的后验概率
$$
\begin{align*}
p(\theta \mid D^n) &= \frac{p(D^n \mid \theta) p(\theta)}{p(D^n)} \\ \\
&= \frac{p(\mathcal{x}_{n} \mid \theta) p(D^{n - 1} \mid \theta) p(\theta)}{p(D^n)} \\ \\
&= \frac{p(D^{n - 1})}{p(D^{n})} p(\mathcal{x}_{n} \mid \theta) p(\theta \mid D^{n - 1}) \\ \\
&\propto p(\mathcal{x}_{n} \mid \theta) p(\theta \mid D^{n - 1})
\end{align*}
$$
其中
$$
p(\theta \mid D^{0}) = p(\theta)
$$
代表未经过样本更新的参数分布，即原始的先验知识