# Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects

> https://arxiv.org/abs/2203.05334

## 概览

**ICG（Iterative Corresponding Geometry）** 是一种面向**无纹理物体**的高效 3D 目标跟踪方法：它把**区域信息**（轮廓附近的 correspondence lines）和**深度信息**（表面上的 correspondence points）融合在一个概率模型中，用正则化牛顿法迭代优化 6DoF 位姿，仅需物体几何模型，不需要纹理。论文报告在 YCB-Video、OPT、Choi 等数据集上达到/超过当时 SOTA，且单 CPU 核心约 **1.3 ms/帧（788.4 Hz）**。


## 论文要解决的问题

6DoF 物体跟踪需要从连续图像中估计物体的位置和姿态。现有方法大致分为：

- 关键点/边缘/直接优化：依赖纹理或边缘，无纹理物体效果差；
- 区域方法：利用颜色统计区分前景/背景，适合无纹理物体，但效率和深度信息利用有限；
- 深度/ICP 方法：依赖深度质量；
- 深度学习方法：精度高但计算资源大、实时性有限。

本文的出发点是：**区域 + 深度**对无纹理物体跟踪很有潜力，但近年来缺少新的高效融合方法。因此提出 ICG，把区域对应线和深度对应点统一到概率框架中。


## Probabilistic Model

### Preliminaries

记

- 模型的 3D 坐标：

    $$
    \pmb{X} = \begin{bmatrix} X& Y& Z\end{bmatrix}^\top\in \mathbb{R}^3, \\
    \pmb{\widetilde{X}} = \begin{bmatrix} X& Y& Z& 1\end{bmatrix}^\top \in \mathbb{R}^4.
    $$

- 图像坐标： $\pmb{x} = \begin{bmatrix} x& y\end{bmatrix}^\top \in \mathbb{R}^2$；

- 颜色 $\pmb{y} = \pmb{I}_\textrm{c}(\pmb{x})$；

- 深度 $d_Z = I_\textrm{d}(\pmb{x})$。

- 针孔相机投影：

    $$
    \begin{equation*}
        \pmb{x} = \pmb{\pi}(\pmb{X}) = 
        \begin{bmatrix}
            \dfrac{X}{Z}f_x + p_x,  \dfrac{Y}{Z}f_y + p_y
        \end{bmatrix}^\top,
    \end{equation*}
    $$

    其中，$f_{x/y}$ 表示焦距，$p_{x/y}$ 表示主点坐标，假设图像已经去畸变。

然后

- 用 ${}_A\pmb{T}_B{}\in \mathbb{SE}(3)$ 描述从 B 到 A 的坐标齐次变换：

    $$
    \begin{equation*}
        _\textrm{A}\pmb{\widetilde{X}} = {}_\textrm{A}\pmb{T}_\textrm{B}{}_\textrm{B}\pmb{\widetilde{X}} =
        \begin{bmatrix}
            _\textrm{A}\pmb{R}_\textrm{B} & _\textrm{A}\pmb{t}_\textrm{B} \\ \pmb{0} & 1
        \end{bmatrix}
        {}_\textrm{B}\pmb{\widetilde{X}},
    \end{equation*}
    $$

    回顾一下，式中的矩阵是一个齐次变换矩阵，$_\textrm{A}\pmb{R}_\textrm{B} \in \mathbb{SO}(3)$ 代表旋转，$_\textrm{A}\pmb{t}_\textrm{B} \in \mathbb{R}^3$ 代表平移。

- 对于模型坐标系 $\textrm{M}$ 上的微小扰动，我们用如下的方法来估计：

    $$
    \begin{equation*}
        _\textrm{M}\pmb{\widetilde{X}}(\pmb{\theta}) =
        {}_\textrm{M}\pmb{T}(\pmb{\theta}){}_\textrm{M}\pmb{\widetilde{X}} =
        \begin{bmatrix}
            \pmb{I} + [\pmb{\theta}_\textrm{r}]_\times & \pmb{\theta}_\textrm{t} \\ \pmb{0} & 1
        \end{bmatrix}
        {}_\textrm{M}\pmb{\widetilde{X}},
    \end{equation*}
    $$

    其中，$[\pmb{\theta}_\textrm{r}]_\times$ 是一个反对称矩阵，例如对于 $\pmb{\theta}_\textrm{r} = \begin{bmatrix}x & y & z\end{bmatrix}^\top$，有
    $$
    [\pmb{\theta}_r]_\times = \begin{bmatrix} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \end{bmatrix}.
    $$

    根据 Rodrigues' rotation formula，一个精确的旋转矩阵应该是 $\exp([\pmb{\theta}_r]_\times)$，在微小扰动时，我们对齐做一阶线性化，得到 $\pmb I + [\pmb{\theta}_r]_\times$。

### Sparse Viewpoint Model

Sparse Viewpoint Model 在 ICG 的前作 SRT3D 中提出。为了避免在 tracking 过程中渲染 3D 模型，在生成过程中，物体由大量虚拟摄像机渲染而成，这些虚拟摄像机放置在围绕物体的测地线网格的顶点上。图像坐标在渲染轮廓的轮廓和表面上随机采样，然后对于每个坐标，重建 3D 点 ${}_\textrm{M}\pmb{X}$ 和 3D 法向量 ${}_\textrm{M}\pmb{N}$。这些向量以及从摄像机指向模型中心的方位角，与每个视点的方位角一起存储。给定姿态估计，获取轮廓和表面表示就简化为搜索最接近的方位角向量。

3D 点和法向量被用于后续的深度匹配与区域轮廓对齐过程中。

### Region Modality

由上一节，我们得到了若干 3D 点和 3D 法向量，对于所有在轮廓（contour）上的点，把他们投影到 2D 图像上，建立对应线（correspondence line）。对应线可以理解为一个函数 $\pmb l$，给出一个沿对应线的线坐标 $r\in \mathbb R$，返回一个颜色 $\pmb y = \pmb l(r)$。建立对应线的过程如下：

1. 对于每个 ${}_\textrm{M}\pmb{X}$ 和 ${}_\textrm{M}\pmb{N}$，投影后，得到中心点 $\pmb{c} \in \mathbb{R}^2$，与 2D 法向量 $\pmb{n} \in \mathbb{R}^2$，沿着 $\pmb n$ 穿过 $\pmb c$ 画一条线，这就是一条轮廓线。

1. 为了计算高效，将像素进行离散化。具体地，引入尺度空间坐标，设一个超参数 $s$ 作为尺度，将每条线按尺度 $s$ 切成一个个 segment，并将坐标系平移 $\Delta r$，抵消中心点 $\pmb c$ 的亚像素偏差，使得采样边界对齐到真实的像素网格上，这么做可以让对应线不受其方向和亚像素位置的影响。

记原线坐标为 $r$，离散化后的现线坐标为 $r_\textrm{s}$。

> 前置知识：颜色直方图
> 
> 为了区分前景（物体） $m_f$ 和背景 $m_b$，算法不看像素的位置，只看像素的颜色。它事先统计前景的颜色分布，得到一个概率 $p(\pmb{y}\mid m_\textrm{f})$，同理也统计得到 $p(\pmb{y}\mid m_\textrm{b})$。举个例子，一个物体是红色的，背景是白色的，那么对于一个红色像素，$p(\red{\pmb{red}}\mid m_\textrm{f})$ 就很高，$p(\red{\pmb{red}}\mid m_\textrm{b})$ 就很低。

接着，我们用归一化颜色直方图来区分前景和背景，得到 $p(\pmb{y}\mid m_\textrm{f})$ 和 $p(\pmb{y}\mid m_\textrm{b})$。然后我们来计算一个 segment 的概率：一个 segment 包含好几个像素，假设它们相互独立，那么有

$$
p'(r_{\textrm{s}i}) = \prod\limits_{\pmb{y} \in \pmb{s}} p(\pmb{y}\mid m_i), \quad i\in \{\textrm{f}, \textrm{b}\},
$$

为了让这个累乘变成一个标准的概率，将它归一化到 $[0, 1]$ 的区间：

$$
\begin{equation*}
	p_{\textrm{s}i}(r_\textrm{s}) = \frac{\prod\limits_{\pmb{y} \in \pmb{s}} p(\pmb{y}\mid m_i)} {\prod\limits_{\pmb{y} \in  \pmb{s}} p(\pmb{y} \mid  m_\textrm{f}) + \prod\limits_{\pmb{y} \in  \pmb{s}} p(\pmb{y} \mid  m_\textrm{b})} , \quad i\in\{\textrm{f}, \textrm{b}\},
\end{equation*}
$$

即坐标 $r_s$ 所处的 segment，属于前景或背景的后验概率。

算完颜色后验概率，算法其实只知道了“每个段是什么颜色”。但位姿优化需要知道“边界偏了多少”，因此，我们引入两个阶跃函数：

$$
\begin{align*}
	h_\textrm{f}(x) &= \frac{1}{2} - \alpha_\textrm{h}\tanh\bigg(\frac{x}{2s_\textrm{h}}\bigg),\\
	h_\textrm{b}(x) &= \frac{1}{2} + \alpha_\textrm{h}\tanh\bigg(\frac{x}{2s_\textrm{h}}\bigg).
\end{align*}
$$

式中 $\alpha_\textrm{h} \in [0, 0.5]$ 控制幅度，$s_\textrm{h}\in \mathbb R^+$ 控制幅度（平滑过渡的宽度）。

引入阶跃函数是为了建立一个理想边界的模板：如果位姿完全准确，理论上每个像素应该以这个方式来分布。

将 $_\textrm{M}\pmb{\widetilde{X}}$ 施加微小扰动，然后转到相机坐标系，得到

$$
\begin{equation*}
	{}_\textrm{C}\pmb{\widetilde{X}}(\pmb{\theta}) = {}_\textrm{C}\pmb{T}_\textrm{M}{}_\textrm{M}\pmb{T}(\pmb{\theta}){}_\textrm{M}\pmb{\widetilde{X}},
\end{equation*}
$$

算出 ${}_\textrm{C}\pmb{{X}}(\pmb{\theta})$ 在对应线上的偏移量：

$$
\begin{equation*}
	d_\textrm{s}(\pmb{\theta}) = \bigg(\pmb{n}^\top\Big(\pmb{\pi}\big({}_\textrm{C}\pmb{X}(\pmb{\theta})\big) - \pmb{c}\Big)-\Delta r\bigg) \frac{\bar{n}}{s},
\end{equation*}
$$

式中的 $\bar{n} = \lVert\pmb{n}\rVert_\mathrm{max}$ 表示将 $\pmb n$ 投影到最接近的水平或垂直图像坐标上，这是为了在方向上作修正。其余记号在上文都定义过。

$d_\textrm{s}(\pmb{\theta})$ 表示：如果位姿偏移了 $\pmb \theta$，投影点会在尺度坐标 $r_s$ 上偏移多少。

接下来我们算 PDF（Probability Density Function，概率密度函数）。在 SRT3D 中，有：

$$
\begin{equation*}
	p(d_\textrm{s}(\pmb{\theta})\mid \omega_\textrm{s},\pmb{l}) \propto \prod_{r_\textrm{s}\in\omega_\textrm{s}}
	\sum_{i \in \{\textrm{f}, \textrm{b}\}}
	h_i\big(r_\textrm{s}-d_\textrm{s}(\pmb{\theta})\big)p_{\textrm{s}i}(r_\textrm{s}),
\end{equation*}
$$

式中，$\omega_\textrm s$ 表示所考虑的线域，即所选的范围集合。

左式意思是，在已知对应线 $\pmb l$ 和其有效范围 $\omega_\textrm s$ 的条件下，当前假设的偏移量 $d_\textrm{s}(\pmb \theta)$ 发生的概率；右式在求，假设真实的偏移量就是 $d_\textrm{s}(\pmb \theta)$，那么我们刚好观察到 $\omega_\textrm s$ 下这组颜色 $\pmb l$ 的概率有多大。左式正比于右式。

通过已经有的证明，这个 PDF 的方差为上述阶跃函数 $h_\textrm{b/h}$ 的参数 $s_\textrm h$，不受我们控制。我们希望对应线独立于尺度和斜率参数，因此，引入用户自定义的标准差 $\sigma_\textrm{r}$，公式中的 PDF 被缩放如下：

$$
\begin{equation*}
p(\pmb{\theta}\mid\omega_{\textrm{s}},\pmb{l}) \propto p(d_{\textrm{s}}(\pmb{\theta})\mid\omega_{\textrm{s}},\pmb{l})^{\left(\dfrac{s_\textrm{h}s^2}{\sigma_\textrm{r}^2\bar{n}^2}\right)}.
\end{equation*}
$$

这也就是我们最终的区域 PDF 公式。

### Depth Modality

对于概率模型，我们构建了一个使用点到面误差度量的正态分布。

找一个 3D 表面点 $\pmb X$，投影到深度图上。在这个投影点周围画一个矩形区域（由用户定义的半径和步长决定），在区域内重建出真实的 3D 点，找到距离 $\pmb X$ 最近的那个，作为对应点 $\pmb P$。如果距离太远（超过阈值），就视作没找到，直接丢弃。

在找到 $\pmb P$ 后，计算点到面距离，具体地，计算 $\pmb N^\top(\pmb X - \pmb P)$，这么做可以忽略表面滑动带来的虚假误差。

然后，对于已知的 $\pmb P$ 和一个位姿变化假设的向量 $\pmb \theta$，其概率：

$$
\begin{equation}
	p(\pmb{\theta}\mid \pmb{P}) \propto \exp\bigg(\hspace{-0.1cm}-\frac{1}{2 d_\textrm{Z}^2 \sigma_\textrm{d}^2 }\Big({}_\textrm{M}\pmb{N}^\top\big({}_\textrm{M}\pmb{X} - {}_\textrm{M}\pmb{P}(\pmb{\theta})\big)\Big)^{\hspace{-0.05cm}2}\bigg)
\end{equation}
$$

