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

Sparse Viewpoint Model 在 ICG 的前作 SRT3D 中提出，这是为了避免在 tracking 过程中渲染 3D 模型。

在生成过程中，物体由大量虚拟摄像机渲染而成，这些虚拟摄像机放置在围绕物体的测地线网格的顶点上。图像坐标在渲染轮廓的轮廓和表面上随机采样，然后对于每个坐标，重建 3D 点 ${}_\textrm{M}\pmb{X}$ 和 3D 法向量 ${}_\textrm{M}\pmb{N}$。这些向量以及从摄像机指向模型中心的方位角，与每个视点的方位角一起存储。给定姿态估计，获取轮廓和表面表示就简化为搜索最接近的方位角向量。