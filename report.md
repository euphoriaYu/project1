 # 全景图拼接实验报告

## 摘要

本实验实现了基于计算机视觉技术的全景图拼接，包括Harris角点检测、特征描述（SIFT和HOG）、特征匹配、RANSAC算法以及图像拼接等关键技术。通过手动实现Harris角点检测算法，并结合SIFT和HOG特征描述子，我们对两组不同的图像进行了特征匹配和图像拼接，并对比了不同特征描述子的性能差异。实验结果表明，SIFT特征在特征描述和匹配方面表现出较好的稳定性和精确度，并且基于SIFT特征的图像拼接效果更为理想。本实验加深了对计算机视觉中特征检测与匹配技术的理解，为全景图像拼接等实际应用奠定了基础。

## 实验目的

本次实验旨在通过实践了解和掌握计算机视觉中的关键技术，具体目的如下：

1. 深入理解Harris角点检测器的原理，并手动实现该算法
2. 掌握RANSAC（随机抽样一致性）算法在特征匹配中的应用
3. 理解HOG（方向梯度直方图）特征描述子的基本原理
4. 比较SIFT和HOG特征描述子在图像匹配中的性能差异
5. 实现基于特征点的图像拼接技术，并应用于全景图像的构建

## 实验内容

本次实验主要包含以下三个任务：

### 任务1：Harris角点检测

- 手动实现Harris角点检测算法
- 对sudoku.png图像进行角点检测
- 将检测结果保存为sudoku_keypoints.png

### 任务2：特征描述与匹配

- 使用Harris算法检测uttower1.jpg和uttower2.jpg中的角点
- 分别使用SIFT和HOG特征描述子获取角点特征
- 使用欧几里得距离计算特征相似度并进行匹配
- 使用RANSAC算法筛选出内点并计算变换矩阵
- 基于变换矩阵进行图像拼接
- 比较SIFT和HOG特征在匹配和拼接过程中的差异

### 任务3：全景图拼接

- 基于SIFT特征和RANSAC算法对约塞米蒂系列图像进行全景拼接
- 将拼接结果保存为yosemite_stitching.png

## 数学原理

### Harris角点检测

Harris角点检测算法是一种经典的特征点检测方法，能够有效地检测图像中的角点。其基本原理是通过分析图像局部区域在各个方向上的灰度变化，判断是否为角点。

#### 基本原理

考虑图像中的一个小窗口，计算窗口平移后的灰度差平方和。对于图像点(x,y)，窗口平移(u,v)后的灰度差平方和为：

$$E(u,v) = \sum_{x,y} w(x,y)[I(x+u, y+v) - I(x,y)]^2$$

其中w(x,y)为窗口函数（通常为高斯窗口），I(x,y)为图像在点(x,y)处的灰度值。

使用泰勒展开可以得到：

$$E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} \mathbf{M} \begin{bmatrix} u \\ v \end{bmatrix}$$

其中M为结构张量：

$$\mathbf{M} = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$


其中Ix和Iy分别为图像在x和y方向的梯度。

#### 角点响应函数

Harris和Stephens提出的角点响应函数为：

$$R = \det(\mathbf{M}) - k \cdot \text{trace}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

其中λ₁和λ₂是结构张量M的特征值，k是经验常数（通常取0.04-0.06）。

角点判断准则：

- 如果λ₁和λ₂都很小，则区域为平坦区域
- 如果一个大一个小，则为边缘
- 如果两者都很大，则为角点

### 特征描述子

#### SIFT (Scale-Invariant Feature Transform)

SIFT特征是一种对尺度、旋转和光照变化具有不变性的局部特征。其描述子构建过程如下：

1. 在检测到的关键点周围选取一个16×16的窗口
2. 将窗口分为4×4的子区域
3. 在每个子区域中计算8方向的梯度直方图
4. 形成4×4×8=128维的特征向量
5. 对特征向量进行归一化，以提高对光照变化的鲁棒性

#### HOG (Histogram of Oriented Gradients)

HOG特征通过计算和统计图像局部区域的梯度方向直方图来表征图像特征。其计算过程包括：

1. 计算图像的梯度（幅值和方向）
2. 将图像分为若干个单元格（cells）
3. 在每个单元格中构建梯度方向直方图
4. 将相邻单元格组合成块（blocks）
5. 对每个块中的直方图进行归一化
6. 将所有归一化后的直方图拼接形成最终的特征向量

### RANSAC算法

RANSAC（随机抽样一致性）是一种用于从包含大量异常值的数据中估计数学模型参数的迭代方法。在图像拼接中，它用于筛选正确的特征匹配对（内点）并估计变换矩阵。

#### 基本流程

1. 从数据集中随机选择最小样本集（对于单应性矩阵，需要至少4对匹配点）
2. 使用最小样本集计算模型参数（如单应性矩阵）
3. 统计在该模型下符合条件的数据点（内点）
4. 重复以上步骤多次，选择内点最多的模型
5. 使用所有内点重新估计模型参数，得到最终模型

#### 单应性矩阵

对于两幅图像之间的平面投影变换，可以用单应性矩阵（Homography Matrix）H表示：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

其中(x,y)和(x',y')分别是两幅图像中的对应点。单应性矩阵有8个自由度，因此至少需要4对点来求解。

## 关键代码实现

### Harris角点检测实现

以下是Harris角点检测算法的核心实现代码：

```python
def harris_corner_detector(image, k=0.04, threshold_ratio=0.01, nms_window_size=5):
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算x和y方向的梯度
    dx = cv2.Sobel(np.float32(gray), cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(np.float32(gray), cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度乘积
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy
    
    # 使用高斯滤波平滑梯度乘积
    Ixx = cv2.GaussianBlur(Ixx, (5, 5), 0)
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)
    Iyy = cv2.GaussianBlur(Iyy, (5, 5), 0)
    
    # 计算Harris响应函数
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    harris_response = det - k * (trace ** 2)
    
    # 设置阈值
    max_response = np.max(harris_response)
    threshold = threshold_ratio * max_response
    
    # 非最大抑制 (NMS)
    corners = []
    marked_image = image.copy()
    
    height, width = gray.shape
    offset = nms_window_size // 2
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # 只考虑高于阈值的像素
            if harris_response[y, x] > threshold:
                # 检查是否是局部最大值
                window = harris_response[y-offset:y+offset+1, x-offset:x+offset+1]
                if harris_response[y, x] == np.max(window):
                    corners.append((x, y))
                    # 在原图上标记角点
                    cv2.circle(marked_image, (x, y), 3, (0, 0, 255), -1)
    
    return corners, marked_image
```

### 特征描述子实现

下面展示SIFT特征描述子计算的实现代码：

```python
def compute_sift_descriptors(image, keypoints):
    # 将自定义关键点转换为OpenCV关键点格式
    kps = [cv2.KeyPoint(x=x, y=y, size=10) for x, y in keypoints]
    
    # 创建SIFT描述子提取器
    sift = cv2.SIFT_create()
    
    # 计算描述子
    _, descriptors = sift.compute(image, kps)
    
    return descriptors, kps
```

### 特征匹配与RANSAC实现

特征匹配使用欧几里得距离和比率测试：

```python
def match_features(desc1, desc2, ratio_threshold=0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # 对每个描述子，找到最近的两个匹配
    matches = []
    for i in range(len(desc1)):
        # 计算当前描述子与所有描述子的欧氏距离
        distances = np.sqrt(np.sum((desc2 - desc1[i])**2, axis=1))
        
        # 找到最近和次近邻
        idx = np.argsort(distances)
        
        if len(idx) > 1:  # 至少需要两个点来计算比率
            # 比率测试
            if distances[idx[0]] < ratio_threshold * distances[idx[1]]:
                matches.append(cv2.DMatch(i, idx[0], distances[idx[0]]))
    
    return matches
```

## 实验结果与分析

### Harris角点检测结果

对sudoku.png图像进行Harris角点检测，检测到950个角点。

从检测结果可以看出，Harris角点检测器成功识别出了数独棋盘格的交叉点，以及边缘处的明显转角。检测到的角点主要分布在图像的结构变化明显的区域，这符合角点检测的基本原理。在实验中，我们通过调整参数k和阈值比例，得到了950个角点，这些角点准确地反映了图像中的几何结构特征。

### 特征描述与匹配结果

对UT塔图像（uttower1.jpg和uttower2.jpg）进行角点检测和特征匹配。

从匹配结果可以看出，SIFT特征和HOG特征在匹配性能上存在明显差异：

- SIFT特征：检测到222对匹配点，其中174对为RANSAC筛选后的内点，内点比例约为78.4%。匹配线主要是水平的，表明两幅图像之间的变换主要是水平平移。
- HOG特征：检测到83对匹配点，其中16对为RANSAC筛选后的内点，内点比例约为19.3%。匹配质量明显低于SIFT特征，存在较多错误匹配。

这表明SIFT特征在应对视角变化方面表现更好，这主要得益于SIFT特征的尺度不变性和旋转不变性。而简化实现的HOG特征主要关注局部梯度方向分布，对视角变化的适应能力较弱。

### 图像拼接结果

基于特征匹配结果，使用RANSAC算法估计单应性矩阵，并进行图像拼接。

从拼接结果可以看出：

- 基于SIFT特征的拼接结果更加自然，两幅图像之间的过渡较为平滑，拼接线不明显。
- 基于HOG特征的拼接结果在某些区域存在轻微的扭曲或错位，这主要是由于HOG特征匹配的内点较少，导致计算的单应性矩阵精度较低。

这进一步证明了SIFT特征在图像匹配和拼接任务中的优越性。

### 全景图拼接结果

使用SIFT特征和RANSAC算法对约塞米蒂系列图像进行全景拼接。

全景拼接过程中，通过顺序拼接四张约塞米蒂图像，成功构建了完整的全景图。拼接结果展示了广阔的约塞米蒂风景，视野比单张图像更加开阔。在拼接过程中，通过优化边界处理和图像融合策略，减少了拼接缝隙和亮度不一致现象，使得拼接后的图像过渡自然。

### 算法性能分析

| 性能指标 | SIFT特征 | HOG特征 |
|---------|---------|---------|
| 检测到的角点数量 (uttower1) | 726 | 726 |
| 检测到的角点数量 (uttower2) | 714 | 714 |
| 特征匹配数量 | 222 | 83 |
| RANSAC筛选后的内点数量 | 174 | 16 |
| 内点比例 | 78.4% | 19.3% |
| 拼接质量 | 高 | 中 |

从上表可以看出，虽然两种方法使用了相同的Harris角点，但SIFT特征在匹配质量上明显优于HOG特征。这主要是因为：

- SIFT特征考虑了多尺度信息，对尺度变化具有更好的鲁棒性
- SIFT特征的描述更加精细，包含了128维信息，而简化的HOG特征只有9维
- SIFT算法已经经过多年优化，而实验中实现的HOG特征是简化版本

## 总结与思考

通过本次实验，我们成功实现了Harris角点检测算法，并基于检测到的角点，使用SIFT和HOG特征描述子进行特征匹配和图像拼接。主要成果和收获如下：

### 主要成果

1. 手动实现了Harris角点检测算法，加深了对角点检测原理的理解
2. 成功使用SIFT和HOG特征描述子进行特征描述和匹配，并比较了两者的性能差异
3. 使用RANSAC算法筛选内点并估计变换矩阵，实现了图像拼接
4. 将上述技术应用于约塞米蒂图像序列，成功构建了全景图

### 思考与改进方向

1. **特征点检测改进**：Harris角点检测对尺度变化不敏感，可以考虑使用多尺度Harris检测或直接使用SIFT关键点检测，以提高特征点的质量。
    
2. **特征描述子优化**：本实验中实现的HOG特征是简化版本，可以考虑实现完整的HOG特征，或尝试其他特征描述子如ORB、BRIEF等，以进一步比较不同特征的性能。
    
3. **图像融合优化**：当前的拼接方法在重叠区域简单选择一个图像的像素值，可以考虑使用加权融合或多频段融合等方法，使拼接结果更加自然。
    
4. **全局优化**：对于多图拼接，可以考虑使用bundle adjustment等全局优化方法，减少累积误差，提高全景图的整体质量。

### 结论

本实验通过实践验证了Harris角点检测、特征描述与匹配以及RANSAC算法在图像拼接中的应用。实验结果表明，SIFT特征在特征匹配和图像拼接任务中表现优异，而HOG特征虽然计算简单，但匹配质量较低。通过比较不同特征描述子的性能，我们加深了对特征表示重要性的理解，为今后在计算机视觉领域的学习和研究奠定了基础。

## 参考文献

1. C. Harris and M. Stephens, "A combined corner and edge detector," in *Proceedings of the 4th Alvey Vision Conference*, 1988, pp. 147-151.

2. D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91-110, 2004.

3. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, 2005, vol. 1, pp. 886-893.

4. M. A. Fischler and R. C. Bolles, "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381-395, 1981.

5. M. Brown and D. G. Lowe, "Automatic panoramic image stitching using invariant features," *International Journal of Computer Vision*, vol. 74, no. 1, pp. 59-73, 2007.