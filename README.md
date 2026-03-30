# CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images
This repository is used to present the figures and tables that are not included in the manuscript due to page limitations.
## 1 Detailed Mechanism and Analysis of the Inner-PIoUv2 Loss Function
### 1.1 Basic Formulation of PIoUv2
The PIoU (Powerful-IoU) loss function achieves adaptive penalization for targets of different scales by introducing a penalty factor $P$, and its expression is as follows:

$$L_{PIoU}=L_{IoU}+1-e^{-P^2}, 0\leq L_{PIoU}\leq 2    \quad\quad\quad\quad\quad\quad\quad {(1)}$$
$$P=\bigg(\frac{dw_1}{w_{gt}}+\frac{dw_2}{w_{gt}}+\frac{dh_1}{h_{gt}}+\frac{dh_2}{h_{gt}}\bigg)/4  \quad\quad\quad\quad\quad\quad\quad {(2)}$$

where $dw_1,dw_2,dh_1$ and $dh_2$ denote the absolute distances between the corresponding boundaries of the predicted box and the target box, respectively, while $w_{gt}$ and $h_{gt}$ represent the width and height of the target box, respectively. The mathematical formulation of PIoUv2 is further defined as:

$$L_{PIoUv2}=u(\lambda q) \cdot L_{PIoU} = 3 \cdot (\lambda q) \cdot e^{-(\lambda q)^2} \cdot L_{PIoU} \quad\quad\quad\quad {(3)}$$

where $q=e^{-P}, q\in (0,1], u(x)=3x \cdot e^{-x^2}$, and $u(\lambda q)$ is a non-monotonic attention function. The penalty factor $P$ is replaced by $q$, where $q$ is used to evaluate the quality of the anchor box. $\lambda$ is a hyperparameter that controls the behavior of the attention function, which is set to 1.1 in this paper.

### 1.2	Introduction of the Inner Mechanism and Final Expression of Inner-PIoUv2
To further optimize the bounding box regression process, Inner-PIoUv2 (as shown in Fig. Supp-1) introduces a scale factor ($ratio$) on the basis of PIoUv2 to dynamically adjust the size of auxiliary bounding boxes. 

<figure style="margin-bottom: 2em;">
  <img src="https://github.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-1.png" alt="Fig.Supp-1" width="60%">
  <figcaption style="margin-top: 0.5em;">Fig.Supp-1 The representation of the Inner-PIoUv2 loss function. The area of the turquoise box represents the intersection between the target box and the anchor box, while the area of the orange box represents the intersection between the inner target box.</figcaption>
</figure>

Let the center coordinates of the ground-truth box $B^{gt}$ and the anchor box $B$ be $(x_c^{gt}, y_c^{gt})$ and $(x_c,y_c)$, respectively, and their widths and heights be $(w^{gt},h^{gt})$ and $(w,h)$, respectively. With the scale factor $ratio$ (usually ranging from 0.5 to 1.5, and the optimal value in this paper is 1.4), the coordinates of the auxiliary bounding boxes are calculated as follows:

$$b_l^{gt}=x_c^{gt}-\frac{w^{gt} \ast ratio}{2}, \quad b_r^{gt}=x_c^{gt} + \frac{w^{gt}\ast ratio}{2}  \quad\quad {(4)}$$ 
$$b_t^{gt}=y_c^{gt}-\frac{h^{gt} \ast ratio}{2}, \quad b_b^{gt}=y_c^{gt} + \frac{h^{gt}\ast ratio}{2}  \quad\quad {(5)}$$ 
$$\quad\quad\quad b_l=x_c-\frac{h \ast ratio}{2}, \quad b_r=x_c + \frac{h^\ast ratio}{2}  \quad\quad\quad\quad {(6)}$$ 
$$\quad\quad\quad b_t=y_c-\frac{h \ast ratio}{2}, \quad b_b=y_c + \frac{h^\ast ratio}{2}  \quad\quad\quad\quad {(7)}$$ 

where $b_l^{gt},b_r^{gt},b_t^{gt},b_b^{gt}$ denote the coordinates of the left, right, top, and bottom boundaries of the ground-truth box, respectively, and $b_l,b_r,b_t,b_b$ denote the coordinates of the left, right, top, and bottom boundaries of the anchor box, respectively. Based on the generated auxiliary bounding boxes, the calculation of the Inner Intersection over Union ($IoU^{inner}$) is defined as follows:

$$ inner=(\min(b_r^{gt},b_r)-\max(b_l^{gt},b_l))\cdot(\min( b_b^{gt},b_b)-\max(b_t^{gt},b_l)) \quad\quad {(8)}$$ 
$$ union=(w^{gt}\cdot h^{gt})\cdot (ratio)^2 + (w\cdot h)\cdot (ratio)^2 -inner \quad\quad {(9)}$$ 
$$ \quad\quad\quad\quad IoU^{inner}=\frac{inner}{union} \quad\quad\quad\quad {(10)} $$

Here, $inner$ denotes the intersection area between the scaled auxiliary ground-truth box and the auxiliary anchor box after applying the scale factor, while $union$ denotes the union area of these two boxes.

The Inner-IoU loss inherits the partial characteristics of the IoU loss, and its value range remains [0,1]. By applying the Inner-IoU loss to the existing boundary regression loss function based on PIoUv2, the final Inner-PIoUv2 expressions can be obtained:

$$ \quad\quad\quad\quad\quad L_{Inner-IoU} = 1- IoU^{inner} \quad\quad\quad\quad\quad {(11)} $$
$$ \quad\quad\quad L_{Inner-IoU} = L_{Inner-PIoU} + 1-e^{-P^2} \quad\quad\quad{(12)} $$
$$L_{Inner-PIoUv2}=u(\lambda q)\cdot L_{Inner-PIoU} = 3\cdot (\lambda q)\cdot e^{-(\lambda q)^2} \cdot L_{Inner-PIoU} {(13)} $$

where $u(\lambda q)$ represents the attention function, $q=e^{-P},q\in(0,1]$, which is used to measure the quality of the anchor box. When $q=1$, it implies $P=0$, meaning the anchor box is perfectly aligned with the target box. As $P$ increases, $q$ decreases gradually, indicating a lower quality of the anchor box. $\lambda$ is a hyperparameter that controls the behavior of the attention function.

### 1.3	In-Depth Analysis of the Dynamic Mechanism of Inner-PIoUv2 and Ablation Studies on the Scale Factor
The dynamic adjustment mechanism of the proposed Inner-PIoUv2 loss function is the core of its superior performance in complex scenarios. This dynamic property is mainly reflected in the following two dimensions:
(1) Structural Dynamic Adaptation (Based on the Scale Factor $ratio$)

We construct auxiliary bounding boxes using the adjustable scale factor $ratio$. Although $ratio$ is a preset hyperparameter during a single training run, different values of ratio enable the loss function to dynamically adapt to regression samples of varying quality and size.
(a) When $ratio > 1$: The auxiliary bounding boxes are larger than the original ones. This amplifies the weight of position deviations in the loss calculation, forcing the model to perform rigorous optimization for even minor center-point offsets. This high-intensity spatial penalty is particularly beneficial for the accurate localization of small target ships.
(b) When $ratio < 1$: The auxiliary bounding boxes are smaller than the original ones. This relatively relaxes the requirement for absolute center-point precision, instead focusing more on the overall intersection area of the boxes, which is more effective for the detection of medium and large targets.

As shown in the ablation studies in Table Supp-I and Table Supp-II, we explore the detection results for $ratio\in [1.1, 1.5]$. When $ratio=1.4$, the model achieves the optimal dynamic balance between small objects detection penalty and overall regression stability, resulting in the best average precision and F1-score on both the LS-SSDD-v1.0 and SAR-Ship-Dataset datasets.

