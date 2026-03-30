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
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-1.png" alt="Fig.Supp-1" width="60%">
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

Table Supp-I Impact of Different   Values on Results on LS-SSDD-v1.0
| Ratio | P(%) | R(%) | AP50(%) | AP(%) | F1 |
|:-------|:-------:|-------:|:-------|:-------:|-------:|
| 1.1   |87.38    |74.25  |84.30   | 35.98   | 80.28   |
| 1.2   |87.26    |77.44  |83.82  | 36.09   | 79.78   |
| 1.3   |87.20    |75.80  |83.56  | 35.80   | 81.10   |
| <strong>1.4</strong> |<b>85.17</b> |<b>78.06</b> |<b>85.29</b> | 35.85   | <b>81.46</b> |
| 1.5   |86.38    |69.12  |83.38  | <b>36.08</b>   | 76.79   |

*Note: AP50 denotes mAP@0.5, AP denotes mAP@0.5:0.95.*

Table Supp-II Impact of Different   Values on Results on SAR-Ship-Dataset
| Ratio | P(%) | R(%) | AP50(%) | AP(%) | F1 |
|:-------|:-------:|-------:|:-------|:-------:|-------:|
| 1.1   |91.38    |91.99  |94.21   | 63.92   | 91.68  |
| 1.2   |91.19   |91.79  |94.41  | 64.36   | 91.49   |
| 1.3   |91.42    |92.13  |94.36  | 64.29   | 91.77   |
| <strong>1.4</strong>  |<b>91.76</b>    |<b>92.53</b>  |<b>94.72</b> | <b>64.41</b>   | <b>92.14</b>   |
| 1.5   |91.88    |90.99  |94.08  | 63.73   | 91.43   |

(2) Dynamic Focus During Training (Based on the Attention Function)
Our loss function introduces the non-monotonic attention function $u(x)=3x\cdot e^{-x^2}$. During training, this function dynamically adjusts the weight allocation of each predicted box in the total loss based on its real-time localization quality (the value of $q$). This mechanism effectively suppresses the harmful gradients from low-quality samples (outliers), and allows the model to dynamically focus on the "medium-quality samples" with the greatest optimization potential, thereby accelerating convergence and raising the final localization upper bound.

### 1.4	Dynamic Comparative Analysis of CIoU and Inner-PIoUv2 Loss Curves on Two Datasets
To intuitively demonstrate the superiority of the proposed Inner-PIoUv2 loss function, we compare it with the baseline CIoU loss in terms of the bounding box regression loss (box loss) during training, with the results shown in Fig. Supp-2.

<figure style="margin-bottom: 2em;">
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-1.png" alt="Fig.Supp-2" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig.Supp-2 Comparison of Box Regression Loss (CIoU vs. Inner-PIoUv2) on LS-SSDD-v1.0 and SAR-Ship-Datasets.</figcaption>
</figure>

When trained for 52 epochs on the large-scale SAR-Ship-Dataset (as shown in the right panel of Fig. Supp-2), compared with CIoU, Inner-PIoUv2 achieves a significantly faster initial convergence speed and maintains a consistently lower loss value throughout the entire training phase. For example, in the early training stage, the loss value of Inner-PIoUv2 drops rapidly to below 0.4481 at around 5 epochs, while CIoU still hovers around 0.686 in the same period; by the end of training, Inner-PIoUv2 further decreases to 0.32672 (lower than CIoU's lower than CIoU's 0.42667). This indicates that our method can provide effective gradient guidance and accelerate the optimization process in large-scale data scenarios.

It is worth noting that when trained for 110 epochs on the more complex LS-SSDD-v1.0 dataset (as shown in the left panel of Fig. Supp-2), the loss value of Inner-PIoUv2 is slightly higher than that of CIoU with larger fluctuations (its loss value mainly oscillates dynamically between 0.23 and 0.31, while CIoU decreases smoothly to 0.2051). This phenomenon does not indicate poor optimization performance; on the contrary, it precisely reflects the core advantage of the proposed mechanism. By introducing auxiliary bounding boxes with a specific scale factor, Inner-PIoUv2 imposes stricter penalties on minor localization deviations (especially for small target ships). This rigorous optimization constraint can prevent the network from premature convergence, prompting the model to continuously refine and correct the bounding boxes. Therefore, despite the higher training loss value, a higher detection accuracy can be ultimately achieved (as shown in Table I in the main manuscript).
