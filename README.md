# CPC-YOLO: Lightweight Framework for Small Ship Detection in SAR Images
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
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-2.png" alt="Fig.Supp-2" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig.Supp-2 Comparison of Box Regression Loss (CIoU vs. Inner-PIoUv2) on LS-SSDD-v1.0 and SAR-Ship-Datasets.</figcaption>
</figure>

When trained for 52 epochs on the large-scale SAR-Ship-Dataset (as shown in the right panel of Fig. Supp-2), compared with CIoU, Inner-PIoUv2 achieves a significantly faster initial convergence speed and maintains a consistently lower loss value throughout the entire training phase. For example, in the early training stage, the loss value of Inner-PIoUv2 drops rapidly to below 0.4481 at around 5 epochs, while CIoU still hovers around 0.686 in the same period; by the end of training, Inner-PIoUv2 further decreases to 0.32672 (lower than CIoU's lower than CIoU's 0.42667). This indicates that our method can provide effective gradient guidance and accelerate the optimization process in large-scale data scenarios.

It is worth noting that when trained for 110 epochs on the more complex LS-SSDD-v1.0 dataset (as shown in the left panel of Fig. Supp-2), the loss value of Inner-PIoUv2 is slightly higher than that of CIoU with larger fluctuations (its loss value mainly oscillates dynamically between 0.23 and 0.31, while CIoU decreases smoothly to 0.2051). This phenomenon does not indicate poor optimization performance; on the contrary, it precisely reflects the core advantage of the proposed mechanism. By introducing auxiliary bounding boxes with a specific scale factor, Inner-PIoUv2 imposes stricter penalties on minor localization deviations (especially for small target ships). This rigorous optimization constraint can prevent the network from premature convergence, prompting the model to continuously refine and correct the bounding boxes. Therefore, despite the higher training loss value, a higher detection accuracy can be ultimately achieved (as shown in Table I in the main manuscript).

## 2 Supplementary Explanation on the Controlled Variable Design in Ablation Experiments
To more rigorously verify the independent effectiveness of the innovative modules proposed in this paper, we specifically designed two sets of single-variable controlled experiments (Lines 7 and 8) in the ablation experiments (Tables III and IV, shown below) in the main text.

Table III The ABLATION EXPERIMENT ON LS-SSDD-v1.0 OF THE ENTIRE SCENES
|        | P(%)    | R(%)   | AP50(%)| AP(%)   | F1     |Params(M)| FLOPS(G)|
|:-------|:-------:|-------:|:-------|:-------:|-------:|:-------:|-------:|
| 1YOLOv11n                |87.00 |76.10  |83.04 | 33.70   | 81.19   |2.5900 | 10.01|
| 2(Add P2)                |85.06 |76.38  |83.55 | 34.33   | 80.49   |2.6668 | 16.07|
| 3(Remove P5)             |84.89 |76.43  |83.54 | 34.32   | 80.44   |1.9389 | 15.17|
| 4(Add Channel Reduction) |85.09 |76.02  |83.52 | 34.28   | 80.34   |1.8629 | 14.69|
| 5(Add CPCA)              |85.12 |76.64  |84.17 | 35.36   | 80.66   |1.8403 | 14.55|
| 6 CPC-YOLO(Ours)         |85.17 |78.06  |85.29 | 35.85   | 81.46   |1.8403 | 14.55|
| 7(Only Add CPCA)         |84.71 |77.73  |84.01 | 34.37   | 81.07   |2.5674 | 9.87 |
| 8(Only Add Inner-PIoUv2) |84.95 |76.92  |83.82 | 34.26   | 80.74   |2.5900 | 10.01|

TABLE IV THE ABLATION EXPERIMENT ON SAR-SHIP-DATASET OF THE ENTIRE SCENES
|               | P (%) | R (%) | AP50 (%) | AP (%) | F1 (%) | Params (M) | FLOPs (G) |
|---------------|-------|-------|----------|--------|--------|------------|-----------|
| 1 YOLOv11n | 91.02 | 91.63 | 94.10 | 62.77 | 91.32 | 2.5900 | 5.75 |
| 2 (Add P2) | 91.12 | 91.76 | 94.56 | 63.81 | 91.44 | 2.6668 | 9.25 |
| 3 (Remove P5) | 91.23 | 91.62 | 94.27 | 63.67 | 91.42 | 1.9389 | 8.72 |
| 4 (Reduct Channels) | 91.18 | 91.52 | 94.31 | 63.72 | 91.35 | 1.8629 | 8.45 |
| 5 (Add CPCA) | 91.36 | 92.12 | 94.41 | 63.89 | 91.74 | 1.8403 | 8.40 |
| 6 CPC-YOLO (Ours) | 91.76 | 92.53 | 94.72 | 64.41 | 92.14 | 1.8403 | 8.40 |
| 7 (Only Add CPCA) | 91.45 | 91.31 | 94.14 | 63.03 | 91.38 | 2.5674 | 5.70 |
| 8 (Only Add Inner-PIoUv2) | 91.51 | 92.05 | 94.49 | 63.47 | 91.78 | 2.5900 | 5.75 |
(1) On the standalone addition of CPCA (Tables III and IV, Line 7):
Although experiments have proven that the final architecture of "Adjusted Feature Map Scale (Add P2, Remove P5) + CPCA + Inner-PIoUv2" achieves the optimal performance, we need to rule out the possibility that the performance improvement is solely attributed to the introduction of high-resolution feature maps. By only adding the CPCA module to the baseline YOLOv11n, we observe that without changing the macroscopic network structure, the model's Precision is significantly improved, and the parameter count is slightly optimized. This fully confirms the independent effectiveness of the CPCA module in feature extraction, and demonstrates its synergistic enhancement effect with multi-scale structure adjustment.
(2) On the standalone addition of Inner-PIoUv2 (Tables III and IV, Line 8):
This experiment aims to isolate the impact of network architecture optimization and independently verify the superiority of the proposed loss function. The results show that, without any modification to the original YOLOv11n structure, simply replacing the loss function with Inner-PIoUv2 can effectively improve the model's Recall metric. This further highlights the inherent contribution of this dynamic loss function in improving bounding box regression and enhancing sensitivity to tiny objects.


## 3 Comparative Experimental Analysis of Different Attention Mechanisms
To fully verify the superiority of the proposed Channel Prior Convolutional Attention (CPCA), we conducted replacement comparison experiments by substituting it with five mainstream attention mechanisms (CBAM, SimAM, CoordAtt, CAA, and TripletAtt) in the C2PSA module. The experimental results on the two datasets are shown in Table Supp-III and Table Supp-IV, respectively.

TABLE Supp-III COMPARISON OF DIFFERENT ATTENTION MECHANISMS IN C2PSA ON LS-SSDD-v1.0
|        | P(%)    | R(%)   | AP50(%)| AP(%)   | F1     |Params(M)| FLOPS(G)|
|:-------|:-------:|-------:|:-------|:-------:|-------:|:-------:|-------:|
| YOLOv11n*    |85.09 |76.02  |83.52 | 34.28   | 80.34   |1.8629 | 14.69|
| CBAM[S4]     |84.88 |75.78  |84.29 | 34.97   | 80.07   |1.8138 | 14.48|
| SimAM[S5]    |84.79 |76.10  |83.93 | 35.08   | 80.21   |1.8116 | 14.48|
| CoordAtt[S6] |85.47 |75.61  |84.76 | 35.10   | 80.80   |1.8149 | 14.78|
| CAA[S7]      |85.52 |75.96  |84.49 | 35.39   | 80.47   |1.8479 | 14.53|
| TripletAtt[S8]|85.78 |74.44 |83.21 | 34.17   | 79.71   |1.8119 | 14.48|
| CPCA(Ours)   |85.12 |76.64  |84.17 | 35.36   | 80.66   |1.8403 | 14.55|

*Note: TripletAtt denotes TripletAttention. YOLOv11n* denotes YOLOv11n + P2 – P5 + Channels reduction.*

TABLE Supp-IV COMPARISON OF DIFFERENT ATTENTION MECHANISMS IN C2PSA ON SAR-Ship-Dataset
|        | P(%)    | R(%)   | AP50(%)| AP(%)   | F1     |Params(M)| FLOPS(G)|
|:-------|:-------:|-------:|:-------|:-------:|-------:|:-------:|-------:|
| YOLOv11n*    |91.18 |91.52  |94.31 | 63.72   | 91.35   |1.8629 | 8.45|
| CBAM[S4]     |91.32 |92.25  |94.61 | 63.56   | 91.78   |1.8138 | 8.36|
| SimAM[S5]    |92.01 |92.04  |94.67 | 63.82   | 92.02   |1.8116 | 8.36|
| CoordAtt[S6] |91.57 |92.00  |94.41 | 63.52   | 91.78   |1.8149 | 8.36|
| CAA[S7]      |90.89 |92.23  |94.13 | 63.44   | 91.56   |1.8479 | 8.39|
| TripletAtt[S8]|90.93 |91.91 |94.82 | 62.94   | 91.42   |1.8119 | 8.37|
| CPCA(Ours)   |91.36 |92.12  |94.41 | 63.89   | 91.74   |1.8403 | 8.40|

Experimental results demonstrate that the proposed CPCA achieves the best overall detection performance. Specifically, CPCA attains the highest AP of 35.85% and 63.89% on the LS-SSDD-v1.0 and SAR-Ship-Dataset, respectively. In terms of model complexity, replacing the original multi-head self-attention mechanism in the baseline model (YOLOv11n*) with our CPCA module successfully reduces the parameter count from 1.86 M to 1.84 M, and also lowers the computational cost (GFLOPs). Although some ultra-lightweight mechanisms (such as SimAM and TripletAtt) require slightly fewer parameters (approximately 1.81 M), they exhibit a significant performance drop in AP metrics compared to CPCA.

In conclusion, CPCA achieves the optimal balance between computational efficiency and feature extraction capability. It demonstrates excellent ability to suppress complex SAR speckle noise and accurately capture the spatial and channel features of tiny ships, making it the most suitable attention mechanism for this architecture.

## 4 Supplementary Experiments in Different Scenarios on LS-SSDD-v1.0
To fully verify the scene generalization ability of the proposed method, we conducted a detailed comparative analysis of CPC-YOLO with mainstream general-purpose object detection algorithms and SAR image-specific detection algorithms (such as DADP[S1], BANet[S2], and FBRNet[S3]) in inshore and offshore scenarios in the supplementary material (Table Supp-V and Table Supp-VI). Since the official LS-SSDD-v1.0 dataset provides txt documents with inshore and offshore ship annotations, while the SAR-Ship-Dataset does not offer inshore/offshore SAR image classification, the experimental verification is only performed on LS-SSDD-v1.0.

Table Supp-V COMPARISON RESULTS OF DIFFERENT SHIP DETECTION METHODS IN VARIOUS SCENES OF LS-SSDD-v1.0
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">Inshore Scenes (%)</th>
      <th colspan="5">Offshore Scenes (%)</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>AP50</th><th>AP</th><th>F1</th>
      <th>P</th><th>R</th><th>AP50</th><th>AP</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>RetinaNet</td><td>26.5</td><td>91.03</td><td>30.55</td><td>10.24</td><td>41.05</td><td>84.40</td><td>85.32</td><td>88.97</td><td>34.03</td><td>84.86</td></tr>
    <tr><td>Cascade R-CNN</td><td>52.30</td><td>51.41</td><td>45.20</td><td>12.00</td><td>51.85</td><td>83.10</td><td>93.49</td><td>92.11</td><td>36.40</td><td>87.99</td></tr>
    <tr><td>Improved-FCOS</td><td>60.60</td><td>60.10</td><td>54.16</td><td>-</td><td>60.35</td><td>89.20</td><td>94.41</td><td>92.84</td><td>-</td><td>91.73</td></tr>
    <tr><td>YOLOv10n</td><td>61.53</td><td>36.14</td><td>47.71</td><td>20.58</td><td>45.53</td><td>86.78</td><td>80.03</td><td>89.02</td><td>39.79</td><td>83.27</td></tr>
    <tr><td>YOLOv12n</td><td>69.42</td><td>41.58</td><td>55.02</td><td>24.86</td><td>52.00</td><td>92.63</td><td>89.17</td><td>93.83</td><td>41.78</td><td>90.87</td></tr>
    <tr><td>YOLOv11n</td><td>66.71</td><td>37.03</td><td>53.62</td><td>24.21</td><td>47.62</td><td>91.15</td><td>88.86</td><td>92.76</td><td>41.27</td><td>90.02</td></tr>
    <tr><td>CPC-YOLO (Ours)</td><td>68.97</td><td>39.13</td><td>54.47</td><td>24.79</td><td>49.93</td><td>91.35↑</td><td>89.44↑</td><td>93.47</td><td>42.34</td><td>90.38</td></tr>
  </tbody>
</table>

Table Supp-VI COMPARISON OF CPC-YOLO WITH SPECIALIZED SAR SHIP DETECTION METHODS ON INSHORE AND OFFSHORE SCENES OF LS-SSDD-V1.0
<table>
  <thead>
     <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">Inshore Scenes (%)</th>
      <th colspan="5">Offshore Scenes (%)</th>
     </tr>
     <tr>
      <th>P</th><th>R</th><th>AP50</th><th>AP</th><th>F1</th>
      <th>P</th><th>R</th><th>AP50</th><th>AP</th><th>F1</th>
     </tr>
  </thead>
  <tbody>
    <tr>
      <td>DADP [S1]</td>
      <td>68.30</td><td>42.90</td><td>39.17</td><td>-</td><td>52.70</td>
      <td>90.70</td><td>90.70</td><td>89.21</td><td>-</td><td>90.7</td>
    </tr>
    <tr>
      <td>BANet [S2]</td>
      <td>53.80</td><td>63.50</td><td>53.99</td><td>-</td><td>58.25</td>
      <td>86.20</td><td>92.70</td><td>91.13</td><td>-</td><td>89.33</td>
    </tr>
    <tr>
      <td>FBRNet [S3]</td>
      <td>46.90</td><td>58.70</td><td>49.55</td><td>-</td><td>52.14</td>
      <td>85.90</td><td>93.70</td><td>91.75</td><td>-</td><td>89.63</td>
    </tr>
    <tr>
      <td>CPC-YOLO (Ours)</td>
      <td>68.97</td><td>39.13</td><td>54.47</td><td>24.79</td><td>49.93</td>
      <td>91.35</td><td>89.44</td><td>93.47</td><td>42.34</td><td>90.38</td>
    </tr>
  </tbody>
</table>

In the extremely challenging inshore scenarios: Due to the strong interference from land and port facility clutter, detection in this scenario is extremely difficult. As shown in Table Supp-V and Supp-VI, CPC-YOLO achieves an AP50 of 54.47% and an AP of 24.79% in this scenario. This performance comprehensively surpasses all compared SAR-specific detection methods (for example, the best-performing BANet achieves 53.99%), and is significantly superior to the baseline YOLOv11n. Although its AP50 is slightly lower than that of the latest model YOLOv12n (55.02%) with a larger parameter count, CPC-YOLO still exhibits excellent anti-clutter interference ability while maintaining an extremely low parameter count.

In offshore scenarios: Facing dense tiny ships on the sea surface, CPC-YOLO demonstrates extremely high localization accuracy. Its AP50 reaches 93.47%, and the stricter AP metric reaches 42.34% (even surpassing YOLOv12n's 41.78%). Compared with classic SAR-specific methods DADP and BANet, our method establishes a leading advantage in both recall and average precision.
Comprehensive experimental results across different scenarios show that CPC-YOLO successfully overcomes the false positive problems caused by complex nearshore backgrounds and the missed detection problems of tiny offshore targets, and exhibits stable and highly competitive detection performance under various conditions.

## 5 Visualization Analysis
To clearly demonstrate the direct effectiveness of the proposed architectural improvements (such as the P2 detection head and CPCA module) and avoid visual clutter, this section focuses exclusively on a one-to-one visual comparison between the baseline network (YOLOv11n) and our CPC-YOLO on the LS-SSDD-v1.0 and SAR-Ship-Dataset datasets. A comprehensive quantitative evaluation against other state-of-the-art (SOTA) methods has been detailed in Tables I and II of the main text.

(1) Qualitative Detection Result Analysis
As shown in Fig. Supp-3 and Fig. Supp-5, when facing complex scenarios with strong speckle interference and dense ship arrangements, the baseline model tends to produce numerous missed detections (red boxes) and false alarms (yellow circles). In contrast, CPC-YOLO can more accurately localize tiny targets, significantly reducing false positives and false negatives, and demonstrating excellent detection robustness in complex SAR images.

(2) Deep Detection Head Heatmap Analysis (Layer 16 vs. Layer 19)
To explore the inherent mechanism of the P2 detection head, we extracted the feature heatmaps of the baseline's deepest small-target detection head (Layer 16) and CPC-YOLO's newly added P2 deep small-target detection head (Layer 19) from the rightmost two columns of Fig. Supp-3 and Fig. Supp-5, with iou=0.3 and conf_threshold=0.16. The heatmaps here are generated using the LayerCAM feature map [S9]. It is important to note that: Combined with the quantitative evaluation for visualization analysis, we extracted the feature heatmaps for visualization purposes. The above parameter configuration of iou=0.3 and conf_threshold=0.16 is only used to clearly present the pattern of feature activation, and does not represent the evaluation metrics of the model during final inference. The standard confidence threshold adopted in the main text for evaluation is 0.2, and the IoU threshold is 0.5, with completely different application scenarios.

The comparison reveals that the features at Layer 19 of CPC-YOLO exhibit a prominent "highly focused" pattern. It effectively filters out redundant background and texture information, retaining only the most critical semantic features of the targets. This highly focused feature representation greatly enhances the semantic information intensity of tiny targets, providing fundamental support for high-precision localization.

(3) Attention Module Heatmap Analysis (Layer 10)
To intuitively compare the feature extraction differences between C2PSA and our proposed C2PSA_CPCA module, we extracted the heatmaps at Layer 10 of both models (as shown in Fig. Supp-4 and Fig. Supp-6). It can be observed that the baseline model's heatmap retains a large amount of invalid texture and background information. Although the target activation region is complete, there is obvious "background activation overflow", and the activation boundary is relatively loose. On the contrary, thanks to the introduction of the CPCA module, the activation region in CPC-YOLO's heatmap is more compact, strictly fitting the core area of the targets. Background noise is effectively suppressed, resulting in a much "cleaner" activation response.

<figure style="margin-bottom: 2em;">
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-3.png" alt="Fig.Supp-3" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig. Supp-3: Visual detection results and deep detection head heatmap comparison between the proposed CPC-YOLO method and the baseline network on LS-SSDD-v1.0. Note: Green boxes, blue boxes, and red boxes denote the ground truth, correctly detected targets, and missed targets, respectively. Yellow circles indicate false alarm targets.</figcaption>
</figure>


<figure style="margin-bottom: 2em;">
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-4.png" alt="Fig.Supp-4" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig. Supp-4: Visual comparison of heatmaps at Layer 10 (attention module) between the proposed CPC-YOLO method and the baseline network on LS-SSDD-v1.0.</figcaption>
</figure>


<figure style="margin-bottom: 2em;">
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-5.png" alt="Fig.Supp-5" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig. Supp-5: Visual detection results and deep detection head heatmap comparison between the proposed CPC-YOLO method and the baseline network on the SAR-Ship-Dataset. Red boxes denote missed detections, and yellow circles indicate false alarm targets.</figcaption>
</figure>


<figure style="margin-bottom: 2em;">
  <img src="https://raw.githubusercontent.com/yangguang-v/CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images/main/Fig.Supp-6.png" alt="Fig.Supp-6" width="100%">
  <figcaption style="margin-top: 0.5em;">Fig. Supp-6: Visual comparison of heatmaps at Layer 10 (attention module) between the proposed CPC-YOLO method and the baseline network on the SAR-Ship-Dataset.</figcaption>
</figure>

Combined with the quantitative evaluation in the main text, the above visualization analysis fully demonstrates that CPC-YOLO possesses significant and distinct advantages in feature focusing, noise suppression, and the detection of small and dense targets in SAR images.

## Supplementary references
[S1] Z. Cui, Q. Li, Z. Cao, and N. Liu, “Dense attention pyramid networks for multi-scale ship detection in SAR images,” IEEE Trans. Geosci. Remote Sens., J., vol. 57, no. 11, pp. 8983-8997, Nov. 2019. 

[S2] Q. Hu, S. Hu, and S. Liu, “BANet: A balance attention network for anchor-free ship detection in SAR images,” IEEE Tans. Geosci. Remote Sens., J., vol. 60, pp. 1-12, 2022.

[S3] J. Fu, X. Sun, Z. Wang, and K. Fu, “An anchor-free method based on feature balancing and refinement network for multiscale ship detection in SAR images,” IEEE Trans. Geosci. Remote Sens., J., vol. 59, no. 2, pp. 1331-1344, Feb. 2021.

[S4] S. Woo, J. Park, J. Lee, et al., “Cbam: Convolutional block attention module,” in Comput. Vis. (ECCV), pp. 3-19, 2018.

[S5] L. Yang, R. Zhang, L Li, et al., “Simam: A simple, parameter-free attention module for convolution neural networks,” in International conference on machine learning. (PMLR) , pp. 11863-11874, 2021.

[S6] Q. Hou, D. Zhou, J. Feng, “Coordinate attention for efficient mobile network design,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 13713-13722, 2021.

[S7] X. Cai, Q. Lai, Y. Wang, et al., “Poly kernel inception network for remote sensing detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 27706-27716, 2024.

[S8] D. Misra, T. Nalamada, A. Arasanipalai, et al., “Rotate to attend: Convolutional triplet attention module,” in Proc. IEEE/CVF winter conference on applications of computer vision. (WACV), pp. 116-131, 2018.

[S9] P. T. Jiang, C. B. Zhang, Q. Hou, et al., “Layercam: Exploring hierarchical class activation maps for localization,” IEEE transactions on image processing, J., vol. 30, pp. 5875-5888, 2021.
