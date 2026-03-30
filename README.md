# CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images
This repository is used to present the figures and tables that are not included in the manuscript due to page limitations.
## 1 Detailed Mechanism and Analysis of the Inner-PIoUv2 Loss Function
### 1.1 Basic Formulation of PIoUv2
The PIoU (Powerful-IoU) loss function achieves adaptive penalization for targets of different scales by introducing a penalty factor  , and its expression is as follows:

$$L_{PIoU}=L_{IoU}+1-e^{-P^2}, 0\leq L_{PIoU}\leq 2    \quad\quad\quad\quad\quad {(1)}$$
$$P=\bigg(\frac{dw_1}{w_{gt}}+\frac{dw_2}{w_{gt}}+\frac{dh_1}{h_{gt}}+\frac{dh_2}{h_{gt}}\bigg)/4  \quad\quad\quad\quad\quad {(2)}$$

where $dw_1,dw_2,dh_1$ and $dh_2$ denote the absolute distances between the corresponding boundaries of the predicted box and the target box, respectively, while $w_{gt}$ and $h_{gt}$ represent the width and height of the target box, respectively. The mathematical formulation of PIoUv2 is further defined as:

$$L_{PIoUv2}=u(\lambda q) \cdot L_{PIoU} = 3 \cdot (\lambda q) \cdot e^{-(\lambda q)^2} \cdot L_{PIoU} \quad\quad\quad\quad {(3)}$$

where $q=e^{-P}, q\in (0,1], u(x)=3x \cdot e^{-x^2}$, and $u(\lambda q)$ is a non-monotonic attention function. The penalty factor $P$ is replaced by $q$, where $q$ is used to evaluate the quality of the anchor box. $\lambda$ is a hyperparameter that controls the behavior of the attention function, which is set to 1.1 in this paper.

### 1.2	Introduction of the Inner Mechanism and Final Expression of Inner-PIoUv2
To further optimize the bounding box regression process, Inner-PIoUv2 (as shown in Fig. Supp-1) introduces a scale factor ($ratio$) on the basis of PIoUv2 to dynamically adjust the size of auxiliary bounding boxes. 

Let the center coordinates of the ground-truth box $B^{gt}$ and the anchor box $B$ be $(x_c^{gt}, y_c^{gt})$ and $(x_c,y_c)$, respectively, and their widths and heights be $(w^{gt},h^{gt})$ and $(w,h)$, respectively. With the scale factor $ratio$ (usually ranging from 0.5 to 1.5, and the optimal value in this paper is 1.4), the coordinates of the auxiliary bounding boxes are calculated as follows:

