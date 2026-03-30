# CPC-YOLO-Lightweight-Framework-for-Small-Ship-Detection-in-SAR-Images
This repository is used to present the figures and tables that are not included in the manuscript due to page limitations.
## 1 Detailed Mechanism and Analysis of the Inner-PIoUv2 Loss Function
### 1.1 Basic Formulation of PIoUv2
The PIoU (Powerful-IoU) loss function achieves adaptive penalization for targets of different scales by introducing a penalty factor  , and its expression is as follows:

$$L_{PIoU}=L_{IoU}+1-e^{-P^2}, 0\leq L_{PIoU}\leq 2    \quad\quad\quad\quad {(1)}$$
$$P=\bigg(\frac{dw_1}{w_{gt}}+\frac{dw_2}{w_{gt}}+\frac{dh_1}{h_{gt}}+\frac{dh_2}{h_{gt}}\bigg)/4  \quad\quad\quad\quad {(2)}$$

where $dw_1,dw_2,dh_1$ and $dh_2$ denote the absolute distances between the corresponding boundaries of the predicted box and the target box, respectively, while $w_{gt}$ and $h_{gt}$ represent the width and height of the target box, respectively. The mathematical formulation of PIoUv2 is further defined as:

$$L_{PIoUv2}=u(\lambda q) \cdot L_{PIoU} = 3 \cdot (\lambda q) \cdot e^{-(\lambda q)^2} \cdot L_{PIoU} \quad\quad\quad\quad {(3)}$$
