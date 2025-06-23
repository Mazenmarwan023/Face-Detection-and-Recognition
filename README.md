# Face Detection and Recognition with PCA and Eigenfaces

## 📌 Overview

This project implements a complete pipeline for **face detection and recognition** using **Principal Component Analysis (PCA)** and **Eigenfaces**. It leverages standard face datasets such as Yale or ORL to detect, compress, and recognize facial images efficiently, with performance evaluation using **ROC curves** and **accuracy metrics**.

---

## 🚀 Features

- ✅ **Face Detection** using Haar Cascades or MTCNN
- ✅ **Dimensionality Reduction** using PCA (Eigenfaces)
- ✅ **Face Recognition** via k-NN in PCA subspace
- ✅ **Performance Evaluation** (Accuracy, ROC curve, AUC)

---

## 🧠 Methodology

### 1. Face Detection
- Convert input images to grayscale.
- Normalize using histogram equalization.
- Detect faces using **Viola-Jones (Haar cascades)** or **deep learning (MTCNN)**.

### 2. PCA & Eigenfaces for Recognition
- Align and crop face images to a fixed size (e.g., 100×100).
- Flatten each image into a vector.
- Perform PCA:
  - Compute the **mean face** and **covariance matrix**.
  - Extract top eigenvectors (**eigenfaces**).
  - Project training faces onto the reduced PCA subspace.
- Recognize faces by projecting test images into the subspace and using **k-NN** or **SSD** for classification.

### 3. Performance Evaluation
- **Accuracy**: % of correct classifications
- **ROC Curve (One-vs-All)**:
  - **TPR** vs **FPR** per class
  - **AUC Score** to measure discrimination power
- **Distance Threshold Analysis**: Lower distance → higher recognition confidence

---

🖼️ **Screenshots**:  

1.Face Detection


<img width="729" alt="Screenshot 2025-06-23 at 4 19 21 PM" src="https://github.com/user-attachments/assets/1ebc11d3-4f33-4a25-827d-6232d37b6783" />

   
2.Face Recognition


<img width="729" alt="Screenshot 2025-06-23 at 4 19 49 PM" src="https://github.com/user-attachments/assets/296ef58c-99f2-42fb-bb15-be6f82866019" />

    
3.ROC Curve


<img width="729" alt="Screenshot 2025-06-23 at 4 20 13 PM" src="https://github.com/user-attachments/assets/1135a052-e2aa-4a2c-836c-f72f63202c6e" />


---

## 📊 Results

| # Eigenfaces | Accuracy (%) | Avg. AUC |
|--------------|--------------|----------|
| 10           | 85.2         | 0.87     |
| 20           | 92.5         | 0.93     |
| 50           | 94.1         | 0.96     |

- Accuracy saturates at around **50 eigenfaces**
- Best AUC: **"Edward" (0.76)** — consistent lighting
- Worst AUC: **"Kenneth" (0.12)** — varied expressions

---


## 🛠️ Technologies

- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

---

## 📈 Future Work

- Integrate deep learning-based face embedding models
- Explore LDA (Linear Discriminant Analysis) for comparison
- Use more challenging datasets with occlusion and background variation

---

## Contributor

<div>
<table align="center">
  <tr>
        <td align="center">
      <a href="https://github.com/Mazenmarwan023" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
        <br />
        <sub><b>Mazen Marwan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mohamedddyasserr" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126451832?v=4" width="150px;" alt="Mohamed yasser"/>
        <br />
        <sub><b>Mohamed yasser</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Seiftaha" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127027353?v=4" width="150px;" alt="Saif Mohamed"/>
        <br />
        <sub><b>Saif Mohamed</b></sub>
      </a>
    </td> 
  </tr>
</table>
</div>



## 📜 License

This project is open-source and available under the [MIT License](LICENSE).



