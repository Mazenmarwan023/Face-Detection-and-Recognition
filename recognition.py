import os
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

class FaceRecognizer:
    def __init__(self, train_dir, test_dir, img_shape=(112, 92), n_components=20):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_shape = img_shape
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.labels = None
        self.label_to_name = {}

    def _load_images(self, directory):
        X = []
        y = []
        label_map = {}
        label_counter = 0
        for person_name in sorted(os.listdir(directory)):
            person_path = os.path.join(directory, person_name)
            if not os.path.isdir(person_path):
                continue
            if person_name not in label_map:
                label_map[person_name] = label_counter
                self.label_to_name[label_counter] = person_name
                label_counter += 1
            for fname in sorted(os.listdir(person_path)):
                if fname.lower().endswith('.jpg'):
                    img = cv2.imread(os.path.join(person_path, fname), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.img_shape)
                    X.append(img.flatten())
                    y.append(label_map[person_name])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
    

    #PCA calculations
    def fit(self):
        X, y = self._load_images(self.train_dir)
        self.labels = y
        # Compute mean face
        self.mean_face = np.mean(X, axis=0)
        X_centered = X - self.mean_face
        # Compute covariance matrix
        cov = np.dot(X_centered, X_centered.T)
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvectors by eigenvalues (descending)
        idx = np.argsort(-eigvals)
        eigvecs = eigvecs[:, idx]
        # Compute actual eigenfaces
        eigenfaces = np.dot(X_centered.T, eigvecs)
        # Normalize eigenfaces
        for i in range(eigenfaces.shape[1]):
            eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
        self.eigenfaces = eigenfaces[:, :self.n_components]
        # Project training images
        self.projections = np.dot(X_centered, self.eigenfaces)

    def predict(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_shape)
        x = img.flatten().astype(np.float32)
        x_centered = x - self.mean_face
        proj = np.dot(x_centered, self.eigenfaces)
        # Find closest training projection
        dists = np.linalg.norm(self.projections - proj, axis=1)
        idx = np.argmin(dists)
        label = self.labels[idx]
        return self.label_to_name[label], dists[idx]

    def evaluate(self):
        X_test, y_test = self._load_images(self.test_dir)
        y_pred = []
        scores = []
        for i in range(len(X_test)):
            x = X_test[i]
            x_centered = x - self.mean_face
            proj = np.dot(x_centered, self.eigenfaces)
            dists = np.linalg.norm(self.projections - proj, axis=1)
            idx = np.argmin(dists)
            label = self.labels[idx]
            y_pred.append(label)
            scores.append(np.min(dists))
        y_test_names = [self.label_to_name[y] for y in y_test]
        y_pred_names = [self.label_to_name[y] for y in y_pred]
        accuracy = np.mean(y_test == y_pred)
        print(f"Accuracy: {accuracy*100:.2f}%")
        self.plot_roc(y_test, y_pred, scores)
        return y_test_names, y_pred_names

    def plot_roc(self, y_true, y_pred, scores):
        # One-vs-all ROC for each class
        from sklearn.metrics import roc_curve, auc
        n_classes = len(set(y_true))
        plt.figure(figsize=(10, 8))
        for c in range(n_classes):
            y_true_bin = (np.array(y_true) == c).astype(int)
            y_pred_score = -(np.array(scores))  # Lower distance = higher score
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {self.label_to_name[c]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (One-vs-All)')
        plt.legend()
        plt.show()

# Example usage:
# recognizer = FaceRecognizer('Processed/train/color', 'Processed/test/color')
# recognizer.fit()
# print(recognizer.predict('Processed/test/color/John/1-01.jpg'))
# recognizer.evaluate()
