# Credit Card Fraud Detection using Denoising Autoencoder

This repository contains a PyTorch implementation of a **denoising autoencoder** for anomaly detection on credit card transactions.  
The model is trained only on *normal* transactions and aims to reconstruct them well; higher reconstruction errors indicate potential fraud.

---

## Dataset
We used the public **Credit Card Fraud Detection** dataset from Kaggle.  
Place the CSV (usually named `creditcard.csv`) under the Kaggle `/input` folder when running in a Kaggle Notebook.

- 284,807 transactions  
- 492 fraud cases (highly imbalanced)

---

## Methodology

1. **Preprocessing**
   - Scaled `Time` and `Amount` features using `StandardScaler`.
   - Split data into train/validation/test with stratified splits.
   - Only normal transactions used to train the autoencoder.

2. **Model**
   - Denoising Autoencoder with encoder → latent dimension 16 → decoder.
   - Added Gaussian noise during training for robustness.

3. **Training**
   - Loss: Mean Squared Error between input and reconstruction.
   - Early stopping based on validation loss.
   - Evaluated reconstruction error on validation set to choose a threshold that maximizes F1.

4. **Evaluation**
   - Compute reconstruction error for test set.
   - Flag as fraud if error ≥ threshold.
   - Additionally trained a One-Class SVM on the learned latent space for comparison.

---

## Results

| Metric | Autoencoder (Threshold) | One-Class SVM on Latent |
|--------|-------------------------|------------------------|
| ROC-AUC | **0.9655** | 0.9032 |
| PR-AUC | **0.4678** | 0.0245 |
| Precision | 0.3801 | – |
| Recall | 0.6633 | – |
| F1-Score | 0.4833 | – |

**Confusion Matrix (Autoencoder):**

                 Predicted Normal   Predicted Fraud
Actual Normal          56,758             106
Actual Fraud               33              65


##Classification Report 

              precision    recall  f1-score   support
         0      0.9994    0.9981    0.9988    56864
         1      0.3801    0.6633    0.4833       98

accuracy                          0.9976    56962
macro avg     0.6898    0.8307    0.7410    56962
weighted avg  0.9984    0.9976    0.9979    56962

## Visualization

We plot the reconstruction error distributions for normal vs. fraud transactions and mark the decision threshold.  
Fraudulent transactions tend to have higher reconstruction errors than normal transactions.

![Reconstruction Error Distribution]

 [   33    65]]
