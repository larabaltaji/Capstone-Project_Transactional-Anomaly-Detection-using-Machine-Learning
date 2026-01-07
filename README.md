# Transactional Anomaly Detection using Machine Learning
## Enhancing Financial Security and Operational Efficiency at IDS Fintech
### Author: Lara Baltaji
### Degree: M.Sc. in Business Data Analytics, American University of Beirut
### Year: 2023
### Collaboration: Integrated Digital Systems (IDS) Fintech
### GitHub Repository: (this repo)

#### Abstract

Detecting anomalies in financial transaction data is a challenging problem due to the rarity, heterogeneity, and evolving nature of abnormal events, as well as the frequent absence of reliable labels. This project investigates supervised and unsupervised machine learning approaches for transactional anomaly detection using real-world financial data from a fintech provider operating in the MENA region.

A comprehensive modeling pipeline was developed, including data preprocessing, domain-driven feature engineering, model benchmarking, hyperparameter tuning, and rigorous evaluation under severe class imbalance. Eight supervised classifiers and four unsupervised models were implemented and compared, including Isolation Forest, One-Class SVM, and deep autoencoder architectures. Model performance was assessed using evaluation metrics appropriate for imbalanced data, such as ROC-AUC, macro-averaged recall, precision, recall, and confusion-matrix analysis.

While several supervised models achieved strong performance when trained on synthetically generated anomalies, the unsupervised autoencoder demonstrated robust detection capabilities without reliance on labels, achieving a ROC-AUC of approximately 0.99 and high recall across multiple anomaly types. Based on both quantitative performance and real-world deployment feasibility, the autoencoder was selected as the optimal model. The project highlights key tradeoffs between supervised and unsupervised approaches in financial anomaly detection and identifies model interpretability as a central limitation and avenue for future research.

#### Problem Setting & Motivation

Financial transaction systems generate high-volume, high-dimensional data where anomalies may correspond to errors, fraud, or operational failures. These anomalies are typically:
- Extremely rare (≈1–2% of records),
- Diverse in nature (point and contextual anomalies),
- Poorly labeled or entirely unlabeled.

Traditional rule-based systems are brittle and difficult to scale, motivating the use of machine learning models that can learn normal transaction behavior and identify deviations automatically.

#### Dataset Overview

- Source: IDS Fintech portfolio management and trading systems
- Size: ~14,500 transactions spanning six months
- Features: Transaction type, pricing, quantities, settlement dates, fees, currencies, and derived temporal/contextual features
- Class Distribution:
-   Normal transactions ≈ 99%
-   Anomalies ≈ 1% (synthetically generated with domain expert guidance)

Seven anomaly types were considered, including abnormal settlement dates, pricing deviations, unusual exchange rates, and invalid quantities.

#### Methodology

The project followed the CRISP-DM framework, with iterative refinement across stages.

##### Data Preparation & Feature Engineering

- Removal of duplicates and leakage prevention via pipeline-based preprocessing
- Feature scaling using standardization
- Domain-informed feature extraction (e.g., settlement date differences, unit price–cost ratios, working-hours indicators)

##### Modeling Approaches

- Supervised Models (8):
Logistic Regression, KNN, Naïve Bayes, SVM, Decision Tree, Random Forest, XGBoost, Artificial Neural Networks

- Unsupervised Models (4):
Isolation Forest, Local Outlier Factor, One-Class SVM, Deep Autoencoder

Hyperparameters were optimized using GridSearch and KerasTuner, targeting ROC-AUC maximization.

#### Evaluation Strategy

Given extreme class imbalance, accuracy was explicitly avoided as a primary metric. Models were evaluated using:
- ROC-AUC
- Macro-averaged recall
- Precision & recall
- Confusion matrices (TP, FP, TN, FN)
Supervised and unsupervised models were compared under identical evaluation conditions.

#### Key Results

- Supervised models achieved near-perfect performance when trained on labeled anomalies (e.g., ANN and XGBoost).
- Among unsupervised methods, the autoencoder outperformed alternatives:
-   ROC-AUC ≈ 0.99
-   Macro-averaged recall ≈ 0.96
-   Successfully detected 32 out of 34 validation anomalies
- 
- The autoencoder generalized across multiple anomaly types without explicit labeling.

#### Model Selection Rationale

Despite strong supervised performance, the autoencoder was selected as the final model due to:
- Independence from labeled anomalies,
- Robust detection of heterogeneous anomaly types,
- Alignment with real-world financial constraints where labels are incomplete or unavailable.

#### Deployment

The final model was deployed using Streamlit, allowing users to input transaction attributes and receive real-time anomaly predictions. Preprocessing pipelines and trained models were serialized to ensure consistency between training and deployment.

#### Limitations & Future Work

- Current model identifies anomalous records but lacks feature-level interpretability.
- Future research directions include:
-   Explainable anomaly detection (e.g., denoising autoencoders, attribution methods),
-   Cell-level anomaly identification,
-   Confidence-based correction suggestions.

#### Technologies Used

Python, Scikit-learn, TensorFlow/Keras, PyOD, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit


_Note: Check out the Jupyter notebook file and project report for more information_


