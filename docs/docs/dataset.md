# Understanding the dataset 

**Title**: Credit Card Transactions Dataset for Fraud Detection (Used in: A Hybrid Anomaly Detection Framework Combining Supervised and Unsupervised Learning)

**About the dataset:**
- The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
- This dataset presents transactions that occurred in two days, where there were 492 frauds out of 284,807 transactions with 30 numeric attributes. 
- The dataset is highly unbalanced; the positive class (frauds) accounts for 0.172% of all transactions. 
- Target variable: Class (binary label; 0 = legitimate, 1 = fraud)

This dataset was used in the research article titled "A Hybrid Anomaly Detection Framework Combining Supervised and Unsupervised Learning for Credit Card Fraud Detection". The study proposes an ensemble model integrating techniques such as Autoencoders, Isolation Forest, Local Outlier Factor, and supervised classifiers including XGBoost and Random Forest, aiming to improve the detection of rare fraudulent patterns while maintaining efficiency and scalability.


**Recommended Evaluation Metric:** Area Under the Precision-Recall Curve (AUPRC). 

#### 1. Basic Dataset Description

* **Source:** Originally published by the Machine Learning Group at Université Libre de Bruxelles.
* **Scope:** Credit card transactions made by European cardholders over two days in September 2013. 
* **Total records (rows):** 284,807
* **Total features (columns):** 30 numeric attributes
* **Target variable:** Class (binary label; 0 = legitimate, 1 = fraud) 
* **Fraudulent transactions:** 492
* **Legitimate transactions:** 284,315
* **Fraud percentage:** ≈ 0.172% (highly imbalanced)


#### 2. Feature Structure and Meaning

##### Non-Anonymized Features

Two features are original and interpretable, which are `time` and `amount`. 
- `time`: number of seconds elapsed between this transaction and the first transaction in the dataset.
- `amount`: monetary amount of the transactions (in unspecified currency units)

##### PCA-Transformed Features 
- The features from `V1` to `V28` are PCA-transformed. 
- These features are the result of PCA applied to the original transactional features for privacy and confidentiality reasons. 
- They do not have direct real-world interpretability (e.g., merchant type, MCC< location, etc.).

#### 3. Dataset Characteristics (Key Challenges) 

1. Extreme Class Imbalance 
- Only ~0.17% of the records are fraudulent.
- Accuracy is not a reliable metric in such an imbalanced setting: a naive model that predicts "non-fraud" will always have >99% accuracy. 

2. Anonymized Input Features 
- You do not have raw business interpretable attributes like transaction category, merchant type, location, etc.
- Anomaly or fraud pattern learning must rely solely on the PCA components and the available non-anonymized fields (time and amount).

3. Real-World Representativeness
- The dataset reflects a real industry problem where fraud is a rare event. 
- Techniques for handling imbalanced data are essential in modelling. 

#### 4. Common Processing and Engineering Practices

1. **Preprocessing**
- Scaling/Normalization for `amount` (PCA features are already scaled).
- Time feature engineering: Extract periodic patterns (hour of day, relative time windows). Segment by transaction age or time buckets. 

2. **Handling Imbalance**
- Sampling methods: Oversampling (SMOTE, ADASYN), Undersampling - reduce legitimate samples 
- Class weights: assign higher weights to minory class durin training
- Threshold tuning: optimize decision boundary for recall/precision 

3. Evaluation metrics: 
- Precision, Recall (Sensitivity)
- F1-Score
- ROC-AUC/ PR-AUC
