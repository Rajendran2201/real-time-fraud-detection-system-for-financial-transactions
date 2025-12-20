The project is titled as Fraud Detection System for Financial Transactions. I designed and engineered it from scratch with a strong focus on production-ready architecture. 

It's a binary classification system operating on highly imbalanced transactional data. I handled the full pipeline: starting with data preprocessing and feature engineering on structured tabular data, then training an XGBoost model optimized for performace and interpretability. 

Beyond just training the model, I paid close attention to the best practices. I explicitly optimized the decision threshold instead of relying on th default 0.5, using validation data to find the sweet spot that balances precision and recall based on business impact. I evaluated thoroughly with ROC-AUC, PR-AUC, and analysed the confusion matrix to quantify false positives versus false negatives. 

To make the system explainable, I integrated SHAP: generating both global feature importance and per-prediction local explanations. From an engineering standpoint, I modularised the entire notebook code into seperated pipelines and performed rigorous unit testing on them. 

To make it production ready, I containerized the entire inference service using Docker, and deployed it to AWS (usinf EC2), This ensures scalability, reproducibility, and easy integration into larger systems. 

Overall, this project strengthened my skills in building end-to-end ML systems with strong software engineering principles-modular code, seperation of concerns, artifact management, conatineriztion, and cloud deployment - which I believe are crtiical for reliabale ML systems. 