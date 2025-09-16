# Customer-Churn-Prediction-
This project explores customer churn data and builds a machine learning model to predict whether a customer will leave (churn) or stay. Churn prediction is valuable for businesses (e.g., telecoms, SaaS) because retaining customers is cheaper than acquiring new ones.

The project walks through:

1. Exploratory Data Analysis (EDA)
2. Data Cleaning & Feature Engineering
3. Preprocessing with pipelines
4. Logistic Regression for classification
5. Model evaluation (accuracy, ROC-AUC, confusion matrix)
6. Saving the trained model for reuse

🛠️ Tools & Libraries
1. Python
2. pandas, numpy, matplotlib – data wrangling & visualization
3. scikit-learn – preprocessing, train/test split, logistic regression, evaluation metrics
4. joblib – model persistence

📊 Exploratory Data Analysis
1. Key insights from the data exploration:
2. Missing values in InternetService were filled with blanks.
3. No significant correlations in raw numeric data.
4. Customers with month-to-month contracts showed higher churn rates.
5. Longer contracts → lower average monthly charges (as expected).
6. Visualizations:
. Pie chart of churn distribution
. Bar chart of contract type vs average monthly charges

🧹 Feature Engineering & Preprocessing
1. Column Cleaning: Standardized column names (lowercase, no spaces/special characters).
2. Target Variable: Converted Churn into binary labels (Yes/No → 1/0).
3. Numeric vs Categorical Split:
4. Numeric → scaled with StandardScaler (mean=0, std=1).
5. Categorical → encoded with OneHotEncoder (handle unknown categories).
6. Preprocessing applied inside a ColumnTransformer

🤖 Model
. A Logistic Regression classifier was used:
. class_weight='balanced' → handles class imbalance (churn ~17% of customers).
. Logistic regression outputs a probability of churn, converted to binary predictions with a 0.5 threshold.

📈 Evaluation
On the test set (20% hold-out):
. Accuracy: 93% 
. ROC-AUC: 0.99 
. Confusion Matrix: shows trade-off between catching churners vs false alarms.
Metrics used:
. Accuracy → proportion of correct predictions.
. ROC-AUC → model’s ability to separate churn vs non-churn across thresholds.
. Confusion Matrix → detailed breakdown of true/false positives and negatives.

💾 Model Saving
. The full pipeline (preprocessing + logistic regression) was saved with:
joblib.dump(pipe, "model.pkl")
. This ensures consistent preprocessing and predictions when applied to new customer data.

🚀 How to Run
1. Clone this repo.

2. Install dependencies:
pip install -r requirements.txt

3. Run the script / notebook.
. Use the trained model for predictions:
import joblib
model = joblib.load("model.pkl")
preds = model.predict(new_data)

📌 Next Steps / Improvements
. Try other models (Random Forest, Gradient Boosting, XGBoost).
. Hyperparameter tuning for logistic regression.
. More feature engineering (e.g., interaction terms, customer lifetime value).
. Deploy as an API with FastAPI or Flask.



