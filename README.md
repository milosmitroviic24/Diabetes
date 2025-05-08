# Diabetes Prediction using Machine Learning

This project aims to predict the likelihood of a person having diabetes based on various health and demographic attributes. The workflow includes data preprocessing, visualization, anomaly correction, feature selection, data balancing, model training with hyperparameter tuning, and final evaluation using multiple ensemble methods.

## 📁 Dataset

- **Source:** `diabetes_prediction_dataset.csv`
- **Target variable:** `diabetes` (0 = No Diabetes, 1 = Diabetes)
- **Features include:** Age, BMI, HbA1c level, blood glucose level, gender, and smoking history.

## 🧪 Workflow

### 1. **Data Preprocessing**
- Handling missing values using mean, mode, and 'Unknown' for categorical columns.
- One-hot encoding for categorical variables (`gender`, `smoking_history`).
- Anomaly correction in key numerical features (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`).

### 2. **Exploratory Data Analysis (EDA)**
- Correlation heatmap to understand feature relationships.
- Histograms, pie charts, line plots, and box plots to analyze distributions and trends.

### 3. **Data Transformation**
- Standardization of numerical features using `StandardScaler`.
- Handling class imbalance using **SMOTETomek** technique.

### 4. **Feature Selection**
- Feature importance ranking using `RandomForestClassifier`.
- Recursive Feature Elimination (RFE) with elbow method to determine optimal number of features.

### 5. **Modeling**
Trained and evaluated the following models with hyperparameter tuning using `GridSearchCV`:

- **Stacking Classifier**  
  Base models: KNN (optimal K from elbow method), Decision Tree  
  Meta model: Logistic Regression

- **Bagging Classifier**  
  Base model: Decision Tree

- **Gradient Boosting Classifier**

- **Logistic Regression**

## 📊 Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve & AUC

## 🛠 Technologies Used

- **Python 3**
- **Pandas**, **NumPy**, **Seaborn**, **Matplotlib**
- **Scikit-learn** for ML models and preprocessing
- **Imbalanced-learn** for data balancing

## 📈 Results

- All models were trained on the resampled dataset.
- Feature selection and hyperparameter tuning improved model performance.
- The best-performing models achieved **over 90% accuracy**.

## 📎 Notes

- The elbow method was used to determine both optimal K for KNN and optimal number of selected features for RFE.
- Model interpretability was supported by analyzing feature importance scores.
