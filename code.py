# Učitavanje paketa i biblioteka
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings("ignore")

# Učitavanje podataka
dataframe = pd.read_csv('diabetes_prediction_dataset.csv')

# Pregled nekoliko prvih redova
print(dataframe.head())

# Provjera nedostajućih vrijednosti
missing_values = dataframe.isnull().sum()
print("Nedostajuće vrijednosti po kolonama:\n", missing_values)

# Popunjavanje nedostajućih vrijednosti
dataframe.fillna(dataframe.mean(numeric_only=True), inplace=True)
dataframe.fillna(dataframe.mode().iloc[0], inplace=True)
dataframe.fillna("Unknown", inplace=True)

# Pretvaranje string kolona u numeričke vrijednosti
dataframe = pd.get_dummies(dataframe, columns=['gender', 'smoking_history'], drop_first=True)


# Regulisanje anomalija
def correct_anomalies(df):
    thresholds = {
        'age': (0, 80),
        'bmi': (10.16, 71.55),
        'HbA1c_level': (0, 20),
        'blood_glucose_level': (0, 500)
    }
    corrected_data = pd.DataFrame()
    indices_to_drop = set()
    for col, (lower, upper) in thresholds.items():
        anomalies = df[(df[col] < lower) | (df[col] > upper)].copy()
        if not anomalies.empty:
            anomalies['distance'] = anomalies[col].apply(lambda x: min(abs(x - lower), abs(x - upper)))
            num_to_drop = int(0.02 * anomalies.shape[0])
            if num_to_drop > 0:
                to_drop = anomalies.nlargest(num_to_drop, 'distance')
                indices_to_drop.update(to_drop.index)
                anomalies = anomalies.drop(to_drop.index)
            if not anomalies.empty:
                anomalies[col] = anomalies[col].apply(lambda x: lower if x < lower else upper)
                corrected_data = pd.concat([corrected_data, anomalies])
    df = df.drop(indices_to_drop, axis=0)
    df = df[~df.index.isin(corrected_data.index)]
    df = pd.concat([df, corrected_data])
    df = df.drop(['distance'], axis=1, errors='ignore')
    return df


# Poziv funkcije za regulisanje anomalija
dataframe = correct_anomalies(dataframe)

# Vizualizacija podataka

# a) Matrica korelacija
correlation_matrix = dataframe.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelaciona matrica atributa')
plt.show()

# b) Histogram raspodjele starosne dobi po dijabetesu
plt.figure(figsize=(10, 6))
sns.histplot(data=dataframe, x='age', hue='diabetes', kde=True)
plt.title('Histogram raspodjele starosne dobi po dijabetesu')
plt.show()

# c) PieChart za raspodjelu dijabetesa
diabetes_counts = dataframe['diabetes'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(diabetes_counts, labels=diabetes_counts.index.map({0: 'No Diabetes', 1: 'Diabetes'}), autopct='%1.1f%%',
        startangle=90, colors=['#66b3ff', '#ff9999'])
plt.title('Raspodjela dijabetesa')
plt.show()

# d) Linijski grafik za HbA1c_level u odnosu na dijabetes
plt.figure(figsize=(10, 6))
sns.lineplot(data=dataframe, x='HbA1c_level', y='diabetes')
plt.title('HbA1c nivo u odnosu na dijabetes')
plt.show()

# e) Boks-plot raspodjele BMI vrijednosti u odnosu na dijabetes
plt.figure(figsize=(8, 8))
sns.boxplot(data=dataframe, x='diabetes', y='bmi')
plt.title('BMI u odnosu na dijabetes')
plt.show()

# Standardizacija numeričkih kolona
scaler = StandardScaler()
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])

# Razdvajanje features-a i ciljne promjenljive
X = dataframe.drop('diabetes', axis=1)
y = dataframe['diabetes']

# Primjena SMOTETomek metode
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Podjela skupa podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42,
                                                    stratify=y_resampled)


# Definisanje baznih modela za Stacking
def define_base_models(optimalK=5):
    knn = KNeighborsClassifier(n_neighbors=optimalK)
    base_models = [
        ('KNN', knn),
        ('Decision Tree', DecisionTreeClassifier(random_state=42))
    ]
    return base_models


# Funkcija za iscrtavanje metode lakta
def plot_elbow_method(X_train, y_train, k_range=range(1, 21), cv_folds=5):
    average_errors = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        average_errors.append(1 - np.mean(scores))
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, average_errors, marker='o', linestyle='dashed', color='b')
    plt.xlabel('Broj susjeda (K)')
    plt.ylabel('Prosječna greška')
    plt.title('Metoda lakta za KNN')
    plt.xticks(k_range)
    plt.show()

plot_elbow_method(X_train, y_train)
optimal_k = 5  # Odabrano na osnovu grafika

# Kreiranje baznih modela za Stacking
base_models_stacking = define_base_models(optimal_k)

# Definisanje baznog modela za Bagging
base_model_bagging = DecisionTreeClassifier(max_depth = 3, criterion='gini')

# Treniranje Random Forest modela za selekciju atributa
rfm = RandomForestClassifier(n_estimators=100, random_state=42)
rfm.fit(X_train, y_train)
importances = rfm.feature_importances_
feature_importances_dataframe = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Hiperparametrizacija
param_grid_stacking = {
    'final_estimator__C': [0.1, 1.0]
}
param_grid_bagging = {
    'max_samples': [0.8, 1.0]
}
param_grid_gb = {
    'learning_rate': [0.05, 0.1]
}
param_grid_reg = {
    'C': [0.1, 0.5, 1.0]
}

models_hyperparams = [
    ('Stacking', StackingClassifier(estimators=base_models_stacking, final_estimator=LogisticRegression(), cv=5),
     param_grid_stacking),
    ('Bagging', BaggingClassifier(estimator=base_model_bagging, n_estimators=100, random_state=42), param_grid_bagging),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
     param_grid_gb),
    ('Logistic Regression', LogisticRegression(), param_grid_reg)
]


def find_best_model(models_hyperparams):
    best_models = {}
    for model_name, model, hyperparams in models_hyperparams:
        grid_search = GridSearchCV(model, hyperparams, cv=5)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
    return best_models


best_models = find_best_model(models_hyperparams)

# Selekcija atributa
print("Važnost atributa:\n", feature_importances_dataframe)

rfe_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_features_range = list(range(1, len(X_train.columns) + 1))

for n_features in num_features_range:
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=n_features)
    scores = cross_val_score(rfe, X_train, y_train, cv=cv, scoring='accuracy')
    rfe_scores.append(np.mean(scores))

plt.figure(figsize=(10, 6))
plt.plot(num_features_range, rfe_scores, marker='o', linestyle='dashed', color='b')
plt.xlabel('Broj selektovanih atributa')
plt.ylabel('Prosječna tačnost (5-fold CV)')
plt.title('Metoda lakta za odabir broja selektovanih atributa')
plt.show()

optimal_num_features = 6

final_rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                n_features_to_select=optimal_num_features)
final_rfe.fit(X_train, y_train)

# Dobijanje selektovanih atributa
selected_features = X_train.columns[final_rfe.support_]
print("Selektovani atributi:", list(selected_features))

# Kreiranje podskupova sa selektovanim atributima
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Evaluacija modela na podacima sa svim i sa selektovanim atributima
def evaluate_model(model, X_train, X_test, y_train, y_test, label=""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- Evaluacija modela: {label} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# Iscrtavanje ROC krive za sve modele
plt.figure(figsize=(10, 8))

for name, model in best_models.items():
    evaluate_model(model, X_train, X_test, y_train, y_test, label=f"{name} (svi atributi)")
    evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test, label=f"{name} (selektovani atributi)")

# Finalno prikazivanje ROC krive
plt.plot([0, 1], [0, 1], 'k--')  # dijagonala
plt.title('ROC Krive za sve modele')
plt.xlabel('Lažno pozitivna stopa')
plt.ylabel('Istinito pozitivna stopa')
plt.legend(loc='lower right')
plt.grid()
plt.show()



