import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
data=pd.read_csv("Employee.csv")
df=pd.DataFrame(data)
target_col ='Attrition'       
df = df.dropna(subset=[target_col]).reset_index(drop=True)
x = df.drop(columns=[target_col])
y = df[target_col].map({'No': 0, 'Yes': 1})  
print(df)

numericaltransformer=Pipeline(steps=[
    ('num_imputer',SimpleImputer(strategy="mean")),
    ('scaling',StandardScaler())
])
categoricaltransformer=Pipeline(steps=[
    ('cat_impuer',SimpleImputer(strategy="most_frequent")),
    ('encoding',OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor=ColumnTransformer(transformers=[
    ('scaling',numericaltransformer,['YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager',
                        'YearsAtCompany','WorkLifeBalance','TrainingTimesLastYear','TotalWorkingYears','StockOptionLevel',
                        'RelationshipSatisfaction','PerformanceRating','PercentSalaryHike','NumCompaniesWorked','MonthlyRate','MonthlyIncome',
                        'JobSatisfaction','JobLevel','JobInvolvement','HourlyRate','EnvironmentSatisfaction',
                        'Education','DistanceFromHome','DailyRate','Age']),
    ('encoding',categoricaltransformer,['BusinessTravel','Department','MaritalStatus','EducationField','Gender','JobRole'])])
pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
     ('smote', SMOTE(random_state=42)),
    ("model",LogisticRegression())
])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
pipeline.fit(x_train, y_train)
y_pred=pipeline.predict(x_test)
a=accuracy_score(y_test,y_pred)
print(a)
print(classification_report(y_test,y_pred))
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}
hyperperameter=GridSearchCV(estimator=pipeline,
                 cv=5,
                 param_grid=param_grid,
                 scoring='accuracy',
                 n_jobs=-1)
hyperperameter.fit(x_train, y_train)
print(hyperperameter)
hyperperameter.fit(x_train, y_train)
best_model = hyperperameter.best_estimator_
y_prob = best_model.predict_proba(x_test)[:, 1]
final_threshold= 0.45
y_pred_final = (y_prob >= final_threshold).astype(int)
print(classification_report(
    y_test,
    y_pred_final,
    target_names=['No', 'Yes'],
    zero_division=0
))
joblib.dump(best_model, "attrition_model_pipeline.joblib")
print("Model saved successfully")
