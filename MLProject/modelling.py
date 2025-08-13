import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("./diabetes_prediction_preprocessing/diabetes_train.csv")
df_test = pd.read_csv("./diabetes_prediction_preprocessing/diabetes_test.csv")

X_train = df_train.drop(columns=['diabetes'])
y_train = df_train['diabetes']

X_test = df_test.drop(columns=['diabetes'])
y_test = df_test['diabetes']

with mlflow.start_run():
    mlflow.autolog()

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
