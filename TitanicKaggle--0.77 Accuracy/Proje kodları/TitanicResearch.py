import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, \
    AdaBoostRegressor, RandomForestClassifier, VotingClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from helpers.eda import *
from helpers.data_prep import *
import missingno as msno
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, recall_score, roc_curve, \
    r2_score, mean_squared_error, precision_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
import missingno as msno

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 800)

# pd.set_option("display.max_rows", 200)
df = pd.read_csv("datasets/titanic.csv")


def titanic_dataPrep(df):
    df.drop("PassengerId", axis=1, inplace=True)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin_NA_FLAG"] = np.where(df["Cabin"].isnull(), 1, 0)

    ticket_c = [col for col in df["Ticket"] if col in ["CA. 2343", "347082", "1601",
                                                       "347088", "CA 2144", "3101295",
                                                       "S.O.C. 14879", "382652", "PC 1775",
                                                       "347077"]]

    df["Ticket_10"] = np.where(df["Ticket"].isin(ticket_c), 1, 0)
    df["Age_NA_FLAG"] = np.where(df["Age"].isnull(), 1, 0)

    df["Age"] = df.groupby(["Sex", "Pclass"])[["Age"]].transform(lambda x: x.fillna(x.mean()))

    df.drop("Cabin", axis=1, inplace=True)

    # Cardinal değişken işe yaramaz
    df.drop("Ticket", axis=1, inplace=True)

    low_limit, up_limit = outlier_thresholds(df, "Fare", 0.05, 0.95)
    df.loc[(df["Fare"] < low_limit), "Fare"] = low_limit
    df.loc[(df["Fare"] > up_limit), "Fare"] = up_limit

    # FEATURE ENGİNEERİNG

    def nameMethod(str):
        returnName = str.split(",")[1].split(".")[0].strip(" ")
        return returnName

    df["NameTitle"] = df["Name"].apply(lambda x: nameMethod(x))
    df["NameLength"] = df["Name"].apply(lambda x: len(x))

    df["Female_Class1_2"] = np.where((df["Sex"] == "female") & (df["Pclass"] <= 2), 1, 0)
    df["Male_Class3"] = np.where((df["Sex"] == "male") & (df["Pclass"] == 3), 1, 0)
    df["EmbarkedS_Pclass1"] = np.where((df["Embarked"] == "S") & (df["Pclass"] == 1), 1, 0)
    df["Male_Fare_Less20"] = np.where((df["Sex"] == "male") & (df["Fare"] < 20), 1, 0)
    df["Female_Fare_More30"] = np.where((df["Sex"] == "female") & (df["Fare"] > 30), 1, 0)

    df[(df["Pclass"] == 2) & (df["Sex"] == "female")]["Survived"].mean()

    df["FamiliySize"] = df["SibSp"] + df["Parch"]
    df["Is_Alone"] = np.where(df["FamiliySize"] == 0, 1, 0)

    df["NameLen*Pclass"] = df["NameLength"] * df["Pclass"]

    df.loc[((df["Sex"] == "male") & (df["Is_Alone"] == 0)), "Male_Alone"] = 1
    df.loc[~((df["Sex"] == "male") & (df["Is_Alone"] == 0)), "Male_Alone"] = 0

    df.loc[df["Age"] <= 11, "Age_Cat"] = "Kid"
    df.loc[(df["Age"] > 11) & (df["Age"] <= 21), "Age_Cat"] = "Young"
    df.loc[(df["Age"] > 21) & (df["Age"] <= 31), "Age_Cat"] = "Middle"
    df.loc[(df["Age"] > 31) & (df["Age"] <= 45), "Age_Cat"] = "Mature"
    df.loc[(df["Age"] > 45) & (df["Age"] <= 80), "Age_Cat"] = "Senior"

    df.loc[((df["Cabin_NA_FLAG"] == 1) & (df["Age_NA_FLAG"] == 1)), "Cabin_Age_NA_FLAG"] = 1
    df.loc[~((df["Cabin_NA_FLAG"] == 1) & (df["Age_NA_FLAG"] == 1)), "Cabin_Age_NA_FLAG"] = 0

    df["Pclass*Age"] = df["Pclass"] * df["Age"]
    df["NameLength*Age"] = df["NameLength"] * df["Age"]

    df["Fare/Age"] = df["Fare"] / df["Age"]

    df["FamilySize*Fare"] = df["FamiliySize"] * df["Fare"]

    df["Pcls*Fare/Age"] = (df["Pclass"] * df["Fare"]) / df["Age"]

    df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

    #### ENCODİNG & SCALING
    df.drop("Name", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    df["Male_Alone"] = df["Male_Alone"].astype(int)
    df["Cabin_Age_NA_FLAG"] = df["Cabin_Age_NA_FLAG"].astype(int)
    ohe_cols = ["Sex", "Embarked", "NameTitle", "Age_Cat"]

    df = one_hot_encoder(df, ohe_cols, True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y


X, y = titanic_dataPrep(df)


def base_models(X, y):
    print("Base Models....")
    regressors = [('LR', LogisticRegression()),
                  ('KNN', KNeighborsClassifier()),
                  ("CART", DecisionTreeClassifier()),
                  ("RF", RandomForestClassifier()),
                  ('Ridge', RidgeClassifier()),
                  ('GBM', GradientBoostingClassifier()),
                  ('XGBoost', XGBClassifier(objective='reg:squarederror')),
                  ('LightGBM', LGBMClassifier()),
                  # ('CatBoost', CatBoostClassifier(verbose=False))
                  ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=5, scoring="accuracy")
        print(f"accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


#### BASE MODEL ACCURACY SCORES
# accuracy: 0.7958 (XGBoost)
# accuracy: 0.8249 (LightGBM)
# accuracy: 0.8002 (LR)
# accuracy: 0.6892 (KNN)
# accuracy: 0.7722 (CART)
# accuracy: 0.798 (RF)
# accuracy: 0.7991 (Adaboost)
# accuracy: 0.8103 (GBM)


### AFTER FEATURE ENGİNERRİNG AND DATA PRE-PROCESSİNG


# accuracy: 0.8221 (LR)
# accuracy: 0.794 (KNN)
# accuracy: 0.7522 (CART)
# accuracy: 0.8097 (RF)
# accuracy: 0.8165 (Adaboost)
# accuracy: 0.8344 (GBM)
# accuracy: 0.8029 (XGBoost)
# accuracy: 0.8097 (LightGBM)


######################################################
# 4. Automated Hyperparameter Optimization
######################################################

# accuracy: 0.8221 (LR)
# accuracy: 0.794 (KNN)
# accuracy: 0.7522 (CART)
# accuracy: 0.8097 (RF)
# accuracy: 0.8165 (Adaboost)
# accuracy: 0.8344 (GBM)
# accuracy: 0.8029 (XGBoost)
# accuracy: 0.8097 (LightGBM)

RidgeClassifier_params = {'alpha': [1.0, 20, 50, 100, 200]}

logistic_params = {'max_iter': [100, 200, 500],
                   "solver": ['liblinear']
                   }

GBM_params = {"n_estimators": [5, 50, 100, 250, 500],
              "max_depth": [1, 3, 5, 7, 9],
              "learning_rate": [0.01, 0.1, 1, 10]}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 15, 20],
             "n_estimators": [200, 300]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "subsample": [1.0, 0.9, 0.5, 0.3]}

regressors = [('LR', LogisticRegression(), logistic_params),
              ('Ridge', RidgeClassifier(), RidgeClassifier_params),
              ('GBM', GradientBoostingClassifier(), GBM_params),
              ('LGBM', LGBMClassifier(), lightgbm_params),
              ('RF', RandomForestClassifier(), rf_params),
              ]

best_models = {}
for name, regressor, params in regressors:
    print(f"########## {name} ##########")

    cv_results = cross_validate(regressor, X, y, cv=3, scoring="accuracy")
    print(f"Accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
    final_model = regressor.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=3, scoring="accuracy")
    print(f"Accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
    best_models[name] = final_model


# accuracy: 0.8221 (LR)
# accuracy: 0.8277 (RF)
# accuracy: 0.8266 (GBM)
# accuracy: 0.8198 (Ridge)
# accuracy: 0.8311 (XGBoost)
# accuracy: 0.8311 (LightGBM)


######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Regressor...")

    voting_clf = VotingClassifier(estimators=[('LGBM', best_models["LGBM"]),
                                              ('RF', best_models["RF"]),
                                              ('LR', best_models["LR"])]).fit(X, y)

    cv_results = cross_validate(final_model, X, y, cv=3, scoring="accuracy")
    print(f"accuracy: {round(cv_results['test_score'].mean(), 5)} ")
    return voting_clf


voting_clf = voting_classifier(best_models, X, y)

######################################################
# 6. Prediction for a New Observation
######################################################
import joblib

X.columns
random_user = X.sample(1, random_state=45)

voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)



