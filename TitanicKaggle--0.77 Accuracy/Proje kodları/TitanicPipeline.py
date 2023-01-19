import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV


pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 800)


def titanic_dataPrep(df):

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

    #### ENCODİNG & SCALING
    df.drop("Name", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    df["Male_Alone"] = df["Male_Alone"].astype(int)
    df["Cabin_Age_NA_FLAG"] = df["Cabin_Age_NA_FLAG"].astype(int)

    for var in ["NameTitle"]:
        tmp = df[var].value_counts() / len(df)
        rare_labels = tmp[tmp < 0.1].index
        df[var] = np.where(df[var].isin(rare_labels), 'Rare', df[var])

    ohe_cols = ["Sex", "Embarked", "NameTitle", "Age_Cat"]

    df = one_hot_encoder(df, ohe_cols, True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])


    return df


def base_models(X, y):
    print("Base Models....")
    regressors = [('SVC', SVC()),
                  ('KNN', KNeighborsClassifier()),
                  ("CART", DecisionTreeClassifier()),
                  ("RF", RandomForestClassifier()),
                  ('Ridge', RidgeClassifier()),
                  ('GBM', GradientBoostingClassifier()),
                  ('LightGBM', LGBMClassifier()),
                  ('CatBoost', CatBoostClassifier(verbose=False))
                  ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring="accuracy")
        print(f"accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


RidgeClassifier_params = {'alpha': [1.0, 20, 50, 100, 200]}

SVC_params = {'C': [0.1, 2, 3, 5]}

GBM_params = {"n_estimators": [5, 50, 100, 250, 400, 800],
              "max_depth": [1, 3, 5, 7, 9],
              "learning_rate": [0.01, 0.1, 1, 0.001]}

rf_params = {"max_depth": [3, 5, 8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 15, 20],
             "n_estimators": [200, 300, 500]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.02],
                   "n_estimators": [300, 500, 1000, 2000],
                   "subsample": [1.0, 0.9, 0.5, 0.3]}

CatBoost_params = {"iterations": [100, 200, 500],
                   "learning_rate": [0.1, 0.03,0.1],
                   "depth": [3, 5, 8]}

regressors = [('SVC', SVC(), SVC_params),
              ('Ridge', RidgeClassifier(), RidgeClassifier_params),
              ('GBM', GradientBoostingClassifier(), GBM_params),
              ('LGBM', LGBMClassifier(), lightgbm_params),
              ('RF', RandomForestClassifier(), rf_params),
              ('CatBoost', CatBoostClassifier(verbose=False), CatBoost_params)]


def hyperparameter_optimization(X, y, cv=3):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=3, scoring="accuracy")
        print(f"Accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=3, scoring="accuracy")
        print(f"Accuracy: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, X, y):
    print("Voting Regressor...")

    voting_clf = VotingClassifier(estimators=[('CatBoost', best_models["CatBoost"]),
                                              ('RF', best_models["RF"]),
                                              ('GBM', best_models["GBM"]),
                                              ('SVC', best_models["SVC"])]).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring="accuracy")
    print(f"Accuracy: {round(cv_results['test_score'].mean(), 5)} ")
    return voting_clf


def main():
    df = pd.read_csv("datasets/titanicData/train.csv")
    df = titanic_dataPrep(df)
    X = df.drop(["Survived","PassengerId"], axis=1)
    y = df["Survived"]
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf5.pkl")
    return voting_clf


if __name__ == "__main__":
    print("Processing...")
    main()

pd.read_csv("SubmissionTitanic.csv")