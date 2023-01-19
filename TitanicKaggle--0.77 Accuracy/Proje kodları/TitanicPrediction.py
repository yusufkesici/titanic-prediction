import joblib
import pandas as pd


df = pd.read_csv("datasets/titanicData/test.csv")

df["Fare"] = df.groupby(["Sex", "Pclass"])["Fare"].transform(lambda x: x.fillna(x.mean()))


from TitanicPipeline import titanic_dataPrep

X = titanic_dataPrep(df)

X.drop("PassengerId", axis=1, inplace=True)

new_model = joblib.load("voting_clf5.pkl")

testPredicted = new_model.predict(X)

submission = pd.read_csv("datasets/titanicData/gender_submission.csv")

submission["Survived"] = testPredicted

submission.to_csv("SubmissionTitanic.csv", index=False)



