from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix


class _Adaboost:

    def __init__(self, csv):
        self.data = pd.read_csv(csv)
        self.mapping = {'Yes': "1", 'No': 0}
        self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
        self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]
        self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]
        self.data.CONTINENT = pd.factorize(self.data.CONTINENT)[0]

        self.df = pd.DataFrame(self.data)
        self.df['COUNTRY'].fillna(0, inplace=True)
        self.df['LOCATION_NAME'].fillna(0, inplace=True)
        self.df = self.df.replace(to_replace="NaN", value=0)
        self.df = self.df.replace(np.nan, 0)
        self.data = self.df

    def normalization(self, X):
        x = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        X_norm = pd.DataFrame(x_scaled)
        return X_norm

    #לחזות האם יהיה צונאמי בהינתן רעידת אדמה ועל סמך פיצ'רים נוספים
    def Q1(self):
        X = self.data[["YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY", "COUNTRY",
                      "LOCATION_NAME",  "REGION_CODE",    "DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED",  "HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION",   "CONTINENT"]]
        X_norm = self.normalization(X)
        Y = self.data["FLAG_TSUNAMI"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, train_size=0.70, random_state=None)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, y_train)
            score = AdaBoost.score(X_test, y_test)
            sum += score
        print("Adaboost accuracy: ", sum / rounds)

        plot_confusion_matrix(AdaBoost, X_test, y_test)
        plt.show()

    def Q2(self):
        X = self.data[["FLAG_TSUNAMI",  "YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY", "COUNTRY",
                      "LOCATION_NAME",  "DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION", "CONTINENT"]]
        X_norm = self.normalization(X)
        Y = self.data["INTENSITY"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, train_size=0.70, random_state=None)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, y_train)
            y_pred = AdaBoost.predict(X_test)
            score = AdaBoost.score(X_test, y_test)
            sum += score
        print("Adaboost accuracy: ", sum / rounds)

        confusion_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(16, 9))
        sns.heatmap(confusion_mat, cmap="Blues", annot=True,
                    xticklabels=np.arange(12),
                    yticklabels=np.arange(12));
        plt.show()
