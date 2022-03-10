import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import seaborn as sns


class lASSO_Regression:

    def __init__(self, csv):
        self.data = pd.read_csv(csv)
        self.mapping = {'Yes': "1", 'No': 0}
        self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
        self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]
        self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]

        self.df=pd.DataFrame(self.data)
        self.df['COUNTRY'].fillna(0, inplace = True)
        self.df['LOCATION_NAME'].fillna(0, inplace = True)
        self.df=self.df.replace(to_replace="NaN",value=0)
        self.df = self.df.replace(np.nan, 0)
        self.data=self.df

    def Q3(self):
        X = self.df[["FLAG_TSUNAMI", "YEAR", "FOCAL_DEPTH", "EQ_PRIMARY", "INTENSITY", "COUNTRY",
                     "LOCATION_NAME", "REGION_CODE", "DEATHS_DESCRIPTION"]]
        Y = self.df["DEATHS"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.70, random_state=None)
            LR = Lasso()
            LR.fit(X_train, y_train)
            score = LR.score(X_test, y_test)
            sum += score

        print("lASSO accuracy: ", sum / rounds)

        sns.lmplot("DEATHS", "INTENSITY", self.df)
        plt.show()
