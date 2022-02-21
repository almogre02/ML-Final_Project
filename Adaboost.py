from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split


class _Adaboost:

    def __init__(self, csv):
        """
        Adapt the data to the algorithm (convert parameters to intgers)
        """
        self.data = pd.read_csv(csv)
        self.mapping = {'Yes': "1", 'No': 0}
        self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
        self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]

        for i in self.data.keys():
            a=self.data[i]
            self.data[i]=self.data[i].fillna(0)

    def Q1(self):
        """
        We adapted the data to the algorithm,To get a logical result.
        we changed the section of alcohol consumption from 1-5 to high or low consumption.
        (4-5 high  consumption , 1-3 not high )
        Dalc - workday alcohol consumption
        Walc - weekend alcohol consumption
        """
        #self.data['Dalc'] = np.where(self.data.Dalc < 4, 0, 1)
        #self.data['Walc'] = np.where(self.data.Walc < 4, 0, 1)

        X = self.data[["FLAG_TSUNAMI",  "YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY",	"COUNTRY",
                       "LOCATION_NAME",	"LATITUDE",	"LONGITUDE",	"REGION_CODE",	"DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION"]]
        Y1 = self.data["FLAG_TSUNAMI"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            # Run adaboost with Dacl
            X_train, X_test, y_train, y_test = train_test_split(X, Y1, train_size=0.50, random_state=None)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, y_train)
            err = AdaBoost.score(X_test, y_test)
            sum += err

        print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)

