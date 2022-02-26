from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split


class _Svm:

    def __init__(self, csv):
        """
        Adapt the data to the algorithm (convert parameters to intgers)
        """
        self.data = pd.read_csv(csv)
        self.mapping = {'Yes': "1", 'No': 0}
        self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
        self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]
        self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]
        self.data.CONTINENT = pd.factorize(self.data.CONTINENT)[0]


    #DataFrame.fillna()
        self.df=pd.DataFrame(self.data)
        self.df['COUNTRY'].fillna(0, inplace = True)
        self.df['LOCATION_NAME'].fillna(0, inplace = True)
        self.df=self.df.replace(to_replace="NaN",value=0)
        self.df = self.df.replace(np.nan, 0)
        self.data=self.df

        #for i in self.data.keys():
         #   a=self.data[i]
          #  self.data[i]=self.data[i].fillna(0)

    def Q1(self):
        """

        """
        #self.data['Dalc'] = np.where(self.data.Dalc < 4, 0, 1)
        #self.data['Walc'] = np.where(self.data.Walc < 4, 0, 1)

        X = self.data[["YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY", "COUNTRY",
                      "LOCATION_NAME","REGION_CODE",    "DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION","CONTINENT"]]
        Y = self.data["FLAG_TSUNAMI"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            # Run svm to predict workday alcohol consumption
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            Accuracy = clf.score(X_test, y_test)
            sum += Accuracy

        print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)


    def Q2(self):

        # X = self.data[["DEATHS", "DEATHS_DESCRIPTION", "MISSING_DESCRIPTION", "INJURIES", "INJURIES_DESCRIPTION"]]
        # Y = self.data["INTENSITY"]
        X = self.data[["INTENSITY"]]
        Y = self.data["DEATHS"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            # Run svm to predict workday alcohol consumption
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            Accuracy = clf.score(X_test, y_test)
            sum += Accuracy

        print("The success rate is:", sum / rounds)


    def Q3(self):
        X = self.data[["FLAG_TSUNAMI",  "YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY", "COUNTRY",
                      "LOCATION_NAME",  "DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION"]]
        Y = self.data["CONTINENT"]

        rounds = 60
        sum = 0


        for round in range(rounds):
            # Run svm to predict workday alcohol consumption
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            Accuracy = clf.score(X_test, y_test)
            sum += Accuracy

        print("The success rate is:", sum / rounds)
