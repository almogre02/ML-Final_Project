import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class _Regression:

    def __init__(self, csv):
        """
        Adapt the data to the algorithm (convert parameters to intgers)
        """
        self.data = pd.read_csv(csv)
        self.mapping = {'Yes': "1", 'No': 0}
        self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
        self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]
        self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]


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
                        "HOUSES_DAMAGED_DESCRIPTION"]]
        Y1 = self.data["FLAG_TSUNAMI"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            # Run adaboost with Dacl
            X_train, X_test, y_train, y_test = train_test_split(X, Y1, train_size=0.50, random_state=None)
            LR = LinearRegression()
            LR.fit(X_train, y_train)
            err = LR.score(X_test, y_test)
            sum += err

        print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)
