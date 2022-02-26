from sklearn.ensemble import AdaBoostClassifier
import numpy as np
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
        self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]
        self.data.CONTINENT = pd.factorize(self.data.CONTINENT)[0]


    #DataFrame.fillna()
        self.df=pd.DataFrame(self.data)
        self.df['COUNTRY'].fillna(0, inplace = True)
        self.df['LOCATION_NAME'].fillna(0, inplace = True)
        self.df=self.df.replace(to_replace="NaN",value=0)
        self.df = self.df.replace(np.nan, 0)
        self.data=self.df
        #self.data['POPULATION']=np.nan

        #for i in self.data.keys():
         #   a=self.data[i]
          #  self.data[i]=self.data[i].fillna(0)

#בהינתן רעידת אדמה - האם ייתכן צונאמי
    def Q1(self):
        """

        """
        X = self.data[["YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY", "COUNTRY",
                      "LOCATION_NAME","REGION_CODE",    "DEATHS",	"DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED",
                        "HOUSES_DAMAGED_DESCRIPTION","CONTINENT"]]
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

#הרוגים מול עוצמת הרעידה
    def Q2(self):

        # X = self.data[["DEATHS", "DEATHS_DESCRIPTION", "MISSING_DESCRIPTION", "INJURIES", "INJURIES_DESCRIPTION"]]
        # Y = self.data["INTENSITY"]
        X = self.data[["INTENSITY"]]
        Y = self.data["DEATHS"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            y_test_len=len(y_test)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, y_train)
            #result = AdaBoost.predict(X_test)
            #y_test = np.array(y_test)
            err = AdaBoost.score(X_test, y_test)
            sum += err

            # sucsses = 0
            # for i in range(len(y_test)):
            #     if result[i] <= y_test[i]  and result[i] >= y_test[i] :
            #         sucsses += 1
            # sum += sucsses / len(y_test)
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
            # Run adaboost with Dacl
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, random_state=None)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, y_train)
            err = AdaBoost.score(X_test, y_test)
            sum += err

        print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)

    # def Q4(self,csv):
    #     """
    #
    #     """
    #
    #     self.data = pd.read_csv(csv)
    #     self.mapping = {'Yes': "1", 'No': 0}
    #     self.data = self.data.replace(to_replace=['Yes', 'No'], value=[1, 0])
    #     X = self.data[["FLAG_TSUNAMI",	"YEAR",	"FOCAL_DEPTH",	"EQ_PRIMARY",	"INTENSITY",	"COUNTRY",
    #                    "LOCATION_NAME",	"LATITUDE",	"LONGITUDE",	"REGION_CODE",	"DEATHS",	"DEATHS_DESCRIPTION",
    #                    "MISSING_DESCRIPTION",	"INJURIES",	"INJURIES_DESCRIPTION",	"DAMAGE_DESCRIPTION",
    #                    "HOUSES_DESTROYED","HOUSES_DESTROYED_DESCRIPTION",	"HOUSES_DAMAGED"
    #         ,"HOUSES_DAMAGED_DESCRIPTION"]]
    #     X["POPULATION"]=np.nan
    #     X_new = X.iloc[3782:6195]
    #     data = pd.read_csv("WorldPopulation.csv")
    #     # data.head()
    #     df = pd.DataFrame(data);
    #     dfNew = df.rename(index=data['Country Name'])
    #     dfNew = dfNew.apply(lambda Y: Y.astype(str).str.upper())
    #     index = 3782
    #     # print(dfNew)
    #     # if dfNew['COUNTRY'] is not None:
    #     for i in range(len(dfNew)):
    #         country = X_new['COUNTRY'][index]
    #         year = X_new['YEAR'][index]
    #         # print(str(country))
    #         # print(str(year))
    #         c = str(country)
    #         #print(c)
    #         # lower_c=c.lower()
    #         # country=c[0]+lower_c[1:len(lower_c)]
    #         # print(country)
    #         y = str(year)
    #         #print(y)
    #         if c in dfNew.values:
    #             #print("##########")
    #             country = c.title()
    #             if c == ("USA"):
    #                 country = "United States"
    #             # print(country)
    #             # print(dfNew.loc[country,y])
    #             X_new['POPULATION'][index] = df[y][i]
    #             #print("@@@@@@@@@@@@")
    #         else:
    #             X_new['POPULATION'][index] = 0
    #         index += 1
    #
    #     print(X_new)
    #     self.data.COUNTRY = pd.factorize(self.data.COUNTRY)[0]
    #     self.data.LOCATION_NAME = pd.factorize(self.data.LOCATION_NAME)[0]
    #
    #     # DataFrame.fillna()
    #     self.df = pd.DataFrame(X_new)
    #     self.df['COUNTRY'].fillna(0, inplace=True)
    #     self.df['LOCATION_NAME'].fillna(0, inplace=True)
    #     self.df = self.df.replace(to_replace="NaN", value=0)
    #     self.df = self.df.replace(np.nan, 0)
    #     self.data = self.df
    #     self.data['POPULATION'] = np.nan
    #
    #     Y1 = self.data["FLAG_TSUNAMI"]
    #     rounds = 50
    #     sum = 0
    #     for round in range(rounds):
    #         # Run adaboost with Dacl
    #         X_train, X_test, y_train, y_test = train_test_split(X, Y1, train_size=0.50, random_state=None)
    #         AdaBoost = AdaBoostClassifier()
    #         AdaBoost.fit(X_train, y_train)
    #         err = AdaBoost.score(X_test, y_test)
    #         sum += err
    #
    #     print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)