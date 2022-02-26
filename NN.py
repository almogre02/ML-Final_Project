from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split


class _NN:

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

        self.df = pd.DataFrame(self.data)
        self.df['COUNTRY'].fillna(0, inplace=True)
        self.df['LOCATION_NAME'].fillna(0, inplace=True)
        self.df = self.df.replace(to_replace="NaN", value=0)
        self.df = self.df.replace(np.nan, 0)
        self.data = self.df
        #self.data['POPULATION'] = np.nan

    def Q1(self):

        X = self.data[["YEAR", "FOCAL_DEPTH", "EQ_PRIMARY", "INTENSITY", "COUNTRY",
                       "LOCATION_NAME", "REGION_CODE", "DEATHS", "DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION", "INJURIES", "INJURIES_DESCRIPTION", "DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED", "HOUSES_DESTROYED_DESCRIPTION", "HOUSES_DAMAGED",
                       "HOUSES_DAMAGED_DESCRIPTION"]]
        Y1 = self.data["FLAG_TSUNAMI"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y1, train_size=0.50, random_state=None)
            NN = MLPClassifier(random_state=1, max_iter=400, activation='relu', hidden_layer_sizes=(100, 2),
                               early_stopping=True)
            #relu: the rectified linear unit function, returns f(x) = max(0, x)
            NN.fit(X_train.values, y_train.values)
            success = NN.score(X_test.values, y_test.values)
            sum += success

        print("Accuracy of the possibility for Tsunami when Earthquake is occur: ", sum / rounds)

    def Q2(self):

        X = self.data[["INTENSITY"]]
        Y = self.data["DEATHS"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            NN = MLPClassifier(random_state=1, max_iter=500, activation='relu', hidden_layer_sizes=(200, 2),
                               early_stopping=True)
            # relu: the rectified linear unit function, returns f(x) = max(0, x)
            NN.fit(X_train.values, y_train.values)
            success = NN.score(X_test.values, y_test.values)
            sum += success

        print("", sum / rounds)


    def Q3(self):

        X = self.data[["FLAG_TSUNAMI", "YEAR", "FOCAL_DEPTH", "EQ_PRIMARY", "INTENSITY", "COUNTRY",
                       "LOCATION_NAME", "DEATHS", "DEATHS_DESCRIPTION",
                       "MISSING_DESCRIPTION", "INJURIES", "INJURIES_DESCRIPTION", "DAMAGE_DESCRIPTION",
                       "HOUSES_DESTROYED", "HOUSES_DESTROYED_DESCRIPTION", "HOUSES_DAMAGED",
                       "HOUSES_DAMAGED_DESCRIPTION"]]
        Y = self.data["CONTINENT"]
        rounds = 60
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            NN = MLPClassifier(random_state=2, max_iter=500, activation='logistic', hidden_layer_sizes=(100, 2),
                               early_stopping=True)
            # relu: the rectified linear unit function, returns f(x) = max(0, x)
            NN.fit(X_train.values, y_train.values)
            success = NN.score(X_test.values, y_test.values)
            sum += success

        print("", sum / rounds)