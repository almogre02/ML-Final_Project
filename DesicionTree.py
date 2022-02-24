from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz


class _Decision_tree:

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

        #for round in range(rounds):
        X_train, X_test, y_train, y_test = train_test_split(X, Y1, train_size=0.50, random_state=None)
        clf = tree.DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)
        prediction=clf.predict(X_test)
        print(accuracy_score(y_test,prediction))
        # class_names=list(set(X.iloc[:,-1]))
        # dot_data=tree.export_graphviz(clf,out_file=None, feature_names=X.keys(),class_names=class_names,filled=True)
        # graph=graphviz.Source(dot_data)
        # graphviz.render(graph)
        #     err = decision_tree.score(X_test, y_test)
        #     sum += err
        #
        # print("Accuracy of the possibility for Tsunami when Earthquake is occur : ", sum / rounds)

    def Q2(self):
        # X = self.data[["DEATHS", "DEATHS_DESCRIPTION", "MISSING_DESCRIPTION", "INJURIES", "INJURIES_DESCRIPTION"]]
        # Y = self.data["INTENSITY"]
        X = self.data[["INTENSITY"]]
        Y = self.data["DEATHS"]
        rounds = 50
        sum = 0

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
        clf = tree.DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        print(accuracy_score(y_test, prediction))
