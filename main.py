from Adaboost import _Adaboost
from Svm import _Svm
from Regression import _Regression
from DesicionTree import _Decision_tree
from NN import _NN
from LassoRegression import lASSO_Regression


EQ_adaboost = _Adaboost("Worldwide-Earthquake-Continent-database.csv")
EQ_svm = _Svm("Worldwide-Earthquake-Continent-database.csv")
EQ_Decision_tree = _Decision_tree("Worldwide-Earthquake-Continent-database.csv")
EQ_NN = _NN("Worldwide-Earthquake-Continent-database.csv")
EQ_LinearR = _Regression("Worldwide-Earthquake-Continent-database.csv")
EQ_lassoR = lASSO_Regression("Worldwide-Earthquake-Continent-database.csv")

print("\n\nClassification:")
print("\n\nQuestion 1:")
print("Adaboost")
EQ_adaboost.Q1()
EQ_svm.Q1()
EQ_Decision_tree.Q1()
EQ_NN.Q1()

print("\n\nQuestion 2:")
EQ_adaboost.Q2()
EQ_svm.Q2()
EQ_Decision_tree.Q2()
EQ_NN.Q2()

print("\n\nRegression:")
print("\n\nQuestion 3:")
EQ_LinearR.Q3()
EQ_lassoR.Q3()
