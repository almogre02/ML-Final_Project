from Adaboost import _Adaboost
from Svm import _Svm
from Regression import _Regression
from DesicionTree import _Decision_tree
from NN import _NN


EQ_adaboost = _Adaboost("Worldwide-Earthquake-Continent-database.csv")
EQ_svm = _Svm("Worldwide-Earthquake-database.csv")
EQ_LR = _Regression("Worldwide-Earthquake-database.csv")
EQ_Decision_tree = _Svm("Worldwide-Earthquake-database.csv")

EQ_NN = _NN("Worldwide-Earthquake-database.csv")


# print("\n\n~~~~~Q1~~~~~~~")
# print("Adaboost algorithm")
#EQ_adaboost.Q1()
# print("\n SVM algorithm")
# EQ_svm.Q1()
# print("\n LinearRegression algorithm")
# EQ_LR.Q1()
# print("\n DecisionTree algorithm")
#EQ_Decision_tree.Q1()

#print("\n\n~~~~~Q1~~~~~~~")
#print("NN algorithm")
#EQ_NN.Q1()

#print("\n\n~~~~~Q2~~~~~~")
#print("Adaboost algorithm")
#EQ_adaboost.Q2()
#print("\n SVM algorithm")
#EQ_svm.Q2()
# print("\n LinearRegression algorithm")
# EQ_LR.Q2()
# print("\n DecisionTree algorithm")
#EQ_Decision_tree.Q2()
# print("NN algorithm")
# EQ_NN.Q2()
#
# print("\n\n~~~~~Q3~~~~~~~")
# print("Adaboost algorithm")
EQ_adaboost.Q3()
# print("\n SVM algorithm")
# EQ_svm.Q3()
# print("\n LinearRegression algorithm")
# EQ_LR.Q3()
# print("\n DecisionTree algorithm")
# EQ_Decision_tree.Q3()
#
# print("\n\n~~~~~Q4~~~~~~~")
# print("Adaboost algorithm")
# EQ_adaboost.Q4()
# print("\n SVM algorithm")
# EQ_svm.Q4()
# print("\n LinearRegression algorithm")
# EQ_LR.Q4()
# print("\n DecisionTree algorithm")
# EQ_Decision_tree.Q4()









