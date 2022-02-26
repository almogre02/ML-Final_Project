from Adaboost import _Adaboost
from Svm import _Svm
from Regression import _Regression
from DesicionTree import _Decision_tree
from NN import _NN


EQ_adaboost = _Adaboost("Worldwide-Earthquake-Continent-database.csv")
EQ_svm = _Svm("Worldwide-Earthquake-Continent-database.csv")
EQ_Decision_tree = _Decision_tree("Worldwide-Earthquake-Continent-database.csv")
EQ_NN = _NN("Worldwide-Earthquake-Continent-database.csv")
#EQ_LR = _Regression("Worldwide-Earthquake-Continent-database.csv")


# print("\n\n Question 1:")
# print("__Adaboost__")
# EQ_adaboost.Q1()
# print("__SVM__")
# EQ_svm.Q1()
#print("\n __DecisionTree__")
#EQ_Decision_tree.Q1()
# print("__NN__")
# EQ_NN.Q1()
#
# # print("\n LinearRegression algorithm")
# # EQ_LR.Q1()
#
# print("\n\n Question 2:")
# print("__Adaboost__")
# EQ_adaboost.Q2()
# print("__SVM__")
# EQ_svm.Q2()
#print("__DecisionTree__")
#EQ_Decision_tree.Q2()
# print("__NN__")
# EQ_NN.Q2()
#
# # print("\n LinearRegression algorithm")
# # EQ_LR.Q2()
#
# print("\n\n Question 3:")
# print("__Adaboost__")
# EQ_adaboost.Q3()
# print("__SVM__")
# EQ_svm.Q3()
#print("__DecisionTree__")
#EQ_Decision_tree.Q3()
# print("__NN__")
# EQ_NN.Q3()

# print("\n LinearRegression algorithm")
# EQ_LR.Q3()


print("\n\n Question 4:")
EQ_adaboost = _Adaboost("Population_EQ.csv")
EQ_svm = _Svm("Population_EQ.csv")
EQ_Decision_tree = _Decision_tree("Population_EQ.csv")
EQ_NN = _NN("Population_EQ.csv")


#print("__Adaboost__")
# EQ_adaboost.Q4()
# print("__SVM__")
# EQ_svm.Q4()
#print("__DecisionTree__")
#EQ_Decision_tree.Q4()
# print("__NN__")
# EQ_NN.Q4()








