# 
#
# Diabetes Diagnosis using Artificial Inteligence and Machine Learning
#
#

# Imports

# Load libraries
import numpy
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Main Code


url = "data.csv"
names = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
dataset = read_csv(url, names=names)

# shape of dataset
print("Tetsing")
print(dataset.shape)


