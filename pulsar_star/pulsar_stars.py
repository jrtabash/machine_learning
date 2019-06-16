import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import tree

def readPulsarCSV(path="~/Data/PulsarStar"):
    return pd.read_csv(path + "/pulsar_stars.csv",
                       header=1,
                       names=["mean_ip",      # Mean of the integrated profile
                              "stddev_ip",    # Standard deviation of the integrated profile
                              "exs_krts_ip",  # Excess kurtosis of the integrated profile
                              "skew_ip",      # Skewness of the integrated profile
                              "mean_dm",      # Mean of the DM-SNR curve
                              "stddev_dm",    # Standard deviation of the DM-SNR curve
                              "exs_krts_dm",  # Excess kurtosis of the DM-SNR curve
                              "skew_dm",      # Skewness of the DM-SNR curv
                              "is_pulsar"])   # Class 0=False, 1=True

def calcPulsarCorrelations(data):
    return pulsar.corr().iloc[0:8, 8]

def splitPulsarData(data, testSize):
    X = data.iloc[:, :8]
    y = data.is_pulsar
    return train_test_split(X, y, test_size=testSize)

def createPulsarPipeline(trainingData):
    pipeline = Pipeline([('scale', MinMaxScaler())])
    pipeline.fit(trainingData)
    return pipeline

def preprocessPulsarData(pipeline, trainingData, testData):
    trainingData2 = pd.DataFrame(pipeline.transform(trainingData))
    testData2 = pd.DataFrame(pipeline.transform(testData))
    return trainingData2, testData2

def createTreeClassifier(trainingXData, trainingYData, seed=None):
    classifier = tree.DecisionTreeClassifier(random_state=seed)
    classifier.fit(trainingXData, trainingYData)
    return classifier

def testPulsarModel(model, trainingXData, trainingYData, testXData, testYData):
    validationScores = cross_val_score(model, trainingXData, trainingYData, cv=5)
    validationAverage = np.average(validationScores)
    
    print("Validation Score: {:.4f}".format(validationAverage))
    print("      Test Score: {:.4f}".format(model.score(testXData, testYData)))

def test(preprocess=False, seed=None, verbose=False):
    if verbose == True:
        print("Test Pulsar Data: preprocess={}".format(preprocess))

    X_train, X_test, y_train, y_test = splitPulsarData(readPulsarCSV(), testSize=0.33)

    if preprocess == True:
        pipeline = createPulsarPipeline(X_train)
        
        if verbose == True:
            print("***** Pipeline:")
            print(pipeline)
            
        X_train, X_test = preprocessPulsarData(pipeline, X_train, X_test)

    treeClass = createTreeClassifier(X_train, y_train, seed=seed)

    if verbose == True:
        print("***** Model:")
        print(treeClass)

    testPulsarModel(treeClass, X_train, y_train, X_test, y_test)
