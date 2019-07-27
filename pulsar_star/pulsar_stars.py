import numpy as np
import pandas as pd
import data_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

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

def createClassifier(whichClass, trainingXData, trainingYData, seed=None, sgdLoss="hinge", sgdPenalty=None):
    classifier = None

    if (whichClass == "tree"):
        classifier = DecisionTreeClassifier(random_state=seed)
    elif (whichClass == "sgd"):
        classifier = SGDClassifier(loss=sgdLoss,
                                   penalty=sgdPenalty,
                                   random_state=seed,
                                   max_iter=1000,
                                   tol=0.003)

    classifier.fit(trainingXData, trainingYData)
    return classifier

def testPulsarModel(model, trainingXData, trainingYData, testXData, testYData):
    validationScores = cross_val_score(model, trainingXData, trainingYData, cv=5)
    validationAverage = np.average(validationScores)
    
    print("Validation Score: {:.4f}".format(validationAverage))
    print("      Test Score: {:.4f}".format(model.score(testXData, testYData)))

def test(whichClass="tree", preprocess=False, seed=None, verbose=False, sgdLoss="hinge", sgdPenalty=None):
    if verbose == True:
        print("Test Pulsar Data: whichClass={} preprocess={}".format(whichClass, preprocess))

    X_train, X_test, y_train, y_test = splitPulsarData(readPulsarCSV(), testSize=0.33)

    if preprocess == True:
        pipeline = data_utils.createPipeline(X_train, scale="minmax")
        
        if verbose == True:
            print("***** Pipeline:")
            print(pipeline)
            
        X_train, X_test = data_utils.preprocessData(pipeline, X_train, X_test)

    classifier = createClassifier(whichClass, X_train, y_train, seed=seed, sgdLoss=sgdLoss, sgdPenalty=sgdPenalty)

    if verbose == True:
        print("***** Model:")
        print(classifier)

    testPulsarModel(classifier, X_train, y_train, X_test, y_test)
