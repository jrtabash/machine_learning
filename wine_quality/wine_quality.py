import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def readWineQualityCSV(whichData='red', path="~/Data/WineQuality"):
    if whichData == 'red':
        return pd.read_csv(path + "/winequality-red.csv", delimiter=";")
    elif whichData == 'white':
        return pd.read_csv(path + "/winequality-white.csv", delimiter=";")
    elif whichData == 'both':
        return pd.concat([readWineQualityCSV('red'), readWineQualityCSV('white')], axis=0, sort=False)
    else:
        return None

def plotCorrelation(data):
    pd.plotting.scatter_matrix(data)
    plt.show()

def showCorrelation(data, colorMap="YlOrRd"):
    corr = data.corr(method="kendall")
    plt.matshow(corr, cmap=plt.get_cmap(colorMap))
    plt.show()

def explorePlot(data, columnName):
    plt.scatter(data[columnName], data["quality"])
    plt.grid(axis="both")
    plt.xlabel(columnName)
    plt.ylabel("quality")
    plt.show()

def splitWineQuality(data, testSize, goodBadLabels=False):
    X = data.iloc[:, :11]
    y = data.quality if goodBadLabels == False else pd.Series(np.vectorize(lambda x: 1 if x >= 7 else 0)(data.quality))
    return train_test_split(X, y, test_size=testSize)

def createPipeline(trainingData, pcaN):
    filters = [('scale', MinMaxScaler())] if pcaN == -1 else [('scale', StandardScaler()), ('pca', PCA(n_components=pcaN))]
    pipeline = Pipeline(filters)
    pipeline.fit(trainingData)
    return pipeline

def preprocessData(pipeline, trainingData, testData):
    trainingData2 = pd.DataFrame(pipeline.transform(trainingData))
    testData2 = pd.DataFrame(pipeline.transform(testData))
    return trainingData2, testData2

def createRbfSVC(trainingXData, traingingYData, c=1.0, g=0.0001, seed=None):
    svc = SVC(C=c, kernel='rbf', gamma=g, random_state=seed)
    svc.fit(trainingXData.values, traingingYData.values)
    return svc

def findBestParams(trainingXData, trainingYData, scoring=None):
    gsc = GridSearchCV(estimator=SVC(kernel='rbf'),
                       param_grid={
                           'C': [0.01, 0.1, 1.0, 1.1, 10, 50, 99, 100, 101, 1000],
                           'gamma': [0.0001, 0.001, 0.005, 0.1, 1.0, 5.0, 10.0, 49.0, 50.0, 51.0, 89.0, 90.0, 91.0, 100.0]
                       },
                       cv=5,
                       scoring=scoring,
                       verbose=0,
                       n_jobs=-1)
    gridResult = gsc.fit(trainingXData, trainingYData)
    return gridResult.best_params_

def findBestEstimator(trainingXData, trainingYData, scoring=None):
    bestParams = findBestParams(trainingXData, trainingYData, scoring=scoring)
    return createRbfSVC(trainingXData, trainingYData, c=bestParams["C"], g=bestParams["gamma"])

def testSVC(model, trainingXData, trainingYData, testXData, testYData, foldCV=5):
    validationScores = cross_val_score(model, trainingXData, trainingYData, cv=foldCV)
    validationAverage = np.average(validationScores)
    
    print("Validation Score: {:.4f}".format(validationAverage))
    print("      Test Score: {:.4f}".format(model.score(testXData, testYData)))

def test(whichData='red', testSize=0.33, pcaN=-1, c=1.0, g=0.001, foldCV=5, seed=None, bestEstimator=False, goodBadLabels=False, verbose=False):
    if verbose == True:
        print("Test WineQuality: data={} bestEstimator={} goodBadLabels={}".format(whichData, bestEstimator, goodBadLabels))

    wineQuality = readWineQualityCSV(whichData=whichData)
    X_train, X_test, y_train, y_test = splitWineQuality(wineQuality, testSize=testSize, goodBadLabels=goodBadLabels)

    pipeline = createPipeline(X_train, pcaN=pcaN)
    X_train_preprocessed, X_test_preprocessed = preprocessData(pipeline, X_train, X_test)

    svc = None
    if bestEstimator == True:
        svc = findBestEstimator(X_train_preprocessed, y_train, scoring=None)
    else:
        svc = createRbfSVC(X_train_preprocessed, y_train, c=c, g=g, seed=seed)
    
    if verbose == True:
        print("***** Pipeline:")
        print(pipeline)
        print("***** Model:")
        print(svc)
        print("***** Running test")
    
    testSVC(svc, X_train_preprocessed, y_train, X_test_preprocessed, y_test, foldCV=foldCV)
