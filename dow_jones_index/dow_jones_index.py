import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def readDowJonesCSV(path="~/Data/DowJonesIndex"):
    dateUpdater = np.vectorize(lambda dStr: np.int64(pd.Timestamp(dStr).value / 1000000000))
    priceUpdater = np.vectorize(lambda pStr: np.float(pStr[1:]))

    dji = pd.read_csv(path + "/dow_jones_index.data")
    dji.date = dateUpdater(dji.date)
    dji.open = priceUpdater(dji.open)
    dji.high = priceUpdater(dji.high)
    dji.low = priceUpdater(dji.low)
    dji.close = priceUpdater(dji.close)
    dji.next_weeks_open = priceUpdater(dji.next_weeks_open)
    dji.next_weeks_close = priceUpdater(dji.next_weeks_close)

    dji.stock = dji.stock.astype('category')

    return dji

def addDowJonesDerivedData(data):
    data['percent_change_high'] = 100.0 * (data.high - data.open) / data.open
    data['percent_change_low'] = 100.0 * (data.low - data.open) / data.open
    data['week'] = (data.date - 1294358400) / 604800
    data.week = data.week.astype('int')
    
def splitDowJonesData(data, columns):
    X_columns = columns[:len(columns) - 1]
    y_column = columns[len(columns) - 1:]
    
    q1 = data[data.quarter == 1]
    q2 = data[data.quarter == 2]
    return q1[X_columns], q2[X_columns], q1[y_column], q2[y_column]

def makeDowJonesColumns(week=True, stock=False, prices=True, volume=True, percent=True, derived=False):
    columns = ['week'] if week else []
    if stock:
        columns.extend(['stock'])
    if prices:
        columns.extend(['open', 'high', 'low', 'close'])
    if volume:
        columns.extend(['volume'])
    if percent:
        columns.extend(['percent_change_price'])
    if derived:
        columns.extend(['percent_change_high', 'percent_change_low'])
    columns.extend(['percent_change_next_weeks_price'])
    return columns

def plotDowJonesData(xData, yData, xName):
    plt.scatter(xData[xName], yData["percent_change_next_weeks_price"])
    plt.grid(axis="both")
    plt.xlabel(xName)
    plt.ylabel("percent_change_next_weeks_price")
    plt.show()

def createPipeline(data, standardScaling=None, components=None):
    pipeline = None
    filters = []
    if standardScaling != None:
        filters.append(('scale', MinMaxScaler() if standardScaling == False else StandardScaler()))
    if components != None:
        filters.append(('pca', PCA(n_components=components)))
    if len(filters) > 0:
        pipeline = Pipeline(filters)
        pipeline.fit(data)
    return pipeline

def preprocessData(pipeline, trainingData, testData):
    preTrainingData = pd.DataFrame(pipeline.transform(trainingData))
    preTestData = pd.DataFrame(pipeline.transform(testData))
    return preTrainingData, preTestData

def createSVR(trainingX, trainingY, C=1.0, gamma="auto", kernel="rbf", coef0=0.0, degree=3, epsilon=0.01):
    svr = SVR(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree, epsilon=epsilon)
    svr.fit(trainingX.values, trainingY.values.ravel())
    return svr

def crossValidate(model, trainingX, trainingY, foldCV, metric):
    validationScores = cross_val_score(model,
                                       trainingX.values,
                                       trainingY.values.ravel(),
                                       cv=foldCV,
                                       scoring=metric)
    validationAverage = np.average(validationScores)
    print(" Validation Scores: {}".format(validationScores))
    print("Validation Average: {}".format(validationAverage))

def floatRange(begin, end, step=1.0):
    values = []
    if begin <= end:
        cur = begin
        while cur < end:
            values.append(cur)
            cur = cur + step
    return values

def findBestParams(trainingX,
                   trainingY,
                   gammaRange,
                   cRange,
                   epsilonRange,
                   kernelRange,
                   coef0Range,
                   degreeRange,
                   scoring=None,
                   verbose=False):
    if verbose:
        print("findBestParams:\n  scoring={}\n  gammaRange={}\n  cRange={}\n  epsilonRange={}\n  kernelRange={}\n  coef0Range={}\n  degreeRange={}"
              .format(scoring, gammaRange, cRange, epsilonRange, kernelRange, coef0Range, degreeRange))
    gsc = GridSearchCV(estimator=SVR(),
                       param_grid={
                           "C": cRange,
                           "gamma": gammaRange,
                           "epsilon": epsilonRange,
                           "kernel": kernelRange,
                           "coef0": coef0Range,
                           "degree": degreeRange
                           },
                       cv=3,
                       scoring=scoring,
                       verbose=verbose,
                       n_jobs=-1)
    gridResult = gsc.fit(trainingX.values, trainingY.values.ravel())
    if verbose:
        print("findBestParams: best_params={}".format(gridResult.best_params_))
    return gridResult.best_params_

def findBestEstimator(trainingX,
                      trainingY,
                      gammaRange=[0.01, 0.1, 1.0, 1.1, 10.0, 50.0, 100.0],
                      cRange=[1.0, 1.1],
                      epsilonRange=[0.01, 0.05],
                      kernelRange=["rbf"],
                      coef0Range=[0.0],
                      degreeRange=[3],
                      scoring=None,
                      validate=False,
                      verbose=False):
    bestParams = findBestParams(trainingX,
                                trainingY,
                                scoring=scoring,
                                gammaRange=gammaRange,
                                cRange=cRange,
                                epsilonRange=epsilonRange,
                                kernelRange=kernelRange,
                                coef0Range=coef0Range,
                                degreeRange=degreeRange,
                                verbose=verbose)
    estimator = createSVR(trainingX,
                          trainingY,
                          C=bestParams["C"],
                          gamma=bestParams["gamma"],
                          kernel=bestParams["kernel"],
                          coef0=bestParams["coef0"],
                          degree=bestParams["degree"],
                          epsilon=bestParams["epsilon"])
    if verbose:
        print("Estimator: {}".format(estimator));
    if validate:
        crossValidate(estimator, trainingX, trainingY, 3, "explained_variance")
    return estimator

def getDataForTesting(columns, standardScaling=None, components=None):
    dji = readDowJonesCSV()
    addDowJonesDerivedData(dji)
    X_train, X_test, y_train, y_test = splitDowJonesData(dji, columns)
    pipeline = createPipeline(X_train, standardScaling=standardScaling, components=components)
    if pipeline != None:
        X_train, X_test = preprocessData(pipeline, X_train, X_test)
    return X_train, X_test, y_train, y_test
