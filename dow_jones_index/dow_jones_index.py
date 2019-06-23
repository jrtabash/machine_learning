import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def readDowJonesCSV(path="~/Data/DowJonesIndex", datesToWeeks=False):
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

    if datesToWeeks:
        dji = dji.rename(columns={"date": "week"})
        dji.week = (dji.week - 1294358400) / 604800
        dji.week = dji.week.astype('int')

    return dji

def splitDowJonesData(data, datesToWeeks=False):
    X_columns = ['week' if datesToWeeks else 'date',
                 'open',
                 'high',
                 'low',
                 'close',
                 'volume',
                 'percent_change_price',
                 'days_to_next_dividend']
    y_column = ['percent_change_next_weeks_price']
    q1 = data[data.quarter == 1]
    q2 = data[data.quarter == 2]
    return q1[X_columns], q2[X_columns], q1[y_column], q2[y_column]

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
    gammas = []
    if begin <= end:
        cur = begin
        while cur < end:
            gammas.append(cur)
            cur = cur + step
    return gammas

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

def getDataForTesting(standardScaling=None, components=None, datesToWeeks=False):
    dji = readDowJonesCSV(datesToWeeks=datesToWeeks)
    X_train, X_test, y_train, y_test = splitDowJonesData(dji, datesToWeeks=datesToWeeks)
    pipeline = createPipeline(X_train, standardScaling=standardScaling, components=components)
    if pipeline != None:
        X_train, X_test = preprocessData(pipeline, X_train, X_test)
    return X_train, X_test, y_train, y_test
