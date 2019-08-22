import numpy as np
import pandas as pd
import data_utils
import misc_utils
from sklearn.svm import SVR
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

def filterStocks(data, names):
    if type(names) == str:
        names = [names]
    return data[data.stock.isin(names)]

def filterStocksByMatchingQuarterDirection(data, sameDirection=True):
    groupBy = data[['quarter', 'stock', 'percent_change_next_weeks_price']].groupby(['quarter', 'stock']).mean().reset_index()
    ones = groupBy[groupBy.quarter == 1]
    twos = groupBy[groupBy.quarter == 2]
    onesSign = np.sign(ones.percent_change_next_weeks_price.values)
    twosSign = np.sign(twos.percent_change_next_weeks_price.values)
    flags = onesSign == twosSign if sameDirection else onesSign != twosSign
    stocks = np.unique(ones.stock[flags].values)
    return data[data.stock.isin(stocks)]

def addDowJonesDerivedData(data):
    data['percent_change_high'] = 100.0 * (data.high - data.open) / data.open
    data['percent_change_low'] = 100.0 * (data.low - data.open) / data.open
    data['week'] = (data.date - 1294358400) / 604800
    data.week = data.week.astype('int')
    
def splitDowJonesData(data, columns):
    X_columns = columns[:len(columns) - 1]
    y_column = columns[len(columns) - 1:]
    
    q1 = data[data.quarter == 1].reset_index()
    q2 = data[data.quarter == 2].reset_index()
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

def createSVR(trainingX, trainingY, C=1.0, gamma="auto", kernel="rbf", coef0=0.0, degree=3, epsilon=0.01):
    svr = SVR(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree, epsilon=epsilon)
    svr.fit(trainingX.values, trainingY.values.ravel())
    return svr

def crossValidate(model, trainingX, trainingY, foldCV, scoring):
    validationScores = cross_val_score(model,
                                       trainingX.values,
                                       trainingY.values.ravel(),
                                       cv=foldCV,
                                       scoring=scoring)
    validationAverage = np.average(validationScores)
    print(" Validation Scores: {}".format(validationScores))
    print("Validation Average: {}".format(validationAverage))

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
        print("Estimator: {}".format(estimator))
    if validate:
        crossValidate(estimator, trainingX, trainingY, 3, scoring)
    return estimator

def makeDowJonesSegments(data, segmentOffset, segmentLength, xAggFtn=None, yAggFtn=None):
    xSegments = data_utils.makeSegments(data, segmentOffset, segmentLength, flatten=True, aggFtn=xAggFtn)

    ySegments = None
    if yAggFtn is not None:
        ySegments = data_utils.makeSegments(data, segmentOffset, segmentLength, flatten=True, aggFtn=yAggFtn)

    return xSegments, ySegments

def getDataForTesting(columns, scale=None, components=None, stocks=None, filterSameDirection=None):
    dji = readDowJonesCSV()
    if stocks is not None:
        dji = filterStocks(dji, stocks)
    if filterSameDirection is not None:
        dji = filterStocksByMatchingQuarterDirection(dji, sameDirection=filterSameDirection)
    addDowJonesDerivedData(dji)
    X_train, X_test, y_train, y_test = splitDowJonesData(dji, columns)
    pipeline = data_utils.createPipeline(X_train, scale=scale, components=components)
    if pipeline is not None:
        X_train, X_test = data_utils.preprocessData(pipeline, X_train, X_test, copyColumns=(components is None))
    return X_train, X_test, y_train, y_test
