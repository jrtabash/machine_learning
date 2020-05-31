import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import data_utils

def readDowJonesCSV(path="~/Data/DowJonesIndex"):
    def str2price(s):
        return np.float(s[1:])

    dji = pd.read_csv(path + "/dow_jones_index.data",
                      converters={'date': pd.Timestamp,
                                  'open': str2price,
                                  'high': str2price,
                                  'low': str2price,
                                  'close': str2price,
                                  'next_weeks_open': str2price,
                                  'next_weeks_close': str2price})

    dji.stock = dji.stock.astype('category')

    dji.sort_values(by=['date'], inplace=True)
    dji.index = dji.date
    dji.drop(columns=['date'], inplace=True)

    return dji

def addDowJonesDerivedData(data):
    data['percent_change_high'] = 100.0 * (data.high - data.open) / data.open
    data['percent_change_low'] = 100.0 * (data.low - data.open) / data.open
    data['week'] = np.array([d.week for d in data.index], dtype=np.int32)

def splitDowJonesData(data, byDate=None):
    columns = data.columns
    X_columns = columns[:len(columns) - 1]
    y_column = columns[len(columns) - 1:]

    if byDate is None:
        q1 = data[data.quarter == 1]
        q2 = data[data.quarter == 2]
    else:
        if isinstance(byDate, str):
            byDate = pd.Timestamp(byDate)
        q1 = data[:byDate]
        q2 = data[byDate + pd.Timedelta('1d'):]
    return q1[X_columns], q2[X_columns], q1[y_column], q2[y_column]

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

def getDowJonesData(stockClusters=0, stockScale=False, scale=None, components=None, window=None):
    data = readDowJonesCSV()

    addDowJonesDerivedData(data)

    columns = ['quarter',
               'stock',
               'volume',
               'percent_change_price',
               'percent_change_high',
               'percent_change_low',
               'days_to_next_dividend',
               'percent_return_next_dividend',
               'percent_change_next_weeks_price']
    data = data[columns].copy()

    if stockScale:
        gmmScaler = data_utils.GroupMinMaxScaler('stock', keepColumns=['quarter']).fit(data)
        data = gmmScaler.transform(data)

    xl, xt, yl, yt = splitDowJonesData(data)
    xl.drop(columns=['quarter', 'stock'], inplace=True)
    xt.drop(columns=['quarter', 'stock'], inplace=True)

    pipeline = data_utils.createPipeline(xl, scale=scale, components=components)
    if pipeline is not None:
        xl, xt = data_utils.preprocessData(pipeline, xl, xt, copyColumns=(components is None))

    if stockClusters > 0:
        km = KMeans(n_clusters=stockClusters).fit(xl)
        xl['cluster'] = km.labels_
        xt['cluster'] = km.predict(xt)
        encoder = data_utils.DataEncoder(columns=['cluster'], oneHotEncoding=True)
        xl = encoder.encode(xl)
        xt = encoder.encode(xt)

    if window:
        xl = xl.rolling(window).mean().dropna()
        xt = xt.rolling(window).mean().dropna()
        yl = yl.rolling(window).mean().dropna()
        yt = yt.rolling(window).mean().dropna()

    xl.reset_index(drop=True, inplace=True)
    xt.reset_index(drop=True, inplace=True)
    yl.reset_index(drop=True, inplace=True)
    yt.reset_index(drop=True, inplace=True)

    return xl, xt, yl, yt

def getResampledDowJonesData(scale=None, components=None, splitByDate=None):
    data = readDowJonesCSV()

    addDowJonesDerivedData(data)

    data = data.resample('1w').mean()

    columns = ['quarter',
               'volume',
               'percent_change_price',
               'percent_change_high',
               'percent_change_low',
               'days_to_next_dividend',
               'percent_return_next_dividend',
               'percent_change_next_weeks_price']
    data = data[columns]

    xl, xt, yl, yt = splitDowJonesData(data, byDate=splitByDate)
    xl.drop(columns=['quarter'], inplace=True)
    xt.drop(columns=['quarter'], inplace=True)

    pipeline = data_utils.createPipeline(xl, scale=scale, components=components)
    if pipeline is not None:
        xl, xt = data_utils.preprocessData(pipeline, xl, xt, copyColumns=(components is None))

    return xl, xt, yl, yt
