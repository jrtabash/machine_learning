import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils
from datetime_utils import TimeStep
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

class MetroTrafficException(Exception):
    def __init__(self, message):
        self.message = message

weekdayMap = dict({1: 'Monday',
                   2: 'Tuesday',
                   3: 'Wednesday',
                   4: 'Thursday',
                   5: 'Friday',
                   6: 'Saturday',
                   7: 'Sunday'})
intensityMap = dict({1: 'Very Low',
                     2: 'Low',
                     3: 'Medium',
                     4: 'High',
                     5: 'Very High'})

def weekdayLabel(wkdy):
    return weekdayMap[wkdy]

def intensityLabel(intsty):
    return intensityMap[intsty]

def volumeToIntensity(volume):
    if volume < 500:    # Very Low
        return 1
    elif volume < 2000: # Low
        return 2
    elif volume < 3500: # Medium
        return 3
    elif volume < 5000: # High
        return 4
    else:               # Very High
        return 5

def readMetroTrafficCSV(path="~/Data/MetroInterstateTrafficVolume/"):
    mt = pd.read_csv(path + "Metro_Interstate_Traffic_Volume.csv")

    mt.date_time = np.vectorize(pd.Timestamp)(mt.date_time)

    mt.holiday = mt.holiday.astype('category')
    mt.weather_main = mt.weather_main.astype('category')
    mt.weather_description = mt.weather_description.astype('category')

    return mt

def plotTrafficVolume(mt, squashHoliday=False):
    # Plot traffic volume by holiday and weather.
    # Expected mt input is result of readMetroTrafficCSV.

    data = mt
    if squashHoliday:
        data = mt[['holiday', 'weather_main', 'traffic_volume']].copy()
        data.holiday = np.vectorize(lambda h: 'None' if h == 'None' else 'Holiday')(data.holiday)
    sns.barplot(x='weather_main', y='traffic_volume', hue='holiday', data=data)
    plt.grid(True)
    plt.title("Traffic Volume by Holiday / Weather")
    plt.xlabel("Weather")
    plt.ylabel("Traffic Volume")
    plt.show()

def cleanupMetroTrafficDups(data, keep):
    if keep != 'first' and keep != 'last':
        raise(MetroTrafficException("Invalid duplicates keep parameter '{}'".format(keep)))
    return data.drop_duplicates(keep=keep, subset=['date_time']).reset_index(drop=True)

def cleanupMetroTrafficGaps(data, action, subAction=None):
    data.index = data.date_time
    data = data.drop(columns=['date_time'])

    data = data.resample('H').max()
    if action == 'fill':
        data = data.ffill()
    elif action == 'back_fill':
        data = data.bfill()
    elif action == 'interpolate':
        data = data.interpolate(method=('linear' if subAction is None else subAction))
        data.holiday = np.round(data.holiday)
        data.weather_main = np.round(data.weather_main)
        data.weather_description = np.round(data.weather_description)
    else:
        raise(MetroTrafficException("Invalid gaps action parameters '{}'".format(action)))

    data.holiday = data.holiday.astype(int)
    data.weather_main = data.weather_main.astype(int)
    data.weather_description = data.weather_description.astype(int)
    return data.reset_index(drop=False)

def updateMetroTrafficData(data, reindex=False, temp=None):
    data.insert(0, 'hour', np.vectorize(lambda x: x.hour)(data.date_time.dt.time))
    data.insert(0, 'week_day', np.vectorize(lambda x: x.isoweekday())(data.date_time.dt.date))
    data.insert(len(data.columns), 'intensity', np.vectorize(volumeToIntensity)(data.traffic_volume))

    if reindex:
        data.index = data.date_time
        data = data.drop(columns=['date_time'])

    if temp is not None:
        if temp == 'C':
            data.temp = np.vectorize(misc_utils.convertK2C)(data.temp)
        elif temp == 'F':
            data.temp = np.vectorize(misc_utils.convertK2F)(data.temp)
        else:
            raise(MetroTrafficException("Invalid update temp parameter '{}'".format(temp)))

    return data

def splitMetroTrafficData(data, intensity=False, approach='random'):
    X = data.drop(columns=['traffic_volume', 'intensity'])
    y = data[['traffic_volume']] if not intensity else data[['intensity']]
    if approach == 'random':
        return train_test_split(X, y, test_size=0.25)
    elif approach == 'datetime':
        # Assumes data is indexed by data_time
        splitPoint = pd.Timestamp('2017-07-01 00:00:00')
        X_learn = X[data.index < splitPoint]
        X_test = X[data.index >= splitPoint]
        y_learn = y[data.index < splitPoint]
        y_test = y[data.index >= splitPoint]
        return X_learn, X_test, y_learn, y_test
    else:
        raise(MetroTrafficException("Invalid split approach parameter '{}'".format(approach)))

def getMetroTrafficData(dupsKeep='last',
                        gapsAction='fill',
                        gapsSubAction=None,
                        dateTimeIndex=False,
                        temp=None):
    mt = readMetroTrafficCSV()

    encoder = data_utils.DataEncoder(['holiday', 'weather_main', 'weather_description'],
                                     oneHotEncoding=False)
    mt = encoder.encode(mt)

    if dupsKeep is not None:
        mt = cleanupMetroTrafficDups(mt, keep=dupsKeep)

    if gapsAction is not None:
        mt = cleanupMetroTrafficGaps(mt, action=gapsAction, subAction=gapsSubAction)

    mt = updateMetroTrafficData(mt, reindex=dateTimeIndex, temp=temp)

    return mt

def findBestRandomForestParams(XData, yData,
                               n_estimators=[150, 160, 170, 180, 190, 200],
                               max_features=[2, 3, 4],
                               min_samples_split=[2, 3]):
    gscv = GridSearchCV(estimator=RandomForestClassifier(max_depth=None),
                        param_grid={
                            "n_estimators": n_estimators,
                            "max_features": max_features,
                            "min_samples_split": min_samples_split
                        },
                        cv=5,
                        verbose=True,
                        n_jobs=-1)
    gridResult = gscv.fit(XData.values, yData.values.ravel())
    print(gridResult.best_params_)

def makeRandomForestModel(XData, yData, n_estimators=10, max_features=3, max_depth=None, min_samples_split=2):
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_features=max_features,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split)
    model.fit(XData.values, yData.values.ravel())
    return model

def testRandomForestModel(model, XData, yData, cv=5):
    scores = cross_val_score(model, XData.values, yData.values.ravel(), cv=cv)
    print(" Scores: {}".format(scores))
    print("Average: {}".format(np.average(scores)))

def makeNeuralNetworkModel(XData, yData,
                           layers=[(27, 'relu', ''), (5, 'softmax', '')],
                           optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'],
                           epochs=10):
    model = tf.keras.models.Sequential()
    for sizeActivReg in layers:
        regularizer = None if (len(sizeActivReg) == 2 or not sizeActivReg[2]) else sizeActivReg[2]
        model.add(tf.keras.layers.Dense(sizeActivReg[0], activation=sizeActivReg[1], kernel_regularizer=regularizer))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(XData.values, yData.values.ravel(), epochs=epochs)
    return model

def getDataForNeuralNetworkModel(standardScaler=True, squashHoliday=False, dropDescription=True):
    mt = readMetroTrafficCSV()

    if squashHoliday:
        mt.holiday = np.vectorize(lambda h: 'None' if h == 'None' else 'Holiday')(mt.holiday)

    columnsToEncode = ['holiday', 'weather_main']
    if not dropDescription:
        columnsToEncode.append('weather_description')
    encoder = data_utils.DataEncoder(columnsToEncode, oneHotEncoding=True)
    mt = encoder.encode(mt)

    mt = cleanupMetroTrafficDups(mt, keep='last')
    mt = updateMetroTrafficData(mt, reindex=False, temp='F')

    columnsToDrop = ['date_time', 'rain_1h', 'snow_1h']
    if dropDescription:
        columnsToDrop.append('weather_description')
    mt = mt.drop(columns=columnsToDrop)

    scaler = StandardScaler() if standardScaler else MinMaxScaler()
    scaleColumns = ['week_day', 'hour', 'temp', 'clouds_all']
    scaler.fit(mt[scaleColumns])
    mt[scaleColumns] = scaler.transform(mt[scaleColumns])

    xl, xt, yl, yt = splitMetroTrafficData(mt, intensity=True, approach='random')
    return xl, xt, yl - 1, yt - 1

def nnPredictionLabel(prediction):
    return intensityLabel(prediction + 1)

def nnPredict(model, data):
    return [np.argmax(p) for p in model.predict(misc_utils.toNPArray(data))]

def nnPredictLabels(model, data):
    return [nnPredictionLabel(p) for p in nnPredict(model, data)]

def nnConfusionMatrix(model, x, y):
    cm = confusion_matrix(nnPredict(model, x.values), y.values.ravel())
    labels = [nnPredictionLabel(lbl) for lbl in np.unique(y.values.ravel())]
    plot_utils.plotConfusionMatrix(cm, cmap='Reds', labels=labels)
