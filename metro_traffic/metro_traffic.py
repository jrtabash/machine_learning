import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils
from datetime_utils import TimeStep
from sklearn.preprocessing import LabelEncoder

class MetroTrafficException(Exception):
    def __init__(self, message):
        self.message = message

holidayEncoder = LabelEncoder()
weatherEncoder = LabelEncoder()
descriptionEncoder = LabelEncoder()
weekdayMap = dict({1: 'Monday',
                   2: 'Tuesday',
                   3: 'Wednesday',
                   4: 'Thursday',
                   5: 'Friday',
                   6: 'Saturday',
                   7: 'Sunday'})

def holidayLabel(encoding):
    return holidayEncoder.classes_[encoding]

def weatherLabel(encoding):
    return weatherEncoder.classes_[encoding]

def descriptionLabel(encoding):
    return descriptionEncoder.classes_[encoding]

def weekdayLabel(wkdy):
    return weekdayMap[wkdy]

def readMetroTrafficCSV(path="~/Data/MetroInterstateTrafficVolume/"):
    mt = pd.read_csv(path + "Metro_Interstate_Traffic_Volume.csv")

    mt.date_time = np.vectorize(pd.Timestamp)(mt.date_time)

    mt.holiday = mt.holiday.astype('category')
    mt.weather_main = mt.weather_main.astype('category')
    mt.weather_description = mt.weather_description.astype('category')

    return mt

def encodeMetroDataCategories(data):
    holidayEncoder.fit(np.unique(data.holiday))
    weatherEncoder.fit(np.unique(data.weather_main))
    descriptionEncoder.fit(np.unique(data.weather_description))

    data.holiday = holidayEncoder.transform(data.holiday)
    data.weather_main = weatherEncoder.transform(data.weather_main)
    data.weather_description = descriptionEncoder.transform(data.weather_description)

def cleanupMetroTrafficDups(data, keep):
    if keep != 'first' and keep != 'last':
        raise(MetroTrafficException("Invalid duplicates keep parameter '{}'".format(keep)))
    return data.drop_duplicates(keep=keep, subset=['date_time']).reset_index(drop=True)

def cleanupMetroTrafficGaps(data, action):
    data.index = data.date_time
    data = data.drop(columns=['date_time'])

    data = data.resample('H').max()
    if action == 'fill':
        data = data.ffill()
    elif action == 'back_fill':
        data = data.bfill()
    elif action == 'interpolate':
        data = data.interpolate()
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
    data.insert(1, 'hour', np.vectorize(lambda x: x.hour)(data.date_time.dt.time))
    data.insert(1, 'week_day', np.vectorize(lambda x: x.isoweekday())(data.date_time.dt.date))

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

def getMetroTrafficData(dupsKeep='last',
                        gapsAction='fill',
                        dateTimeIndex=False,
                        temp=None):
    mt = readMetroTrafficCSV()

    encodeMetroDataCategories(mt)

    if dupsKeep is not None:
        mt = cleanupMetroTrafficDups(mt, keep=dupsKeep)

    if gapsAction is not None:
        mt = cleanupMetroTrafficGaps(mt, action=gapsAction)

    mt = updateMetroTrafficData(mt, reindex=dateTimeIndex, temp=temp)

    return mt
