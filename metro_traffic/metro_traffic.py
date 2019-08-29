import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils
from datetime_utils import TimeStep
from sklearn.preprocessing import LabelEncoder

holidayEncoder = LabelEncoder()
weatherEncoder = LabelEncoder()
descriptionEncoder = LabelEncoder()

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

    return data.reset_index(drop=False)

def updateMetroTrafficData(data, reindex=False, temp=None):
    if reindex:
        data.index = data.date_time
        data = data.drop(columns=['date_time'])

    if temp == 'C':
        data.temp = np.vectorize(misc_utils.convertK2C)(data.temp)
    elif temp == 'F':
        data.temp = np.vectorize(misc_utils.convertK2F)(data.temp)

    return data

def getMetroTrafficData(dupsKeep='last',
                        gapsAction='fill',
                        dateTimeIndex=False,
                        temp=None,
                        encode=True):
    mt = readMetroTrafficCSV()
    if encode:
        encodeMetroDataCategories(mt)
    if dupsKeep is not None:
        mt = cleanupMetroTrafficDups(mt, keep=dupsKeep)
    if gapsAction is not None:
        mt = cleanupMetroTrafficGaps(mt, action=gapsAction)
    mt = updateMetroTrafficData(mt, reindex=dateTimeIndex, temp=temp)
    return mt
