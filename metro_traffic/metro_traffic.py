import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils
from datetime_utils import TimeStep

def readMetroTrafficCSV(path="~/Data/MetroInterstateTrafficVolume/"):
    mt = pd.read_csv(path + "Metro_Interstate_Traffic_Volume.csv")

    mt.date_time = np.vectorize(pd.Timestamp)(mt.date_time)

    mt.holiday = mt.holiday.astype('category')
    mt.weather_main = mt.weather_main.astype('category')
    mt.weather_description = mt.weather_description.astype('category')

    return mt

def cleanupMetroTrafficDups(data, action='drop_first'):
    dups = datetime_utils.findDateTimeDuplicates(data, step=TimeStep.Hour, calcPreceding=False)
    if action == 'drop_second':
        data = data.drop(dups)
    elif action == 'drop_first':
        data = data.drop([x - 1 for x in dups])
    return data

def updateMetroTrafficData(data, reindex=False, temp=None):
    if reindex:
        data.index = data.date_time
        data = data.drop(columns=['date_time'])

    if temp == 'C':
        data.temp = np.vectorize(misc_utils.convertK2C)(data.temp)
    elif temp == 'F':
        data.temp = np.vectorize(misc_utils.convertK2F)(data.temp)

    return data

def getMetroTrafficData(dupsAction='drop_first', dateTimeIndex=False, temp=None):
    mt = readMetroTrafficCSV()
    if dupsAction is not None:
        mt = cleanupMetroTrafficDups(mt, action=dupsAction)
    mt = updateMetroTrafficData(mt, reindex=dateTimeIndex, temp=temp)
    return mt
