import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils

def getMetroTrafficData(path="~/Data/MetroInterstateTrafficVolume/", reindex=False, temp=None):
    mt = pd.read_csv(path + "Metro_Interstate_Traffic_Volume.csv")

    mt.date_time = np.vectorize(pd.Timestamp)(mt.date_time)
    if reindex:
        mt.index = mt.date_time
        mt = mt.drop(columns=['date_time'])

    if temp == 'C':
        mt.temp = np.vectorize(misc_utils.convertK2C)(mt.temp)
    elif temp == 'F':
        mt.temp = np.vectorize(misc_utils.convertK2F)(mt.temp)

    mt.holiday = mt.holiday.astype('category')
    mt.weather_main = mt.weather_main.astype('category')
    mt.weather_description = mt.weather_description.astype('category')

    return mt
