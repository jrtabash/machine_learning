import numpy as np
import pandas as pd
import plot_utils
import data_utils
import misc_utils
import datetime_utils
from datetime_utils import TimeStep
from cleanup_utils import DuplicatesProcessor
from cleanup_utils import GapsProcessor

def readMetroTrafficCSV(path="~/Data/MetroInterstateTrafficVolume/"):
    mt = pd.read_csv(path + "Metro_Interstate_Traffic_Volume.csv")

    mt.date_time = np.vectorize(pd.Timestamp)(mt.date_time)

    mt.holiday = mt.holiday.astype('category')
    mt.weather_main = mt.weather_main.astype('category')
    mt.weather_description = mt.weather_description.astype('category')

    return mt

def cleanupMetroTrafficDups(data, action=DuplicatesProcessor.Action.DropFirst):
    dupsProc = DuplicatesProcessor(
        action,
        lambda mt: datetime_utils.findDateTimeDuplicates(mt,
                                                         step=TimeStep.Hour,
                                                         calcPreceding=True,
                                                         flatten=False))
    return dupsProc.process(data)

def cleanupMetroTrafficGaps(data, action=GapsProcessor.Action.CarryForward):
    gapsProc = GapsProcessor(
        action,
        list(data.columns).index('date_time'), # gap column index
        lambda mt: datetime_utils.findDateTimeGaps(mt,
                                                   step=TimeStep.Hour,
                                                   calcPreceding=True,
                                                   flatten=False),
        lambda begin, end: datetime_utils.dateTimeRange(begin,
                                                        end,
                                                        step=TimeStep.Hour))
    return gapsProc.process(data)

def updateMetroTrafficData(data, reindex=False, temp=None):
    if reindex:
        data.index = data.date_time
        data = data.drop(columns=['date_time'])

    if temp == 'C':
        data.temp = np.vectorize(misc_utils.convertK2C)(data.temp)
    elif temp == 'F':
        data.temp = np.vectorize(misc_utils.convertK2F)(data.temp)

    return data

def getMetroTrafficData(dupsAction=DuplicatesProcessor.Action.DropFirst,
                        gapsAction=GapsProcessor.Action.CarryForward,
                        dateTimeIndex=False,
                        temp=None):
    mt = readMetroTrafficCSV()
    if dupsAction is not None:
        mt = cleanupMetroTrafficDups(mt, action=dupsAction)
    if gapsAction is not None:
        mt = cleanupMetroTrafficGaps(mt, action=gapsAction)
    mt = updateMetroTrafficData(mt, reindex=dateTimeIndex, temp=temp)
    return mt
