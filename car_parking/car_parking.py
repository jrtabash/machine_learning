import numpy as np
import pandas as pd

DefaultCSV = '~/Data/ParkingBirmingham/parking_birmingham.csv'

def readData(csvFile=DefaultCSV):
    return pd.read_csv(csvFile,
                       converters={'update_time': pd.to_datetime},
                       header=0,
                       names=['system_code', 'capacity', 'occupancy', 'update_time'],)

def cleanupData(data):
    data.occupancy = np.vectorize(min)(data.occupancy, data.capacity)
    return data[data.occupancy > 0].reset_index(drop=True)

def reindexData(data):
    data.sort_values(by=['update_time'], inplace=True)
    data.index = data.update_time
    data.drop(columns=['update_time'], inplace=True)
    return data

def addFeatures(data):
    maxOccupancy = data[['system_code', 'occupancy']].groupby(by=['system_code']).max().reset_index(drop=False)
    maxOccupancy.index = maxOccupancy.system_code
    maxOccupancy.drop(columns=['system_code'], inplace=True)

    getMaxOccupancy = np.vectorize(lambda c: maxOccupancy[c:c].values[0][0])

    data['rate'] = data.occupancy / np.float64(data.capacity)
    data['hrate'] = data.occupancy / getMaxOccupancy(data.system_code)
    data['weekday'] = np.vectorize(lambda x: x.isoweekday())(data.reset_index(drop=False).update_time.dt.date)

    return data

def getData(csvFile=DefaultCSV, reindex=True, cleanup=True, features=True):
    data = readData(csvFile)
    if cleanup:
        data = cleanupData(data)
    if reindex:
        data = reindexData(data)
    if features:
        data = addFeatures(data)
    return data
