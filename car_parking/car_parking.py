import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def addFeatures(data, rateFeatures=True, timeFeatures=False):
    #  rate := Occupancy rate based on capacity
    # hrate := Occupancy rate based on historic max occupancy
    if rateFeatures:
        maxOccupancy = data[['system_code', 'occupancy']].groupby(by=['system_code']).max().reset_index(drop=False)
        maxOccupancy.index = maxOccupancy.system_code
        maxOccupancy.drop(columns=['system_code'], inplace=True)

        getMaxOccupancy = np.vectorize(lambda c: maxOccupancy[c:c].values[0][0])

        data['rate'] = data.occupancy / np.float64(data.capacity)
        data['hrate'] = data.occupancy / getMaxOccupancy(data.system_code)

    #  mnth := Month Jan=1, Feb=2, ...
    #  wkdy := Day of week 1=Monday, 2=Tuesday, ...
    #   ssm := Seconds since midnight
    if timeFeatures:
        dts = data.update_time.dt

        data['mnth'] = np.vectorize(lambda x: x.month)(dts.date)
        data['wkdy'] = np.vectorize(lambda x: x.isoweekday())(dts.date)
        data['ssm'] = np.vectorize(lambda t: (t.hour * 3600) + (t.minute * 60) + (t.second))(dts.time.values)

    return data

def getData(csvFile=DefaultCSV, reindex=True, cleanup=True, rateFeatures=True, timeFeatures=False, columnsToDrop=None):
    data = readData(csvFile)
    if cleanup:
        data = cleanupData(data)
    if rateFeatures or timeFeatures:
        data = addFeatures(data, rateFeatures, timeFeatures)
    if reindex:
        data = reindexData(data)
    if columnsToDrop:
        data.drop(columns=columnsToDrop, inplace=True)
    return data

def makePipeline(data, scaler=None, pca=None, clusters=None):
    pipeline=None
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler() if scaler == 'std' else MinMaxScaler()))
    if pca:
        steps.append(('pca', PCA(n_components=pca)))
    if clusters:
        steps.append(('kmeans', KMeans(n_clusters=clusters)))

    if len(steps) > 0:
        pipeline = Pipeline(steps).fit(data)

    return pipeline

def plotKMeansInertia(data, ns=range(1, 10)):
    inertias = [KMeans(n_clusters=n).fit(data).inertia_ for n in ns]

    plt.plot(ns, inertias, '-o')
    plt.title('KMeans Inertia')
    plt.xlabel('clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
