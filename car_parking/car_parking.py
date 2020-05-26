import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
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

def evalKMeansInertia(ns, data):
    return [KMeans(n_clusters=n).fit(data).inertia_ for n in ns]

def findBestInertia(ns, inertias, relativeThreshold=0.1, verbose=False):
    def relChange(fromIdx, toIdx):
        return np.abs(inertias[toIdx] - inertias[fromIdx]) / inertias[fromIdx]

    relChgs = np.array([relChange(i - 1, i) for i in range(1, len(inertias))])
    relDffs = np.abs(np.diff(relChgs))

    relIdx = len(relDffs) - 1
    for i in range(0, relIdx - 1):
        if relDffs[i] <= relativeThreshold:
            relIdx = i
            break

    ret = (ns[relIdx], inertias[relIdx])

    if verbose:
        print("findBestInertia: ns={}".format(ns))
        print("               : inertias={}".format(inertias))
        print("               : relChgs={}".format(relChgs))
        print("               : relDffs={}".format(relDffs))
        print("               : relIdx={}".format(relIdx))
        print("               : ret={}".format(ret))

    return ret

def plotKMeansInertia(ns, inertias, markBest=False):
    plt.plot(ns, inertias, '-o', label='Inertia')

    if markBest:
        bn, bi = findBestInertia(ns, inertias)
        plt.plot(bn, bi, color='r', marker='o', label='Best')

    plt.title('KMeans Inertia')
    plt.xlabel('clusters')
    plt.ylabel('Inertia')
    plt.grid(True)

    if markBest:
        plt.legend(loc='upper right')

    plt.show()
