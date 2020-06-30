from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class DataUtilException(Exception):
    pass

class DataEncoder:
    def __init__(self, columns, oneHotEncoding=False):
        self.columns = columns
        self.oneHotEncoding = oneHotEncoding
        self.labelEncoders = dict()

    def getColumns(self):
        return self.columns

    def isOneHotEncoding(self):
        return self.oneHotEncoding

    def getLabel(self, columnName, encoding):
        if not self.oneHotEncoding:
            if columnName in self.labelEncoders:
                return self.labelEncoders[columnName].classes_[encoding]
        return ""

    def encode(self, data):
        if self.oneHotEncoding:
            return pd.get_dummies(data, columns=self.columns)

        cpy = data.copy()
        for col in self.columns:
            self.labelEncoders[col] = LabelEncoder().fit(np.unique(cpy[col]))
            cpy[col] = self.labelEncoders[col].transform(cpy[col])
        return cpy

class GroupMinMax:
    def __init__(self, data, byColumn):
        self.minMax = defaultdict(dict)
        self.populateMinMax_(data, byColumn)

    def __len__(self):
        return len(self.minMax)

    def __contains__(self, byValue):
        return byValue in self.minMax

    def getMin(self, byValue, column):
        return self.minMax[byValue][column][0]

    def getMax(self, byValue, column):
        return self.minMax[byValue][column][1]

    def populateMinMax_(self, data, byColumn):
        mins = data.groupby(by=[byColumn]).min().reset_index(drop=False).sort_values(by=[byColumn])
        maxs = data.groupby(by=[byColumn]).max().reset_index(drop=False).sort_values(by=[byColumn])

        for i in range(len(mins)):
            minRow = mins[i:i+1]
            maxRow = maxs[i:i+1]
            byValue = minRow[byColumn].values[0]
            assert(byValue == maxRow[byColumn].values[0])
            for column in data.columns:
                if column == byColumn:
                    continue
                self.minMax[byValue][column] = (minRow[column].values[0], maxRow[column].values[0])

class GroupMinMaxScaler:
    def __init__(self, byColumn):
        self.byColumn = byColumn
        self.minMax = None

    def fit(self, data):
        self.minMax = GroupMinMax(data, self.byColumn)
        return self

    def transform(self, data):
        return self.transform_data_(data, self.normalize_)

    def inverse_transform(self, data):
        return self.transform_data_(data, self.inverse_normalize_)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def normalize_(self, value, vmin, vmax):
        if vmax == vmin:
            return 0.0
        return (value - vmin) / (vmax - vmin)

    def inverse_normalize_(self, value, vmin, vmax):
        return value * (vmax - vmin) + vmin

    def transform_data_(self, data, ftn):
        outData = data.copy()
        for row in range(len(data)):
            for col in range(len(data.columns)):
                column = data.columns[col]

                if column == self.byColumn:
                    continue

                outData.iloc[row, col] = ftn(
                    data.iloc[row, col],
                    self.minMax.getMin(data[self.byColumn].values[row], column),
                    self.minMax.getMax(data[self.byColumn].values[row], column))
        return outData

def createPipeline(data, scale="minmax", components=None):
    steps = []
    pipeline = None

    if scale in ["mm", "minmax"]:
        steps.append(("scale", MinMaxScaler()))
    elif scale in ["std", "standard"]:
        steps.append(("scale", StandardScaler()))
    elif scale is not None:
        raise DataUtilException("Unsupported scale value '{}'".format(scale))

    if components is not None:
        steps.append(("pca", PCA(n_components=components)))

    if len(steps) > 0:
        pipeline = Pipeline(steps=steps).fit(data)

    return pipeline

def preprocessData(pipeline, trainingData, testData, copyColumns=False):
    newColumns = trainingData.columns if copyColumns else None
    trainingData2 = pd.DataFrame(pipeline.transform(trainingData), columns=newColumns, index=trainingData.index)
    testData2 = pd.DataFrame(pipeline.transform(testData), columns=newColumns, index=testData.index)
    return trainingData2, testData2

def makeSegColAggFtn(ftn, col, nRows):
    return lambda seg: np.array([ftn([seg[row][col] for row in range(nRows)])])

def makeSegRowAggFtn(ftn, nCols, nRows):
    return lambda seg: np.array(
        [ftn([seg[row][col] for row in range(nRows)]) for col in range(nCols)])

def makeSegSelectFtn(colBegin, colEnd, rowBegin, rowEnd):
    return lambda seg: np.array([seg[r][colBegin:colEnd] for r in range(rowBegin, rowEnd)])

def makeSegments(data, segmentOffset, segmentLength, flatten=True, aggFtn=None):
    values = np.asarray(data)
    segments = []
    for segIdx in range(0, len(values) - segmentLength + 1, segmentOffset):
        segment = np.copy(values[segIdx:(segIdx + segmentLength)])
        if aggFtn:
            segment = aggFtn(segment)
        if flatten:
            segment = segment.flatten()
        segments.append(segment)
    return np.array(segments)
