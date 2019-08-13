import numpy as np
import pandas as pd
import misc_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def createPipeline(data, scale="minmax", components=None):
    steps = []
    pipeline = None

    if scale == "minmax":
        steps.append(("scale", MinMaxScaler()))
    elif scale == "standard":
        steps.append(("scale", StandardScaler()))

    if components is not None:
        steps.append(("pca", PCA(n_components=components)))

    if len(steps) > 0:
        pipeline = Pipeline(steps=steps).fit(data)

    return pipeline

def preprocessData(pipeline, trainingData, testData, copyColumns=False):
    newColumns = trainingData.columns if copyColumns else None
    trainingData2 = pd.DataFrame(pipeline.transform(trainingData), columns=newColumns)
    testData2 = pd.DataFrame(pipeline.transform(testData), columns=newColumns)
    return trainingData2, testData2

def floatRange(begin, end, step=1.0):
    values = []
    if begin <= end:
        cur = begin
        while cur < end:
            values.append(cur)
            cur = cur + step
    return values

def makeSegments(data, segmentOffset, segmentLength, flatten=True, aggFtn=None):
    values = misc_utils.toNPArray(data)
    segments = []
    for segIdx in range(0, len(values) - segmentLength + 1, segmentOffset):
        segment = np.copy(values[segIdx:(segIdx + segmentLength)])
        if flatten:
            segment = segment.flatten()
        if aggFtn:
            segment = aggFtn(segment)
        segments.append(segment)
    return np.array(segments)
