import pandas as pd
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
