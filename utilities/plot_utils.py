from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

class PlotUtilException(Exception):
    pass

def switchToDarkBackground():
    plt.style.use('dark_background')

def switchToDefaultBackground():
    plt.style.use('default')

def plotCorrelation(data):
    pd.plotting.scatter_matrix(data)
    plt.show()

def plotCorrelationMatrix(data, colorMap="YlOrRd"):
    corr = data.corr(method="kendall")
    plt.matshow(corr, cmap=plt.get_cmap(colorMap))
    plt.show()

def plotCorrelationHeatmap(data, colorMap=None):
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cbar=True, cmap=colorMap)
    plt.show()

def plotData(data, xName, yName):
    plt.scatter(data[xName], data[yName])
    plt.grid(axis="both")
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.show()

def plotPredictVsActual(yPredict, yActual, overlay=False):
    x = range(0, len(yActual))
    yDataAndNameList = [(yPredict, "Prediction"), (yActual, "Actual")]
    titleText = "Prediction vs. Actual"

    if overlay:
        for yDataColorAndName in yDataAndNameList:
            plt.plot(x, yDataColorAndName[0], label=yDataColorAndName[1])
        plt.title(titleText)
        plt.grid(axis="both")
        plt.legend()
    else:
        index = 1
        for yDataColorAndName in yDataAndNameList:
            plt.subplot(2, 1, index)
            plt.plot(x, yDataColorAndName[0])
            plt.ylabel(yDataColorAndName[1])
            if index == 1:
                plt.title(titleText)
            plt.grid(axis="both")
            index += 1
    plt.show()

def plotConfusionMatrix(cm, cmap=None, labels=None):
    if labels is not None:
        if len(cm) != len(labels):
            raise(PlotUtilException(
                "plotConfusionMatrix: invalid labels length cm_len={} labels_len={}".format(
                    len(cm),
                    len(labels))))

    ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False, cmap=cmap)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.show()
