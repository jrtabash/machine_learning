from matplotlib import pyplot as plt

def plotCorrelation(data):
    pd.plotting.scatter_matrix(data)
    plt.show()

def plotCorrelationMatrix(data, colorMap="YlOrRd"):
    corr = data.corr(method="kendall")
    plt.matshow(corr, cmap=plt.get_cmap(colorMap))
    plt.show()

def plotData(data, xName, yName):
    plt.scatter(data[xName], data[yName])
    plt.grid(axis="both")
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.show()

def plotPredictVsActual(yPredict, yActual):
    x = range(0, len(yActual))
    index = 1
    for yDataColorAndName in [(yPredict, "b", "Prediction"), (yActual, "r", "Actual")]:
        plt.subplot(2, 1, index)
        plt.plot(x, yDataColorAndName[0], yDataColorAndName[1])
        plt.ylabel(yDataColorAndName[2])
        plt.grid(axis="both")
        index += 1
    plt.show()
