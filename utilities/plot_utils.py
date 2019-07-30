from matplotlib import pyplot as plt

plt.style.use('dark_background')

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

def plotPredictVsActual(yPredict, yActual, overlay=False):
    x = range(0, len(yActual))
    yDataColorAndNameList = [(yPredict, "c", "Prediction"), (yActual, "r", "Actual")]
    titleText = "Prediction vs. Actual"

    if overlay:
        for yDataColorAndName in yDataColorAndNameList:
            plt.plot(x, yDataColorAndName[0], yDataColorAndName[1], label=yDataColorAndName[2])
        plt.title(titleText)
        plt.grid(axis="both")
        plt.legend()
    else:
        index = 1
        for yDataColorAndName in yDataColorAndNameList:
            plt.subplot(2, 1, index)
            plt.plot(x, yDataColorAndName[0], yDataColorAndName[1])
            plt.ylabel(yDataColorAndName[2])
            if index == 1:
                plt.title(titleText)
            plt.grid(axis="both")
            index += 1
    plt.show()
