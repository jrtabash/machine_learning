import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

class MiscUtilException(Exception):
    pass

def toNPArray(data):
    dataType = type(data)
    if dataType == np.ndarray:
        return data
    elif dataType == pd.DataFrame:
        return data.values
    elif dataType == list:
        return np.array(data)

    raise(MiscUtilException("toNPArray: Unsupported data type '{}'".format(dataType)))

def profitScore(yActual, yPredict, calcSign=True):
    """ Calculates a profit score of predicted relative to actual.

    The smaller the absolute of the score the better.

    When calcSign=True, the function calculates a signed result for which the sign
    is based on the following actual change vs predicted change rules:

      Actual | Predicted | Sign
     --------+-----------+------
        +    |     +     |  +
        +    |     -     |  -
        -    |     +     |  -
        -    |     -     |  +

    The function can be used as a loss metric with make_scorer by using calcSign=False.

    """

    initialAmount = 100.0
    predictAmount = initialAmount
    actualAmount = initialAmount

    predictValues = toNPArray(yPredict).flatten()
    actualValues = toNPArray(yActual).flatten()

    if len(predictValues) != len(actualValues):
        raise(MiscUtilException("profitScore: yPredict and yActual must have same length"))

    for i in range(len(predictValues)):
        # This assumes predict amounts are percent change amounts
        # Example: 2.75 => 2.75% => 2.75 / 100.0
        #          So, amount => amount + (amount * 2.75 / 100.0)
        predictAmount += (predictValues[i] * predictAmount / 100.0)
        actualAmount += (actualValues[i] * actualAmount / 100.0)

    predictChange = predictAmount - initialAmount
    actualChange = actualAmount - initialAmount

    scoreSign = -1 if calcSign and np.sign(predictChange) != np.sign(actualChange) else 1

    return scoreSign * abs(predictChange - actualChange) / initialAmount

def makeProfitLossFtn():
    return make_scorer(profitScore, greater_is_better=False, calcSign=False)
