import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

class MiscUtilException(Exception):
    pass

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

    predictValues = np.asarray(yPredict).flatten()
    actualValues = np.asarray(yActual).flatten()

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

def convertK2C(kelvin):
    return kelvin - 273.15

def convertC2F(celsius):
    return (celsius * 9 / 5) + 32

def convertF2C(fahrenheit):
    return (fahrenheit - 32) * 5 / 9

def convertC2K(celsius):
    return celsius + 273.15

def convertK2F(kelvin):
    return convertC2F(convertK2C(kelvin))

def convertF2K(fahrenheit):
    return convertC2K(convertF2C(fahrenheit))

def makePrecedingPairs(values, flatten=False):
    precedingPairs = np.array([(x - 1, x) for x in values])
    if flatten:
        precedingPairs = precedingPairs.flatten()
    return precedingPairs
