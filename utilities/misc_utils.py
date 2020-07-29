import numpy as np
from sklearn.metrics import make_scorer

class MiscUtilException(Exception):
    pass

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
