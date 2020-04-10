import unittest
import numpy as np
import misc_utils

def fequal(lhs, rhs):
    return abs(lhs - rhs) <= 0.0000001

def aequal(lhsArray, rhsArray):
    return np.all(lhsArray == rhsArray)

def TempConvertEqual(ftn):
    return lambda srcT, destT: fequal(ftn(srcT), destT)

class TestMiscUtils(unittest.TestCase):
    def testProfitScore(self):
        ps = lambda a, p: misc_utils.profitScore(a, p, calcSign=False)
        self.assertTrue(fequal(ps([0.0], [0.0]), 0.0))
        self.assertTrue(fequal(ps([0.0], [0.01]), 0.0001))
        self.assertTrue(fequal(ps([0.0], [-0.01]), 0.0001))

        ps = lambda a, p: misc_utils.profitScore(a, p, calcSign=False)
        self.assertTrue(fequal(ps([0.0, 0.1], [0.0, 0.1]), 0.0))
        self.assertTrue(fequal(ps([0.0, 0.1], [0.01, 0.11]), 0.00020011))
        self.assertTrue(fequal(ps([0.0, 0.1], [-0.01, 0.09]), 0.00020011))

    def testProfitScoreSign(self):
        ps = lambda a, p: misc_utils.profitScore(a, p, calcSign=True)
        self.assertTrue(fequal(ps([0.0], [0.0]), 0.0))
        self.assertTrue(fequal(ps([0.0], [0.01]), -0.0001)) # Should this be -ve?
        self.assertTrue(fequal(ps([0.0], [-0.01]), -0.0001))
        self.assertTrue(fequal(ps([0.1], [0.1]), 0.0))
        self.assertTrue(fequal(ps([0.1], [0.11]), 0.0001))
        self.assertTrue(fequal(ps([0.1], [0.09]), 0.0001))

    def testKelvinToCelcius(self):
        eql = TempConvertEqual(misc_utils.convertK2C)
        self.assertTrue(eql(0.0, -273.15))
        self.assertTrue(eql(150.0, -123.15))
        self.assertTrue(eql(273.15, 0.0))
        self.assertTrue(eql(300.0, 26.85))

    def testCelciusToKelvin(self):
        eql = TempConvertEqual(misc_utils.convertC2K)
        self.assertTrue(eql(-273.15, 0.0))
        self.assertTrue(eql(-123.15, 150.0))
        self.assertTrue(eql(0.0, 273.15))
        self.assertTrue(eql(26.85, 300.0))

    def testCelciusToFahrenheit(self):
        eql = TempConvertEqual(misc_utils.convertC2F)
        self.assertTrue(eql(-20.0, -4.0))
        self.assertTrue(eql(-5.0, 23.0))
        self.assertTrue(eql(0.0, 32.0))
        self.assertTrue(eql(10.0, 50.0))
        self.assertTrue(eql(100.0, 212.0))

    def testFahrenheitToCelcius(self):
        eql = TempConvertEqual(misc_utils.convertF2C)
        self.assertTrue(eql(-4.0, -20.0))
        self.assertTrue(eql(23.0, -5.0))
        self.assertTrue(eql(32.0, 0.0))
        self.assertTrue(eql(50.0, 10.0))
        self.assertTrue(eql(212.0, 100.0))

    def testKelvinToFahrenheit(self):
        eql = TempConvertEqual(misc_utils.convertK2F)
        self.assertTrue(eql(0.0, -459.67))
        self.assertTrue(eql(150.0, -189.67))
        self.assertTrue(eql(273.15, 32.0))
        self.assertTrue(eql(300.0, 80.33))

    def testFahrenheitToKelvin(self):
        eql = TempConvertEqual(misc_utils.convertF2K)
        self.assertTrue(eql(-459.67, 0.0))
        self.assertTrue(eql(-189.67, 150.0))
        self.assertTrue(eql(32.0, 273.15))
        self.assertTrue(eql(80.33, 300.0))

    def testMakePrecedingPairs(self):
        values = np.array([1, 2, 3])
        self.assertTrue(aequal(misc_utils.makePrecedingPairs(values),
                               np.array([(0, 1), (1, 2), (2, 3)])))
        self.assertTrue(aequal(misc_utils.makePrecedingPairs(values, flatten=True),
                               np.array([0, 1, 1, 2, 2, 3])))

        values = np.array([-5, 0, 10])
        self.assertTrue(aequal(misc_utils.makePrecedingPairs(values),
                               np.array([(-6, -5), (-1, 0), (9, 10)])))
        self.assertTrue(aequal(misc_utils.makePrecedingPairs(values, flatten=True),
                               np.array([-6, -5, -1, 0, 9, 10])))

if __name__ == "__main__":
    unittest.main()
