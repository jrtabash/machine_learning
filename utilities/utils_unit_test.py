import unittest
import numpy as np
import pandas as pd
import misc_utils
import data_utils

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

class TestDataUtils(unittest.TestCase):
    def testDataEncoder(self):
        df = pd.DataFrame({'A': ['11', '11', '22'], 'B': ['33', '44', '55']})
        de = data_utils.DataEncoder(['A', 'B'])
        self.assertEqual(de.getColumns(), ['A', 'B'])
        self.assertFalse(de.isOneHotEncoding())

        adf = de.encode(df)
        edf = pd.DataFrame({'A': [0, 0, 1], 'B': [0, 1, 2]})
        self.assertTrue(aequal(adf.values, edf.values))

        self.assertEqual(de.getLabel('A', 0), '11')
        self.assertEqual(de.getLabel('A', 1), '22')
        self.assertEqual(de.getLabel('B', 0), '33')
        self.assertEqual(de.getLabel('B', 1), '44')
        self.assertEqual(de.getLabel('B', 2), '55')
        self.assertEqual(de.getLabel('C', 0), '')

    def testDataEncoderOneHotEncoding(self):
        df = pd.DataFrame({'A': ['11', '11', '22'], 'B': ['33', '44', '55']})
        de = data_utils.DataEncoder(['A', 'B'], oneHotEncoding=True)
        self.assertEqual(de.getColumns(), ['A', 'B'])
        self.assertTrue(de.isOneHotEncoding())

        adf = de.encode(df)
        edf = pd.DataFrame(
            {'A_11': [1, 1, 0],
             'A_22': [0, 0, 1],
             'B_33': [1, 0, 0],
             'B_44': [0, 1, 0],
             'B_55': [0, 0, 1]})
        self.assertTrue(aequal(adf.values, edf.values))

        self.assertEqual(de.getLabel('A', 0), '')
        self.assertEqual(de.getLabel('A', 1), '')
        self.assertEqual(de.getLabel('B', 0), '')
        self.assertEqual(de.getLabel('B', 1), '')
        self.assertEqual(de.getLabel('C', 0), '')

    def testGroupMinMax(self):
        df = pd.DataFrame({'grp':  ['A', 'B', 'A', 'C', 'A', 'B'],
                           'col1': [  1,   2,   3,   4,   5,   6],
                           'col2': [-10, -20,   0, -40,  50, -60]})
        minMax = data_utils.GroupMinMax(df, 'grp')
        self.assertEqual(len(minMax), 3)
        self.assertTrue('A' in minMax)
        self.assertTrue('B' in minMax)
        self.assertTrue('C' in minMax)
        self.assertFalse('D' in minMax)
        self.assertEqual(minMax.getMin('A', 'col1'), 1)
        self.assertEqual(minMax.getMax('A', 'col1'), 5)
        self.assertEqual(minMax.getMin('A', 'col2'), -10)
        self.assertEqual(minMax.getMax('A', 'col2'), 50)
        self.assertEqual(minMax.getMin('B', 'col1'), 2)
        self.assertEqual(minMax.getMax('B', 'col1'), 6)
        self.assertEqual(minMax.getMin('B', 'col2'), -60)
        self.assertEqual(minMax.getMax('B', 'col2'), -20)
        self.assertEqual(minMax.getMin('C', 'col1'), 4)
        self.assertEqual(minMax.getMax('C', 'col1'), 4)
        self.assertEqual(minMax.getMin('C', 'col2'), -40)
        self.assertEqual(minMax.getMax('C', 'col2'), -40)

if __name__ == "__main__":
    unittest.main()
