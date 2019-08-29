import numpy as np
import pandas as pd
import data_utils

#
# Gaps Processor
#
class GapsProcessor:
    class Action:
        CarryForward  = 0
        CarryBackward = 1
        FillAverage   = 2

    def __init__(self, action, columnIndex, findGapsFtn, gapRangeFtn):
        self.processGapFtn = self.makeProcessGapFtn(action)
        self.columnIndex = columnIndex
        self.findGapsFtn = findGapsFtn
        self.gapRangeFtn = gapRangeFtn

    def process(self, data):
        if self.processGapFtn is None:
            return data

        gaps = self.findGapsFtn(data)
        while len(gaps) > 0:
            begin, end = gaps[0]
            data = data_utils.insertRows(data, end, self.processGapFtn(data, begin, end))
            gaps = self.findGapsFtn(data)

        return data

    #
    # Below this point for class internal use
    #
    def makeProcessGapFtn(self, action):
        if action == self.Action.CarryForward:
            return self.processGap_carryForward
        elif action == self.Action.CarryBackward:
            return self.processGap_carryBackward
        elif action == self.Action.FillAverage:
            return self.processGap_fillAverage
        else:
            return None

    def getGapRange(self, data, begin, end):
        gapRange = self.gapRangeFtn(data.iloc[begin, self.columnIndex],
                                    data.iloc[end, self.columnIndex])
        return gapRange[1:]

    def processGap_carryForward(self, data, begin, end):
        rows = []
        for gapValue in self.getGapRange(data, begin, end):
            row = data.iloc[begin, :].values.copy()
            row[self.columnIndex] = gapValue
            rows.append(row)
        return rows

    def processGap_carryBackward(self, data, begin, end):
        rows = []
        for gapValue in self.getGapRange(data, begin, end):
            row = data.iloc[end, :].values.copy()
            row[self.columnIndex] = gapValue
            rows.append(row)
        return rows

    def processGap_fillAverage(self, data, begin, end):
        rows = []
        gapRange = self.getGapRange(data, begin, end)
        rangeMid = len(gapRange) / 2
        index = 0
        for gapValue in gapRange:
            row = self.processGap_makeAvrgRow(data.iloc[[begin, end], :].values, index > rangeMid)
            row[self.columnIndex] = gapValue
            rows.append(row)
            index += 1
        return rows

    def processGap_makeAvrgRow(self, rows, reachedMiddle):
        newRow = []
        for i in range(len(rows[0])):
            cellType = type(rows[0][i])
            if np.issubdtype(cellType, np.integer):
                newRow.append(np.floor(np.mean([rows[0][i], rows[1][i]])))
            elif np.issubdtype(cellType, np.floating):
                newRow.append(np.mean([rows[0][i], rows[1][i]]))
            else:
                newRow.append(rows[1][i] if reachedMiddle else rows[0][i])
        return np.array(newRow)
