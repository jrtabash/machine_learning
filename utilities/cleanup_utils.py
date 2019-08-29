import numpy as np
import pandas as pd
import data_utils

#
# Gaps Processor
#
# This class is for filling missing data in which full observation
# rows are missing and nothing is marked as missing with NA.
#
# An example of such data might look something like this:
#
#             date_time    temp  rain_1h weather_main  traffic_volume
# 0 2012-10-03 03:00:00  281.09      0.0        Clear             367
# 1 2012-10-03 04:00:00  279.53      0.0        Clear             814
# 2 2012-10-03 05:00:00  278.62      0.0        Clear            2718
# 3 2012-10-03 06:00:00  278.23      0.0        Clear            5673
# 4 2012-10-03 09:00:00  282.48      0.0        Clear            6471
# 5 2012-10-03 10:00:00  283.12      0.0        Clear            5511
#
# Note at a first glance, the DataFrame looks ok, but two measurements
# or rows are missing, 07:00:00 and 08:00:00.
#
# Given the follwing:
#   columnIndex: 0 (column index of date_time column)
#   findGapsFtn: returning list of index pairs
#                e.g. [(3, 4)]
#   gapRangeFtn: returning a range of Timestamps between given date_time values
#                e.g. For date_time at index 3 and date_time at index 4 return
#                     [Timestamp('2012-10-03 06:00:00'),
#                      Timestamp('2012-10-03 07:00:00'),
#                      Timestamp('2012-10-03 08:00:00')]
#
# Then for each of the following actions, the following rows will be added:
# action: CarryForward
# rows:
# 4 2012-10-03 07:00:00  278.23      0.0        Clear            5673
# 5 2012-10-03 08:00:00  278.23      0.0        Clear            5673
#
# action: CarryBackward
# rows:
# 4 2012-10-03 07:00:00  282.48      0.0        Clear            6471
# 5 2012-10-03 08:00:00  282.48      0.0        Clear            6471
#
# action: FillAverage
# rows:
# 4 2012-10-03 07:00:00  280.35      0.0        Clear            6072
# 5 2012-10-03 08:00:00  280.35      0.0        Clear            6072
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
