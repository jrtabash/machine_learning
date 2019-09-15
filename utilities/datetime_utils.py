import numpy as np
import pandas as pd
import misc_utils

class TimeStep:
    Nanosecond  = 1
    Microsecond = 1000 * Nanosecond
    Millisecond = 1000 * Microsecond
    Second      = 1000 * Millisecond
    Minute      = 60 * Second
    Hour        = 60 * Minute
    Day         = 24 * Hour
    Week        = 7 * Day

def calcDateTimeDiff(dateTime, step=TimeStep.Hour):
    df = dateTime.diff(periods=1)
    if type(df) == pd.DataFrame:
        df = df[df.columns[0]]
    df.values[0] = step
    df = pd.Series(np.float64(df.values) / step, dtype=np.float64)
    return df

def findDateTimeDuplicates(data, step, dateTimeColumn='date_time', calcPreceding=False, flatten=True):
    dups = calcDateTimeDiff(data[[dateTimeColumn]], step=step)
    dups = data[dups == 0.0].dropna().index.values
    if calcPreceding:
        dups = misc_utils.makePrecedingPairs(dups, flatten=flatten)
    return dups

def findDateTimeGaps(data, step, dateTimeColumn='date_time', calcPreceding=False, flatten=True):
    gaps = calcDateTimeDiff(data[[dateTimeColumn]], step=step)
    gaps = data[gaps > 1.0].dropna().index.values
    if calcPreceding:
        gaps = misc_utils.makePrecedingPairs(gaps, flatten=flatten)
    return gaps

def dateTimeRange(begin, end, step=TimeStep.Nanosecond):
    values = []
    if begin <= end:
        cur = begin
        while cur < end:
            values.append(cur)
            cur = cur + pd.Timedelta(step)
    return values
