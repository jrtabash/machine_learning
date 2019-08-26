import numpy as np
import pandas as pd
import misc_utils

class TimeDelta:
    Nanosecond  = 1
    Microsecond = 1000 * Nanosecond
    Millisecond = 1000 * Microsecond
    Second      = 1000 * Millisecond
    Minute      = 60 * Second
    Hour        = 60 * Minute
    Day         = 24 * Hour
    Week        = 7 * Day

def calcDateTimeDiff(dateTime, delta=TimeDelta.Hour):
    df = dateTime.diff(periods=1)
    if type(df) == pd.DataFrame:
        df = df[df.columns[0]]
    df.values[0] = delta
    df = pd.Series(df.values / delta, dtype=float)
    return df

def findDateTimeDuplicates(data, timeDelta, dateTimeColumn='date_time', calcPreceding=False):
    dups = calcDateTimeDiff(data[[dateTimeColumn]], delta=timeDelta)
    dups = data[dups == 0.0].dropna().index.values
    if calcPreceding:
        dups = misc_utils.makePrecedingPairs(dups, flatten=True)
    return dups

def findDateTimeGaps(data, timeDelta, dateTimeColumn='date_time', calcPreceding=False):
    gaps = calcDateTimeDiff(data[[dateTimeColumn]], delta=timeDelta)
    gaps = data[gaps > 1.0].dropna().index.values
    if calcPreceding:
        gaps = misc_utils.makePrecedingPairs(gaps, flatten=True)
    return gaps
