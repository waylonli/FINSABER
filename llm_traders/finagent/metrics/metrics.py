import numpy as np

def ARR(ret):
    res = (np.cumprod(ret + 1.0)[-1] - 1.0) / ret.shape[0] * 252
    return res

def VOL(ret):
    res = np.std(ret)
    return res

def DD(ret):
    res = np.std(ret[np.where(ret<0, True, False)])
    return res

def MDD(ret):
    iter_ret = np.cumprod(ret + 1.0)
    peak = iter_ret[0]
    mdd = 0
    for value in iter_ret:
        if value > peak:
            peak =value
        dd = (peak - value)/peak
        if dd > mdd:
            mdd =dd
    return mdd

def SR(ret):
    res = 1.0 * np.mean(ret) * np.sqrt(ret.shape[0]) / np.std(ret)
    return res

def CR(ret, mdd):
    res = np.mean(ret) * 252 / mdd
    return res

def SOR(ret, dd):
    res = 1.0 * np.mean(ret) * 252 / dd
    return res