import numpy as np
from datetime import datetime
import pims
import numba
from numba import jit


@jit(nopython=True, parallel=True)
def _hists_np(a, bins=np.arange(100)):
    print(a.shape)
    h = np.zeros((a.shape[0], len(bins) - 1))
    for i in numba.prange(a.shape[0]):
        h[i] = np.histogram(a[i], bins=bins)[0]
    return h


@jit(nopython=True, parallel=True)
def _median(a):
    print(a.shape)
    b = np.zeros(a.shape[0])
    for i in numba.prange(a.shape[0]):
        b[i] = np.median(a[i])
    return b


if __name__ == "__main__":

    nFrames = 3200  # 8-sec chunk @ 400fps, 4-sec chunk @ 800 fps
    nReps = 5
    # a = np.random.randint(0, high=4096, size=(512*4096, 1200), dtype=np.uint16)
    d = pims.open('/scratch/loh/abhik/For_KianFong/DE16_data/DE16/14-53-39.592_Gain_Ref_Dose_39_40_400fps.seq')
    a = np.zeros((nFrames, d[0].shape[0] * d[0].shape[1]), dtype=np.uint16)
    for frame_index in range(nFrames):
        a[frame_index] = d[frame_index].flatten()
        print(frame_index)

    a = a.transpose()
    print(a.shape)

    for rep in range(nReps):
        start = datetime.now()
        b = _median(a)
        print('Elapsed Time =', datetime.now() - start)

    h = _hists_np(a)  # warm up
    for _nBins in range(100, 500, 50):
        for rep in range(nReps):
            _bins = np.arange(0, _nBins + 1)
            start = datetime.now()
            h = _hists_np(a, bins=_bins)
            print('nBins =', _nBins, ' Elapsed Time =', datetime.now() - start)
