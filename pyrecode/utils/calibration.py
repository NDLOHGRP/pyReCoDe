import numpy as np
from datetime import datetime
import pims
import os
import argparse
import scipy.ndimage as nd
from datetime import datetime
import pathlib
import numba
from numba import jit
from scipy.optimize import curve_fit


def _save_file(arr, filename, dtype):
    arr.astype(dtype).tofile(filename)


def _count_events(frame, t):
    _binary_frame = frame > t
    s = nd.generate_binary_structure(2, 2)
    labeled_foreground, num_features = nd.measurements.label(_binary_frame, structure=s)
    return num_features, np.sum(_binary_frame)


@jit(nopython=True, parallel=True)
def _get_pixel_thresh_2(d, nx, ny, expected_n_events, t):
    acc_t = np.zeros((nx, ny), dtype=np.float32)
    for r in numba.prange(nx):
        for c in range(ny):
            rc = d[:, r, c].flatten()
            v = [pixval for pixval in d[:, r, c] if pixval > t[r, c]]
            top_values = np.ones((expected_n_events + 1), dtype=np.float32) * np.finfo(
                np.float32).min  # replace with np.finfo(np.float32).min
            for rep in range(expected_n_events + 1):
                top_value_index = -1
                for _index, pixval in enumerate(v):
                    if pixval >= top_values[rep]:
                        top_values[rep] = pixval
                        top_value_index = _index
                if top_value_index > -1:
                    v[top_value_index] = np.finfo(np.float32).min  # replace with np.finfo(np.float32).min
            top_values.sort()
            acc_t[r, c] = (top_values[0] + top_values[1]) / 2
    return acc_t


@jit(nopython=True, parallel=True)
def _median_std_nb(d, nx, ny):
    _med = np.zeros((nx, ny), dtype=np.float32)
    _std = np.zeros((nx, ny), dtype=np.float32)
    for r in numba.prange(nx):
        for c in range(ny):
            rc = d[:, r, c].flatten()
            _med[r, c] = np.median(rc)
            _std[r, c] = np.std(rc)
    return _med, _std


def _gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def _get_fit_params(d, nFrames, n_stats_frames, nx, ny, _m):
    # zero centre
    dsd = np.zeros((n_stats_frames, ny, nx))
    for i, f in enumerate(range(nFrames - n_stats_frames, nFrames)):
        dsd[i] = d[f] - _m

    # histogram
    h, edges = np.histogram(dsd.flatten(), bins=100, density=False)
    c = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    hn = h / np.sum(h)

    # initial params
    mean = np.average(c, weights=hn)
    sigma = np.sqrt(np.average((c - mean) ** 2, weights=hn))
    _p0 = [np.max(hn), mean, sigma]

    # fit
    popt, pcov = curve_fit(_gaussian, c, hn, p0=_p0)

    print("\n Fit Result \n Init params=", _p0, "\n Optimal params=", popt)
    return popt[2]


def make_calibration_frames(filepath, dtype, nFrames, n_stats_frames, n_sigmas, savepath='', filename_prefix='',
                            use_acc=False, sigma_acc=-1):
    if not filename_prefix.endswith('_'):
        filename_prefix += '_'

    start = datetime.now()
    fp = pims.open(filepath)
    nx, ny = fp[0].shape

    d = np.zeros((nFrames, nx, ny), dtype=dtype)
    for frame_index in range(nFrames):
        d[frame_index] = fp[frame_index]

    # _m = np.median(d, axis=0)
    # _std = np.std(d, axis=0)

    _m, _stds = _median_std_nb(d, nx, ny)
    _fit_std = _get_fit_params(d, nFrames, n_stats_frames, nx, ny, _m)
    print('\nAvg. std.dev. per pixel:', np.average(_stds))
    print('Global intensity std. dev.:', _fit_std)
    print("Calibration time:", datetime.now() - start, "\n")

    n_pixels_in_frame = nx * ny
    for i in range(n_sigmas):
        t = np.floor(_m + _fit_std * i).astype(dtype)
        _save_file(t, os.path.join(savepath, filename_prefix + "_dark_ref_" + str(i) + ".bin"), dtype)
        n_events = 0
        p_foreground_pixels = 0
        for f in range(nFrames - n_stats_frames, nFrames):
            n_e, n_fp = _count_events(d[f], t)
            n_events += n_e
            p_foreground_pixels += (n_fp / n_pixels_in_frame)
        avg_n_events = n_events / n_stats_frames
        avg_p_foreground_pixels = p_foreground_pixels / n_stats_frames
        print("Avg. prop. foreground pixels for sigma=" + str(i) + " is: " + str(avg_p_foreground_pixels))
        print("Avg. electron count for sigma=" + str(i) + " is: " + str(avg_n_events))
        print("Avg. dose rate for sigma=" + str(i) + " is: " + str(avg_n_events / n_pixels_in_frame))
        print("")

        if use_acc and i == sigma_acc:
            expected_n_events = int(np.ceil(nFrames * (avg_n_events / n_pixels_in_frame)))
            print(expected_n_events)
            if expected_n_events < 2:
                print("Unable to compute accurate thresholds: too few events in dataset")
            else:
                acc_t = _get_pixel_thresh_2(d, nx, ny, expected_n_events, _m)
                _save_file(acc_t, os.path.join(savepath, filename_prefix + "_dark_ref_" + str(i) + "A.bin"), dtype)
                print(acc_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReCoDe Calibration Routine')
    parser.add_argument('--flatfield_filepath', dest='filepath', action='store', default='',
                        help='path to the flat-field illuminated file to be used for calibration')
    parser.add_argument('--n_frames', dest='n_frames', action='store', type=int, default=100,
                        help='number of frames to use for calibration')
    parser.add_argument('--n_stats_frames', dest='n_stats_frames', action='store', type=int, default=10,
                        help='number of frames on which to estimate dose rate')
    parser.add_argument('--n_sigmas', dest='n_sigmas', action='store', type=int, default=4,
                        help='the number of sigmas to try')
    parser.add_argument('--savepath', dest='savepath', action='store', default='',
                        help='path to folder where calibration frames are to be stored')
    parser.add_argument('--save_prefix', dest='filename_prefix', action='store', default='',
                        help='prefix for calibration filename')
    parser.add_argument('--use_acc', dest='use_acc', action='store_true', help='do accurate calibration step')
    parser.add_argument('--sigma_acc', dest='sigma_acc', action='store', type=int, default=3,
                        help='which sigma to use for accurate calibration when use_acc=True')
    args = parser.parse_args()
    make_calibration_frames(
        str(pathlib.Path(args.filepath)),
        np.uint16,
        args.n_frames,
        args.n_stats_frames,
        args.n_sigmas,
        str(pathlib.Path(args.savepath)),
        args.filename_prefix,
        use_acc=True,
        sigma_acc=args.sigma_acc
    )
