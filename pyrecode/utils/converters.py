import numpy as np
from scipy.sparse import coo_matrix
import scipy.ndimage as nd
from numba import jit
from datetime import datetime
import copy

"""
To Dos for recalibrate_l1 and l1_to_l4_converter:
1. multiprocessing support
2. support for RecodeReader object as input and output in addition to dictionary of frames
"""


def recalibrate_l1(l1_frames, n_frames=-1, original_calibration_frame=None, new_calibration_frame=None,
                   epsilon=0.0, in_place=False):

    if n_frames < 1:
        n_frames = len(l1_frames)

    calibration_diff = original_calibration_frame.astype(np.float64) - (new_calibration_frame.astype(np.float64) +
                                                                        epsilon)

    frame_id = list(l1_frames.keys())[0]
    dtype = l1_frames[frame_id]['data'].dtype
    if np.dtype(dtype).kind in ['u', 'i']:
        _min = np.iinfo(dtype).min
        _max = np.iinfo(dtype).max
    elif np.dtype(dtype).kind in ['f']:
        _min = np.finfo(dtype).min
        _max = np.finfo(dtype).max
    else:
        print(np.dtype(dtype).kind)
        raise ValueError("Unknown kind of frame dtype. Expected 'u', 'i', or 'f'.")

    l1_re_calibrated = {}
    start_time = datetime.now()
    for frame_count, key in enumerate(l1_frames):
        frame = l1_frames[key]['data'].todense()
        f = frame.astype(np.float64)
        frame = f + calibration_diff
        frame[frame < _min] = _min
        frame[frame > _max] = _max
        frame = frame.astype(dtype)

        if in_place:
            l1_re_calibrated[key] = l1_frames[key]
        else:
            _deep_copy_frame_metadata(l1_frames, l1_re_calibrated, key)
        l1_re_calibrated[key]['data'] = coo_matrix(frame, dtype=frame.dtype)

        if n_frames > 0 and n_frames == frame_count:
            break

    print ('Total processing time: ' + str(datetime.now()-start_time))
    return l1_re_calibrated


def l1_to_l4_converter(l1_frames, frame_shape, n_frames=-1, area_threshold=0, verbosity=0, method='weighted_average',
                       in_place=False):

    # get data type for centroids
    max_dim = np.max(frame_shape)
    _centroids_dtype = None
    for _d in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max_dim < np.iinfo(_d).max:
            _centroids_dtype = _d
            break
    if _centroids_dtype is None:
        raise ValueError("Unable to identify data type for centroids")

    _n_pixels_in_frame = frame_shape[0]*frame_shape[1]*1.
    t = np.ones(int(_n_pixels_in_frame), dtype=np.bool)
    
    s = nd.generate_binary_structure(2, 2)
    cd = {}

    stats_shown = False
    avg_dose_rate = 0.0

    start_time = datetime.now()
    for frame_count, key in enumerate(l1_frames):

        s1 = datetime.now()
        frame = l1_frames[key]['data'].todense()

        dark_subtracted_binary = frame > 0
        labeled_foreground, num_features = nd.measurements.label(dark_subtracted_binary, structure=s)
        sc = datetime.now()
        centroids = get_centroids_2D_nb(labeled_foreground, dark_subtracted_binary, frame, area_threshold, method)
        tc = datetime.now() - sc
        centroids = (np.round(centroids)).astype(_centroids_dtype)

        if in_place:
            cd[key] = l1_frames[key]
        else:
            _deep_copy_frame_metadata(l1_frames, cd, key)

        if len(centroids) > 0:
            cd[key]['data'] = coo_matrix((t[:len(centroids)], (centroids[:, 1], centroids[:, 0])),
                                         shape=(frame_shape[0], frame_shape[1]), dtype=np.bool)
        else:
            cd[key]['data'] = coo_matrix((frame_shape[0], frame_shape[1]), dtype=np.bool)
        t1 = datetime.now() - s1

        if verbosity > 0:
            print(key)
            print('Dose Rate =', num_features / _n_pixels_in_frame)
            print('Processing time: ' + str(t1) + ', ' + str(tc))
        else:
            avg_dose_rate += num_features / _n_pixels_in_frame
            
        if verbosity == 0:
            if key > 100 and not stats_shown:
                print('Avg. Dose Rate (First {0:d} frames) = {1:0.4f}'.format(frame_count, avg_dose_rate/frame_count))
                print('Processing time  (First ' + str(frame_count) + ' frames) = ' + str(datetime.now()-start_time))
                stats_shown = True

        if n_frames > 0 and n_frames == frame_count:
            break

    print('Total processing time: ' + str(datetime.now()-start_time))
    return cd


def _deep_copy_frame_metadata(src, target, frame_id):
    target[frame_id] = {}
    for key in src[frame_id]:
        if key is not 'data':
            target[frame_id][key] = copy.deepcopy(src[frame_id][key])


def _get_centroids_2d(labelled_image, b_frame, frame):
    
    n_cols = b_frame.shape[1]
    _pixels = np.argwhere(b_frame)

    centroids = {}
    for p in _pixels:
        L = labelled_image[p[0], p[1]]
        v = (frame[p[0], p[1]])*1.
        if L in centroids:
            centroids[L]['x'] += v*p[0]
            centroids[L]['y'] += v*p[1]
            centroids[L]['w'] += v
        else:
            centroids[L] = {'x':v*p[0], 'y':v*p[1], 'w':v}
            
    centroid_arr = np.zeros((len(centroids),2))
    for i, c in enumerate(centroids):
        centroid_arr[i,0] = centroids[c]['x'] / centroids[c]['w']
        centroid_arr[i,1] = centroids[c]['y'] / centroids[c]['w']

    return centroid_arr


def get_centroids_2D_nb(labelled_image, b_frame, frame, area_threshold, method='weighted_average'):

    if method == 'weighted_average':
        return _get_centroids_2d_nb_w(labelled_image, b_frame, frame, area_threshold)
    elif method == 'weighted_average':
        return _get_centroids_2d_nb_u(labelled_image, b_frame, area_threshold)
    elif method == 'weighted_average':
        return _get_centroids_2d_nb_m(labelled_image, b_frame, frame, area_threshold)


@jit(nopython=True)
def _get_centroids_2d_nb_w(labelled_image, b_frame, frame, area_threshold):
    
    centroids = dict()
    centroids[0] = np.zeros(4, dtype=np.float32)

    n_rows = b_frame.shape[0]
    n_cols = b_frame.shape[1]

    for r in range(n_rows):
        for c in range(n_cols):
            if b_frame[r][c]:
                label = labelled_image[r][c]
                v = frame[r][c]*1.
                if label in centroids:
                    centroids[label][0] += v*r
                    centroids[label][1] += v*c
                    centroids[label][2] += v
                    centroids[label][3] += 1        # area
                else:
                    centroids[label] = np.zeros(4, dtype=np.float32)
                    centroids[label][0] = v*r
                    centroids[label][1] = v*c
                    centroids[label][2] = v
                    centroids[label][3] = 1         # area
    
    centroids.pop(0, None)
    centroid_arr = [[centroids[label][0]/centroids[label][2], centroids[label][1]/centroids[label][2]]
                    for label in centroids if centroids[label][3] > area_threshold]

    return centroid_arr


@jit(nopython=True)
def _get_centroids_2d_nb_u(labelled_image, b_frame, area_threshold):
    centroids = dict()
    centroids[0] = np.zeros(4, dtype=np.float32)

    n_rows = b_frame.shape[0]
    n_cols = b_frame.shape[1]

    for r in range(n_rows):
        for c in range(n_cols):
            if b_frame[r][c]:
                label = labelled_image[r][c]
                if label in centroids:
                    centroids[label][0] += r
                    centroids[label][1] += c
                    centroids[label][2] += 1  # area
                else:
                    centroids[label] = np.zeros(3, dtype=np.float32)
                    centroids[label][0] = r
                    centroids[label][1] = c
                    centroids[label][2] = 1  # area

    centroids.pop(0, None)
    centroid_arr = [[centroids[label][0] / centroids[label][2], centroids[label][1] / centroids[label][2]]
                    for label in centroids if centroids[label][2] > area_threshold]

    return centroid_arr


@jit(nopython=True)
def _get_centroids_2d_nb_m(labelled_image, b_frame, frame, area_threshold):
    centroids = dict()
    centroids[0] = np.zeros(4, dtype=np.float32)

    n_rows = b_frame.shape[0]
    n_cols = b_frame.shape[1]

    for r in range(n_rows):
        for c in range(n_cols):
            if b_frame[r][c]:
                label = labelled_image[r][c]
                v = frame[r][c] * 1.
                if label in centroids:
                    if v > centroids[label][2]:
                        centroids[label][0] = r
                        centroids[label][1] = c
                        centroids[label][2] = v
                    centroids[label][3] += 1  # area
                else:
                    centroids[label] = np.zeros(4, dtype=np.float32)
                    centroids[label][0] = r
                    centroids[label][1] = c
                    centroids[label][2] = v
                    centroids[label][3] = 1  # area

    centroids.pop(0, None)
    centroid_arr = [[centroids[label][0], centroids[label][1]] for label in
                    centroids if centroids[label][3] > area_threshold]

    return centroid_arr


@jit(nopython=True)
def get_summary_stats_nb(labelled_image, frame, area_threshold, dtype, method='sum'):

    if method not in ['sum', 'max']:
        raise ValueError("Only allowed values for summary stats are: 'sum' and 'max'")

    stats = dict()
    stats[0] = np.zeros(4, dtype=dtype)

    areas = dict()
    areas[0] = np.zeros(4, dtype=np.uint16)

    n_rows = frame.shape[0]
    n_cols = frame.shape[1]

    for r in range(n_rows):
        for c in range(n_cols):
            label = labelled_image[r][c]
            if label:
                v = frame[r][c] * 1.
                if label in stats:
                    if method == 'sum':
                        stats[label] += v
                    elif method == 'max':
                        if v > stats[label]:
                            stats[label] = v
                    areas[label] += 1
                else:
                    stats[label] = v
                    areas[label] = 1

    stats.pop(0, None)
    areas.pop(0, None)
    stats_arr = [v[label] for label in stats if areas[label] > area_threshold]

    return stats_arr


@jit(nopython=True)
def make_binary_map(nx, ny, centroids):
    a = np.array((nx, ny), dtype=np.uint8)
    for centroid in centroids:
        row = 0
        col = 0
        np.round_(centroid[0], 0, row)
        np.round_(centroid[1], 0, col)
        a[row, col] = 1
    return a


def read_dark_ref(fname, shape, dtype):
    with open(fname, "rb") as binary_file:
        data = binary_file.read()
    a = np.frombuffer(data, dtype=dtype, count=shape[0]*shape[1])
    ref = np.reshape(a,shape)
    return ref


def apply_DE16_common_mode_correction(f):
    fc = f
    for c in range(0, f.shape[1], 256):
        fc[:, c:c+256:2] = f[:, c:c+256:2] - np.median(f[:, c:c+256:2])
        fc[:, c+1:c+256:2] = f[:, c+1:c+256:2] - np.median(f[:, c+1:c+256:2])
    return fc


if __name__== "__main__":

    test_frame = [
    [0,0,0,0,0,0,0,1,1],
    [0,1,1,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [1,0,0,1,0,0,0,0,0],
    [0,1,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1]]

    test_b_frame = [
    [0,0,0,0,0,0,0,1,1],
    [0,1,1,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [1,0,0,1,0,0,0,0,0],
    [0,1,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1]]

    labelled_image = [
    [0,0,0,0,0,0,0,1,1],
    [0,2,2,0,0,0,0,0,0],
    [0,2,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [3,0,0,4,0,0,0,0,0],
    [0,3,0,0,4,0,0,0,0],
    [0,0,3,0,0,0,0,0,0],
    [0,3,3,0,0,0,0,0,5],
    [3,0,0,0,0,0,0,5,5]]

    c = get_centroids_2D_nb (
        np.array(labelled_image, dtype=np.uint16), 
        np.array(test_b_frame, dtype=np.bool), 
        np.array(test_frame, dtype=np.uint16)
    )
    start = datetime.now()
    for i in range(500):
        c = get_centroids_2D_nb (
            np.array(labelled_image, dtype=np.uint16), 
            np.array(test_b_frame, dtype=np.bool), 
            np.array(test_frame, dtype=np.uint16),
        )
    end = datetime.now()
    print("Elapsed (after compilation) = %s" % (end - start))
    print(c)
