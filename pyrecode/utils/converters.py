import numpy as np
from scipy.sparse import coo_matrix
import scipy.ndimage as nd
from numba import jit
from datetime import datetime

def recalibrate_L1 (L1_frames, dtype, n_Frames=-1, original_calibration_frame=None, new_calibration_frame=None, epsilon=0.0):

    if n_Frames < 1:
        n_Frames = len(L1_frames)

    fid = list(L1_frames.keys())[0]
    sz = L1_frames[fid].shape

    calibration_diff = original_calibration_frame.astype(np.float32) - (new_calibration_frame.astype(np.float32) + epsilon)

    if np.dtype(dtype).kind in ['u','i']:
        _min = np.iinfo(dtype).min
        _max = np.iinfo(dtype).max
    elif np.dtype(dtype).kind in ['f']:
        _min = np.finfo(dtype).min
        _max = np.finfo(dtype).max

    print(_min, _max)

    L1_recalibrated = {}
    start=datetime.now()
    for key in range(n_Frames):
        if key in L1_frames:
            frame = L1_frames[key].todense()
            f = frame.astype(np.float32)
            frame = f + calibration_diff
            frame[frame<_min] = _min
            frame[frame>_max] = _max
            frame = frame.astype(dtype)
            L1_recalibrated[key] = coo_matrix(frame, dtype=np.uint16)

    print ('Total processing time: ' + str(datetime.now()-start))
    return L1_recalibrated


def L1_to_L4 ( L1_frames, n_Frames=-1, area_threshold=0, verbosity=0):
    """
    To Do:
    1. centroiding scheme option for _get_centroids_2D and implementing 'max' centroiding scheme. For 'unweighted' centroiding scheme set frame = b_frame
    2. multithreading support
    3. support for RecodeReader object in addition to dictionary of frames
    """

    if n_Frames < 1:
        n_Frames = len(L1_frames)

    fid = list(L1_frames.keys())[0]
    sz = L1_frames[fid].shape
    _n_pixels_in_frame = sz[0]*sz[1]*1.
    t = np.ones(int(_n_pixels_in_frame), dtype=np.bool)
    
    s = nd.generate_binary_structure(2, 2)
    cd = {}

    stats_shown = False
    avg_dose_rate = 0.0
    frame_count = 0

    start = datetime.now()
    for key in range(n_Frames):
        if key in L1_frames:
            
            s1 = datetime.now()
            frame = L1_frames[key].todense()
            
            dark_subtracted_binary = frame > 0
            labeled_foreground, num_features = nd.measurements.label(dark_subtracted_binary, structure=s)
            sc = datetime.now()
            # centroids = _get_centroids_2D (labeled_foreground, dark_subtracted_binary, frame)
            centroids = _get_centroids_2D_nb(labeled_foreground, dark_subtracted_binary, frame, area_threshold)
            tc = datetime.now()-sc
            centroids = (np.round(centroids)).astype(np.uint16)
            if len(centroids) > 0:
                cd[key] = coo_matrix((t[:len(centroids)], (centroids[:,1], centroids[:,0])), shape=(sz[0], sz[1]), dtype=np.bool)
            else:
                cd[key] = coo_matrix((sz[0],sz[1]), dtype=np.bool)
            t1 = datetime.now()-s1

            if verbosity > 0:
                print(key)
                print('Dose Rate =', num_features/_n_pixels_in_frame)
                print('Processing time: ' + str(t1) + ', ' + str(tc))
            else:
                avg_dose_rate += num_features/_n_pixels_in_frame
                frame_count += 1
            
        if verbosity == 0:
            if key > 100 and not stats_shown:
                print('Avg. Dose Rate (First {0:d} frames) = {1:0.4f}'.format(frame_count, avg_dose_rate/frame_count))
                print('Processing time  (First ' + str(frame_count) + ' frames) = ' + str(datetime.now()-start))
                stats_shown = True

    print ('Total processing time: ' + str(datetime.now()-start))
    return cd


def _get_centroids_2D(labelled_image, b_frame, frame):
    
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
    for i,c in enumerate(centroids):
        centroid_arr[i,0] = centroids[c]['x'] / centroids[c]['w']
        centroid_arr[i,1] = centroids[c]['y'] / centroids[c]['w']

    return centroid_arr

'''
@jit(nopython=True)
def _get_centroids_2D_nb (labelled_image, b_frame, frame):
    
    centroids = dict()
    centroids[0] = np.zeros(3, dtype=np.float32)

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
                else:
                    centroids[label] = np.zeros(3, dtype=np.float32)
                    centroids[label][0] = v*r
                    centroids[label][1] = v*c
                    centroids[label][2] = v
            
    centroid_arr = np.zeros((len(centroids)-1,2))
    for i,label in enumerate(centroids):
        if label > 0:
            centroid_arr[i-1,0] = centroids[label][0] / centroids[label][2]
            centroid_arr[i-1,1] = centroids[label][1] / centroids[label][2]

    return centroid_arr
'''


def get_centroids_2D_nb (labelled_image, b_frame, frame, area_threshold, method='weighted_average'):

    if method == 'weighted_average':
        return _get_centroids_2D_nb_w(labelled_image, b_frame, frame, area_threshold)
    elif method == 'weighted_average':
        return _get_centroids_2D_nb_u(labelled_image, b_frame, area_threshold)
    elif method == 'weighted_average':
        return _get_centroids_2D_nb_m(labelled_image, b_frame, frame, area_threshold)


@jit(nopython=True)
def _get_centroids_2D_nb_w (labelled_image, b_frame, frame, area_threshold):
    
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
def _get_centroids_2D_nb_u(labelled_image, b_frame, area_threshold):
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
def _get_centroids_2D_nb_m(labelled_image, b_frame, frame, area_threshold):
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

    c = _get_centroids_2D_nb (
        np.array(labelled_image, dtype=np.uint16), 
        np.array(test_b_frame, dtype=np.bool), 
        np.array(test_frame, dtype=np.uint16)
    )
    start = datetime.now()
    for i in range(500):
        c = _get_centroids_2D_nb (
            np.array(labelled_image, dtype=np.uint16), 
            np.array(test_b_frame, dtype=np.bool), 
            np.array(test_frame, dtype=np.uint16),
        )
    end = datetime.now()
    print("Elapsed (after compilation) = %s" % (end - start))
    print(c)
    