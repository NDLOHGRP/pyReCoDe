import numpy as np
import scipy

from pyrecode.structures import ReCoDeStructures
from pyrecode.utils.converters import get_centroids_2D_nb, get_summary_stats_nb, make_binary_map
import pyrecode.recode_compressors as compressors
from .params import InitParams, InputParams
from .recode_header import ReCoDeHeader
from .em_reader import MRCReader, SEQReader, emfile
from .misc import rc_cfg as rc, get_dtype_string
from .fileutils import read_file
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import math
from numba import jit
import c_recode
import warnings
import scipy.ndimage as nd


class ReCoDeWriter:

    def __init__(self, image_filename, dark_data=None, dark_filename='', output_directory='', input_params=None,
                 params_filename='', mode='batch', validation_frame_gap=-1, log_filename='recode.log', run_name='run',
                 verbosity=0, use_c=False, max_count=-1, chunk_time_in_sec=0, node_id=0, buffer_size_in_frames=10):
        """
        Validates and holds user specified parameters for initializing ReCoDe.

        Parameters
        ----------
        image_filename : string
            file to be processed. If processing in-memory data, this should be the desired output file name
        dark_data : numpy array
            calibration_data (required if calibration_filename is None),
            if both dark_data and calibration_filename are given, dark_data will be used and calibration_filename will be ignored
        dark_filename : string
            file containing calibration data (required if dark_data is None)
        output_directory : string
            location where processed data will be written to
        input_params : InputParams or dict
            object of type pyrecode.params.InputParams or a dictionary containing input parameters (required if params_file is None),
            if both input_params and params_filename are given, input_params will be used and params_filename will be ignored
        params_filename : string
            file containing input parameters (required if input_params is None)
        mode: string
            can be either 'batch' or 'stream', indicating offline and online processing modes respectively
        validation_frame_gap : int
            number of frames to skip before saving validation frames
        log_filename : string
            the name of the log file
        run_name : string
            the name used to identify this run in the log file
        verbosity : int
            0, 1 or 2
        use_c : boolean
            indicates whether the optimized C implementation should be used
        max_count : int
            maximum number of data chunks (files) to process when mode='stream', ignored when mode='batch'
        chunk_time_in_sec : int
            acquisition time of data chunks (files) when mode='stream', ignored when mode='batch'
        node_id : int
            index of the processing node / thread in a distributed / multiprocess call
        """

        # parse and validate initialization params
        self._init_params = InitParams(mode, output_directory, image_filename=image_filename,
                                       calibration_filename=dark_filename, params_filename=params_filename,
                                       validation_frame_gap=validation_frame_gap, log_filename=log_filename,
                                       run_name=run_name, verbosity=verbosity, use_c=use_c)

        # parse and validate input params
        if input_params is None:
            self._input_params = InputParams()
            self._input_params.load(Path(self._init_params.params_filename))
        else:
            self._input_params = input_params

        if not self._input_params.validate():
            raise ValueError('Invalid input params')

        # check if initialization and input params are consistent
        if self._init_params.use_c:
            if self._input_params.source_numpy_dtype != np.uint16 or self._input_params.target_numpy_dtype != np.uint16:
                raise ValueError('use_c=True can only be used if source and target dtypes are both unsigned 16-bit')

        # create ReCoDe header
        self._rc_header = ReCoDeHeader()
        self._rc_header.create(self._init_params, self._input_params, True)
        if self._input_params.source_file_type in [rc.FILE_TYPE_MRC, rc.FILE_TYPE_SEQ]:
            self._rc_header.set('source_header_length', 1024)
        else:
            self._rc_header.set('source_header_length', 0)
        self._rc_header.print()
        if not self._rc_header.validate():
            raise ValueError('Invalid ReCoDe header created')
        self._header = self._rc_header.as_dict()

        # load and validate calibration frame
        if dark_data is None:
            if self._input_params.calibration_file_type == rc.FILE_TYPE_MRC:
                _t = MRCReader(self._init_params.calibration_filename)

            elif self._input_params.calibration_file_type == rc.FILE_TYPE_SEQ:
                _t = SEQReader(self._init_params.calibration_filename)

            elif self._input_params.calibration_file_type == rc.FILE_TYPE_BINARY:
                _t = read_file(self._init_params.calibration_filename, self._header['ny'], self._header['nx'],
                              self._input_params.source_numpy_dtype)
            else:
                raise NotImplementedError("No implementation available for loading calibration file of type 'Other'")
            t = np.squeeze(_t[0])
        else:
            t = dark_data

        print('Source shape:', t.shape)
        if self._header['ny'] != t.shape[0] or self._header['nx'] != t.shape[1]:
            raise RuntimeError('Data and Calibration frames have different shapes')

        self._calibration_frame = t
        self._calibration_frame_p_threshold = self._calibration_frame + self._input_params.calibration_threshold_epsilon
        if self._input_params.calibration_file_type in [rc.FILE_TYPE_MRC, rc.FILE_TYPE_SEQ]:
            _t.close()

        # ensure calibration frame has the same data type as source
        self._src_dtype = self._input_params.source_numpy_dtype
        if self._calibration_frame.dtype != self._src_dtype:
            print("calibration data type =", self._calibration_frame.dtype, "source data type =", self._src_dtype)
            warnings.warn('Calibration data type not same as source. Attempting to cast.')
            self._calibration_frame = self._calibration_frame.astype(self._src_dtype)
            self._calibration_frame_p_threshold = self._calibration_frame_p_threshold.astype(self._src_dtype)

        self._node_id = node_id

        self._intermediate_file_name = None
        self._intermediate_file = None
        self._validation_file_name = None
        self._validation_file = None

        self._buffer_sz = None
        self._rct_buffer = None
        self._rct_buffer_fill_position = None
        self._available_buffer_space = None
        self._frame_sz = None
        self._frame_buffer = None
        self._chunk_offset = None
        self._num_frames_in_part = None
        self._n_bytes_in_binary_image = None

        # variable used to ensure source header is serialized only once, even though
        # _do_sanity_check is called once for each chunk
        self._is_first_chunk = True

        # variables used for counting on validation frames
        self._vc_struct = nd.generate_binary_structure(2, 2)
        self._vc_dose_rate = 0.0
        self._vc_roi = {'x_start': None, 'y_start': None, 'nx': None, 'ny': None}
        self._vc_n_pixels = None

        self._s = nd.generate_binary_structure(2, 2)

        self._structures = ReCoDeStructures(self._header)

        if self._init_params.use_c:
            self._c_reader = None
            self._pixvals = None
            self._packed_pixvals = None

        if self._input_params.compression_scheme == 1:
            import zstandard as zstd
            self._compressor_context = zstd.ZstdCompressor(level=self._input_params.compression_level,
                                                           write_content_size=False)
        else:
            self._compressor_context = None

        self._buffer_size_in_frames = buffer_size_in_frames

    def start(self):
        """"
        Prepare for processing based on available input parameters. Assume source data is not available at this moment.
        Create output files and internal buffers
        """

        # create part-file
        if self._init_params.mode == 'batch':
            base_filename = Path(self._init_params.image_filename).stem
        elif self._init_params.mode == 'stream':
            base_filename = self._init_params.run_name

        self._intermediate_file_name = os.path.join(self._init_params.output_directory, base_filename +
                                                    '.rc' + str(self._input_params.reduction_level) +
                                                    '_part' + '{0:03d}'.format(self._node_id))
        self._intermediate_file = open(self._intermediate_file_name, 'wb')

        # serialize ReCoDe header
        self._rc_header.serialize_to(self._intermediate_file)

        # create validation file
        if self._init_params.validation_frame_gap > 0:
            self._validation_file_name = os.path.join(self._init_params.output_directory, base_filename +
                                                      '_part' + '{0:03d}'.format(self._node_id) +
                                                      '_validation_frames.bin')
            self._validation_file = open(self._validation_file_name, 'wb')

        # create buffers to hold reduced_compressed data
        # best to ensure buffer size is large enough to hold the expected amount of data to be processed
        # by this thread for a single chunk
        _bytes_per_pixel = np.dtype(self._src_dtype).itemsize
        _n_pixels_in_frame = self._header['ny'] * self._header['nx']
        self._frame_sz = np.uint64(_n_pixels_in_frame) * _bytes_per_pixel
        self._frame_buffer = bytearray(self._frame_sz)
        self._n_bytes_in_binary_image = math.ceil(_n_pixels_in_frame / 8)

        self._buffer_sz = self._frame_sz * self._buffer_size_in_frames
        self._rct_buffer = bytearray(self._buffer_sz)
        self._rct_buffer_fill_position = -1
        self._available_buffer_space = self._buffer_sz

        if self._init_params.use_c:
            self._c_reader = c_recode.Reader()
            _max_sz = int(math.ceil((_n_pixels_in_frame * self._input_params.source_bit_depth * 1.0) / 8.0))
            self._pixvals = memoryview(bytearray(_n_pixels_in_frame * _bytes_per_pixel))
            self._packed_pixvals = memoryview(bytearray(_max_sz))

        self._chunk_offset = 0
        self._num_frames_in_part = 0

        # initialize validation counting parameters
        self._vc_roi['nx'] = min(self._header['nx'], 128)
        self._vc_roi['ny'] = min(self._header['ny'], 128)
        self._vc_roi['x_start'] = math.floor((self._header['nx'] - self._vc_roi['nx']) / 2.0)
        self._vc_roi['y_start'] = math.floor((self._header['ny'] - self._vc_roi['ny']) / 2.0)
        self._vc_n_pixels = self._vc_roi['nx'] * self._vc_roi['ny']

    def _do_sanity_checks(self, is_first_chunk, data=None):
        """
        Source data is now available. Validate if input params agree with data. 
        This function is called separately for each chunk by each thread.
        Load source header here.
        """
        # get source file headers
        if data is None:
            if self._input_params.source_file_type == rc.FILE_TYPE_MRC:
                self._source = MRCReader(self._init_params.image_filename)
                self._source_shape = self._source.shape

            elif self._input_params.source_file_type == rc.FILE_TYPE_SEQ:
                self._source = SEQReader(self._init_params.image_filename)
                self._source_shape = self._source.shape

            elif self._input_params.source_file_type == rc.FILE_TYPE_BINARY:
                self._source_shape = tuple([self._header['nz'], self._header['ny'],
                                            self._header['nx']])
            else:
                raise NotImplementedError("No implementation available for loading calibration file of type 'Other'")

            # serialize source header
            if self._is_first_chunk:
                self._source.serialize_header(self._intermediate_file)

        else:
            self._source = data
            self._source_shape = self._source.shape

        # Validate that user given header values agree with source (MRC or SEQ) Header values of source file
        if self._source_shape[1] != self._header['ny']:
            raise RuntimeError('Expected height does not match height in source file')

        if self._source_shape[2] != self._header['nx']:
            raise RuntimeError('Expected width does not match width in source file')

        if self._input_params.num_frames == -1:
            self._header['nz'] = self._source_shape[0]
        else:
            if self._input_params.num_frames > self._source_shape[0]:
                raise RuntimeError('Number of frames requested in config file is larger than available in source file')
            else:
                self._header['nz'] = self._input_params.num_frames

        # close source file to reduce read overhead
        if self._input_params.calibration_file_type in [rc.FILE_TYPE_MRC, rc.FILE_TYPE_SEQ]:
            self._source.close()

    def run(self, data=None):
        """
        Source data is now available. This function is called separately for each chunk by each thread.
        Each thread will independently read and process data
        """
        run_metrics = {}

        # do sanity checks
        self._do_sanity_checks(self._is_first_chunk, data)

        # buffer should be initialized to 0 only the first time run() is called
        if self._is_first_chunk:
            self._rct_buffer_fill_position = 0
            self._available_buffer_space = self._buffer_sz

        # turn off first chunkiness (this could be done later)
        if self._is_first_chunk:
            self._is_first_chunk = False

        # determine the number of available frames for this thread
        # this is probably should be done by the caller and passed as params frame_offset and available_frames
        if self._init_params.mode == 'batch':
            n_frames_in_chunk = self._input_params.nz
        elif self._init_params.mode == 'stream':
            n_frames_in_chunk = self._source_shape[0]
        else:
            raise ValueError("Invalid input params: mode. Can be 'batch' or 'stream'.")

        n_frames_per_thread = int(math.ceil((n_frames_in_chunk * 1.0) / (self._input_params.num_threads * 1.0)))
        frame_offset = self._node_id * n_frames_per_thread
        available_frames = min(n_frames_per_thread, n_frames_in_chunk - frame_offset)

        # read the thread-specific data from chunk into memory
        stt = datetime.now()

        if data is None:
            with emfile(self._init_params.image_filename, self._input_params.source_file_type, mode="r") as f:
                try:
                    # try to load the expected frames all at once. The no. of expected frames is based on the header,
                    # which may not be accurate. This will result in an index out of range exception
                    data = f[frame_offset:frame_offset + available_frames]
                except IndexError as e1:
                    # in case of index out of range exception, try to load one frame at a time
                    count = 0
                    frame_list = []
                    a = 1
                    while a is not None and count < available_frames:
                        try:
                            a = f[frame_offset + count]
                            frame_list.append(np.squeeze(a))
                            count += 1
                        except IndexError as e2:
                            a = None
                    data = np.asarray(frame_list)
                    available_frames = data.shape[0]
        else:
            data = data[frame_offset:frame_offset + available_frames]

        if data.dtype != self._src_dtype:
            warnings.warn('Source data type either not as specified or does not match params specs. Attempting to cast.')
            data = data.astype(self._src_dtype)
        run_metrics['run_data_read_time'] = datetime.now() - stt

        # map L2 statistics code to string
        _statistics = 'max'
        if self._header['reduction_level'] == 2:

            if self._header['L2_statistics'] == 1:
                _statistics = 'max'

            elif self._header['L2_statistics'] == 2:
                _statistics = 'sum'

        # map L4 centroiding scheme code to string
        _centroiding_scheme = 'weighted_average'
        if self._header['reduction_level'] == 4:

            if self._header['L4_centroiding'] == 1:
                _centroiding_scheme = 'weighted_average'

            elif self._header['L4_centroiding'] == 2:
                _centroiding_scheme = 'max'

            elif self._header['L4_centroiding'] == 3:
                _centroiding_scheme = 'unweighted'

        # process frames
        run_start = datetime.now()

        for count, frame in enumerate(data):

            absolute_frame_index = self._chunk_offset + frame_offset + count
            compressed_frame_length, _metrics, binary_frame = self._reduce_compress(frame, absolute_frame_index,
                                                                                    _statistics, _centroiding_scheme)
            # if buffer doesn't have enough space to hold new data offload buffer data to file
            if self._available_buffer_space < compressed_frame_length:
                self._offload_buffer()
                self._rct_buffer_fill_position = 0
                self._available_buffer_space = self._buffer_sz

            # copy the last received frame that was never written to buffer
            self._rct_buffer[self._rct_buffer_fill_position:self._rct_buffer_fill_position + compressed_frame_length] \
                                                                        = self._frame_buffer[:compressed_frame_length]

            self._rct_buffer_fill_position += compressed_frame_length
            self._available_buffer_space -= compressed_frame_length

            # serialize validation data
            if self._init_params.validation_frame_gap > 0:
                if absolute_frame_index % self._init_params.validation_frame_gap == 0:
                    # serialize validation frame
                    self._validation_file.write(frame.tobytes())
                    # count for dose rate estimation
                    _vc_frame = binary_frame[self._vc_roi['y_start']:self._vc_roi['y_start'] + self._vc_roi['ny'],
                                self._vc_roi['x_start']:self._vc_roi['x_start'] + self._vc_roi['nx']]
                    labeled_foreground, num_features = scipy.ndimage.measurements.label(_vc_frame,
                                                                                        structure=self._vc_struct)
                    self._vc_dose_rate = num_features / self._vc_n_pixels
                    if 'run_dose_rates' in run_metrics:
                        run_metrics['run_dose_rates'].append(self._vc_dose_rate)
                    else:
                        run_metrics['run_dose_rates'] = [self._vc_dose_rate]

            for key in _metrics:
                if key in run_metrics:
                    run_metrics[key] += _metrics[key]
                else:
                    run_metrics[key] = _metrics[key]

        self._chunk_offset += available_frames
        self._num_frames_in_part += available_frames

        run_metrics['run_time'] = datetime.now() - run_start
        run_metrics['run_frames'] = available_frames
        return run_metrics

    def _reduce_compress(self, frame, absolute_frame_index, _statistics=None, _centroiding_scheme=None):

        _run_metrics = {}
        start = datetime.now()
        stt = datetime.now()

        # binarization: common to all reduction levels
        binary_frame = frame > self._calibration_frame_p_threshold  # ensure dtypes match

        if self._header['reduction_level'] == 1:
            pixel_intensities = frame[binary_frame] - self._calibration_frame_p_threshold[binary_frame]

        elif self._header['reduction_level'] in [2, 4]:
            labeled_foreground, num_features = nd.measurements.label(binary_frame, structure=self._s)

            if self._header['reduction_level'] == 2:
                pixel_intensities = get_summary_stats_nb(labeled_foreground, frame, 0, self._header['target_dtype'],
                                                         _statistics)
            elif self._header['reduction_level'] == 4:
                centroids = get_centroids_2D_nb(labeled_foreground, binary_frame, frame, 0, method=_centroiding_scheme)
                binary_frame = make_binary_map(self._header['nx'], self._header['ny'], centroids)

        _run_metrics['frame_thresholding_and_counting_time'] = datetime.now() - stt

        # pack binary map: all levels (binary_frame)
        # packed_binary_frame = self._pack_binary_frame(binary_frame)
        stt = datetime.now()
        packed_binary_frame = bytearray(_pack_binary_frame(binary_frame, self._n_bytes_in_binary_image))
        _run_metrics['frame_binary_image_packing_time'] = datetime.now() - stt

        # pack intensities: L1 (pixel_intensities) and L2 (summary_stats)
        stt = datetime.now()
        if self._header['reduction_level'] in [1, 2]:

            if self._input_params.source_bit_depth % 8 == 0:
                packed_pixel_intensities = pixel_intensities.tobytes()
            else:
                if self._init_params.use_c:
                    n_pixels = len(pixel_intensities)
                    n_packed = int(math.ceil((n_pixels * self._input_params.source_bit_depth * 1.0) / 8.0))
                    b = pixel_intensities.tobytes()
                    self._pixvals[:len(b)] = b
                    self._c_reader.bit_pack_pixel_intensities(n_packed, n_pixels, self._input_params.source_bit_depth,
                                                              self._pixvals, self._packed_pixvals)
                    packed_pixel_intensities = np.frombuffer(self._packed_pixvals, dtype=np.uint8, count=n_packed)
                else:
                    packed_pixel_intensities = _bit_pack(pixel_intensities, self._input_params.source_bit_depth)

            _n_bytes_in_packed_pixvals = len(packed_pixel_intensities)

        _run_metrics['frame_pixel_intensity_packing_time'] = datetime.now() - stt

        # Reduce Only
        if self._input_params.rc_operation_mode == 0:

            if self._header['reduction_level'] in [1, 2]:
                _frame_id = {'size': 4, 'value': absolute_frame_index.to_bytes(4, byteorder=sys.byteorder)}
                _md1 = {'size': 4, 'value': _n_bytes_in_packed_pixvals.to_bytes(4, byteorder=sys.byteorder)}
                _d1 = {'size': self._n_bytes_in_binary_image, 'value': packed_binary_frame}
                _d2 = {'size': _n_bytes_in_packed_pixvals, 'value': packed_pixel_intensities}
                compressed_frame_length = self._write_to_frame_buffer([_frame_id, _md1, _d1, _d2])

            elif self._header['reduction_level'] in [3, 4]:
                _frame_id = {'size': 4, 'value': absolute_frame_index.to_bytes(4, byteorder=sys.byteorder)}
                _d1 = {'size': self._n_bytes_in_binary_image, 'value': packed_binary_frame}
                compressed_frame_length = self._write_to_frame_buffer([_frame_id, _d1])

        # Reduce-Compress
        elif self._input_params.rc_operation_mode == 1:

            if self._header['reduction_level'] in [1, 2]:

                # compress
                stt = datetime.now()
                compressed_binary_frame = compressors.compress(
                    self._header['compression_scheme'], self._header['compression_level'], packed_binary_frame,
                    self._compressor_context)
                _run_metrics['frame_binary_image_compression_time'] = datetime.now() - stt

                stt = datetime.now()
                compressed_packed_pixel_intensities = compressors.compress(
                    self._header['compression_scheme'], self._header['compression_level'], packed_pixel_intensities,
                    self._compressor_context)
                _run_metrics['frame_pixel_intensity_compression_time'] = datetime.now() - stt

                # get compressed sizes
                _n_compressed_binary_frame = len(compressed_binary_frame)
                _n_compressed_packed_pixel_intensities = len(compressed_packed_pixel_intensities)

                _frame_id = {'size': 4, 'value': absolute_frame_index.to_bytes(4, byteorder=sys.byteorder)}
                _md1 = {'size': 4, 'value': _n_compressed_binary_frame.to_bytes(4, byteorder=sys.byteorder)}
                _md2 = {'size': 4, 'value': _n_compressed_packed_pixel_intensities.to_bytes(4, byteorder=sys.byteorder)}
                _md3 = {'size': 4, 'value': _n_bytes_in_packed_pixvals.to_bytes(4, byteorder=sys.byteorder)}
                _d1 = {'size': _n_compressed_binary_frame, 'value': compressed_binary_frame}
                _d2 = {'size': _n_compressed_packed_pixel_intensities, 'value': compressed_packed_pixel_intensities}

                compressed_frame_length = self._write_to_frame_buffer([_frame_id, _md1, _md2, _md3, _d1, _d2])

                """
                print('No. of foreground pixels', len(pixel_intensities),
                      '_n_bytes_in_packed_pixvals', _n_bytes_in_packed_pixvals,
                      '_n_compressed_binary_frame', _n_compressed_binary_frame,
                      '_n_compressed_packed_pixel_intensities', _n_compressed_packed_pixel_intensities)
                """

            elif self._header['reduction_level'] in [3, 4]:

                # compress
                stt = datetime.now()
                compressed_binary_frame = compressors.compress(
                    self._header['compression_scheme'], self._header['compression_level'], packed_binary_frame,
                    self._compressor_context)
                _run_metrics['frame_binary_image_compression_time'] = datetime.now() - stt

                # get compressed sizes
                _n_compressed_binary_frame = len(compressed_binary_frame)

                _frame_id = {'size': 4, 'value': absolute_frame_index.to_bytes(4, byteorder=sys.byteorder)}
                _md1 = {'size': 4, 'value': _n_compressed_binary_frame.to_bytes(4, byteorder=sys.byteorder)}
                _d1 = {'size': _n_compressed_binary_frame, 'value': compressed_binary_frame}

                compressed_frame_length = self._write_to_frame_buffer([_frame_id, _md1, _d1])

        else:
            raise ValueError('Unknown RC Operation Mode')

        _run_metrics['frame_time'] = datetime.now() - start

        return compressed_frame_length, _run_metrics, binary_frame

    def _write_to_frame_buffer(self, d):

        compressed_frame_length = 0
        for field in d:
            compressed_frame_length += field['size']

        if compressed_frame_length > self._frame_sz:
            raise ValueError('Buffer size smaller than compressed data size')

        index = 0
        for field in d:
            sz = field['size']
            self._frame_buffer[index:index+sz] = field['value']
            index += sz

        return compressed_frame_length

    def _pack_binary_frame(self, binary_frame):
        x = bytearray(self._n_bytes_in_binary_image)
        count = 0
        index = 0
        for b in binary_frame.flatten():
            if b == 1:
                x[count] += pow(2, index) * b
            index += 1
            if index == 8:
                count += 1
                index = 0
        return x

    def close(self):
        # clear buffer (send remaining data to file)
        self._offload_buffer()

        # serialize the true number of frames and the process id ReCoDe header: self._num_frames_in_part
        self._rc_header.update('nz', self._num_frames_in_part)
        self._intermediate_file.seek(0)
        self._rc_header.serialize_to(self._intermediate_file)

        # close part file
        self._intermediate_file.close()

        # close validation file
        if self._init_params.validation_frame_gap > 0:
            self._validation_file.close()

    def _offload_buffer(self):
        self._intermediate_file.write(self._rct_buffer[:self._rct_buffer_fill_position])
        self._intermediate_file.flush()


def print_run_metrics(run_metrics):
    for key in run_metrics:
        if key.startswith('frame_'):
            print(key, "\t", run_metrics[key] / run_metrics['run_frames'], "\t",
                  run_metrics[key] / run_metrics['frame_time'])
        else:
            if key == 'run_dose_rates':
                print(key, "\t", run_metrics[key], "\t", 'Avg.=', np.mean(run_metrics[key]))
            else:
                print(key, "\t", run_metrics[key])


@jit(nopython=True)
def _pack_binary_frame(binary_frame, n_bytes_in_binary_image):
    x = np.zeros((n_bytes_in_binary_image), dtype=np.uint8)
    count = 0
    index = 0
    for b in binary_frame.flatten():
        if b == 1:
            x[count] |= (1 << index)
        index += 1
        if index == 8:
            count += 1
            index = 0
    return x


@jit(nopython=True)
def _bit_pack(pixel_intensities, source_bit_depth):
    n_pixels = len(pixel_intensities)
    n_packed = int(math.ceil((n_pixels * source_bit_depth * 1.0) / 8.0))
    packed = np.zeros(n_packed, dtype=np.uint8)
    bp = 0  # bit position in jth byte of target byte array
    j = 0  # byte number in target byte array
    for pixval in pixel_intensities:
        for i in range(source_bit_depth):
            if pixval & (1 << i):
                packed[j] = packed[j] | (1 << bp)
            bp += 1
            if bp == 8:
                j += 1
                bp = 0
    return packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReCoDe Queue Manager')
    parser.add_argument('--image_filename', dest='image_filename', action='store', default='',
                        help='path of folder containing data (typically inside RAM disk for on-the-fly)')
    parser.add_argument('--calibration_file', dest='calibration_file', action='store', default='',
                        help='path to calibration file')
    parser.add_argument('--out_dir', dest='out_dir', action='store', default='', help='output directory')
    parser.add_argument('--params_file', dest='params_file', action='store', default='', help='path to params file')
    parser.add_argument('--mode', dest='mode', action='store', default='batch', help='batch or stream')
    parser.add_argument('--validation_frame_gap', dest='validation_frame_gap', action='store', type=int, default=-1,
                        help='validation frame gap')
    parser.add_argument('--log_file', dest='log_file', action='store', default='', help='path to log file')
    parser.add_argument('--run_name', dest='run_name', action='store', default='run_1', help='run name')
    parser.add_argument('--verbosity', dest='verbosity', action='store', type=int, default=0, help='verbosity level')
    parser.add_argument('--use_c', dest='use_c', action='store_true', help='')
    parser.add_argument('--max_count', dest='max_count', action='store', type=int, default=1,
                        help='the number of chunks to process')
    parser.add_argument('--chunk_time_in_sec', dest='chunk_time_in_sec', action='store', type=int, default=1,
                        help='seconds of data contained in each chunk')

    args = parser.parse_args()

    writer = ReCoDeWriter(
        args.image_filename,
        args.calibration_file,
        output_directory=args.out_dir,
        params_filename=args.params_file,
        mode='batch',
        validation_frame_gap=-1,
        log_filename='recode.log',
        run_name='run',
        verbosity=0, use_c=False, max_count=-1, chunk_time_in_sec=0, node_id=0)

    writer.start()
    run_metrics = writer.run()
    writer.close()
    print(run_metrics)

    """
    print(self._input_params.source_bit_depth)
    a = np.random.randint(0, high=4096, size=10)
    pa = _bit_pack(a, 12)
    print(pa)
    """
