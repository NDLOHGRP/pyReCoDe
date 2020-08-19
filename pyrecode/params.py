from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from .misc import map_dtype


class InitParams:

    def __init__(self, mode, output_directory, image_filename='', directory_path='', calibration_filename='',
                 params_filename='', validation_frame_gap=-1, log_filename='recode.log', run_name='run',
                 verbosity=0, use_c=False, max_count=-1, chunk_time_in_sec=0):
        """
        Validates and holds user specified parameters for initializing ReCoDe.

        Parameters
        ----------
        mode : string
            run mode can be 'batch' for offline processing and 'stream' for online processing
        verbosity : int
            0, 1 or 2
        validation_frame_gap : int
            number of frames to skip before saving validation frames
        image_filename : string
            file to be processed when mode is 'batch', ignored when mode is 'stream'
        directory_path : string
            folder to be processed when mode is 'stream', ignored when mode is 'batch'
        calibration_filename : string
            file containing calibration data (required)
        params_filename : string
            file containing input parameters (required)
        output_directory : string
            location where processed data will be written to
        log_filename : string
            the name of the log file
        run_name : string
            the name used to identify this run in the log file
        use_c : boolean
            indicates if the optimized C implementation will be used
        max_count : int
            maximum number of chunks to process when mode is 'stream', ignored when mode is 'batch'
        chunk_time_in_sec : int
            size of each data chunk in seconds when mode is 'stream', ignored when mode is 'batch'
            used to estimate the number of frames in chunk, based on frame rate in params file
        """
        self._mode = mode.strip().lower()
        self._verbosity = verbosity
        self._validation_frame_gap = validation_frame_gap
        self._image_filename = image_filename
        self._calibration_filename = calibration_filename
        self._params_filename = params_filename
        self._output_directory = output_directory
        self._log_filename = log_filename
        self._run_name = run_name
        self._use_c = use_c
        self._directory_path = directory_path
        self._max_count = max_count
        self._chunk_time_in_sec = chunk_time_in_sec

        if not self._validate_init_params():
            self.show_usage()
            raise ValueError('Invalid initialization parameters')

    def validate(self):
        self._validate_init_params()

    def _validate_init_params(self):

        if self._output_directory == '':
            print('Output Directory cannot be empty')
            return False

        if self._mode not in ['batch', 'stream']:
            print("Unknown mode: mode can only be 'batch' or 'stream'")
            return False

        if self._mode == 'batch':
            if self._image_filename == '':
                print('Image filename cannot be empty')
                return False

        if self._mode == 'stream':
            """
            if self._directory_path == '':
                print("directory_path cannot be empty when mode is 'stream'")
                return False
                
            if self._max_count < 1:
                print("max_count cannot be less than 1 when mode is 'stream'")
                return False

            if self._chunk_time_in_sec < 1:
                print("chunk_time_in_sec cannot be be less than 1 when mode is 'stream'")
                return False
            """

        """
        if self._calibration_filename == '':
            print ('Calibration filename cannot be empty')
            return False
        """

        if self._verbosity > 2:
            self._verbosity = 2

        if self._verbosity < 0:
            self._verbosity = 0

        return True

    @property
    def mode(self):
        """Returns the run mode: rc or de
        """
        return self._mode

    @property
    def verbosity(self):
        """Returns the verbosity level: 0-2
        """
        return self._verbosity

    @property
    def validation_frame_gap(self):
        """Returns the gap between validation frames
        """
        return self._validation_frame_gap

    @property
    def image_filename(self):
        """Returns the path to the file to be processed
        """
        return self._image_filename

    @property
    def calibration_filename(self):
        """Returns the path to the file containing calibration data
        """
        return self._calibration_filename

    @property
    def params_filename(self):
        """Returns the path to input params file
        """
        return self._params_filename

    @property
    def output_directory(self):
        """Returns the output directory
        """
        return self._output_directory

    @property
    def log_filename(self):
        """Returns the name of the log file
        """
        return self._log_filename

    @property
    def run_name(self):
        """Returns the name used to identify this run in the log file
        """
        return self._run_name

    @property
    def use_c(self):
        """Returns indicator showing if the optimized C implementation will be used
        """
        return self._use_c

    @property
    def directory_path(self):
        """Returns the directory path to be monitored during online processing
        """
        return self._directory_path

    @property
    def max_count(self):
        """Returns the maximum number of data chunks to be processed in an online processing session
        """
        return self._max_count

    @property
    def chunk_time_in_sec(self):
        """Returns the the time length of data chunks in seconds
        """
        return self._chunk_time_in_sec

    @staticmethod
    def show_usage():
        print("See documentation at https://github.com/NDLOHGRP/pyReCoDe for usage details")


class InputParams:

    '''
    ToDo
    1. calibration_threshold_epsilon can be float; currently all inputs are assumed to be ints (see line 197)
    2. source_bit_depth and bit_depth should be replaced with data_types to support floats
    3. num_calibration_frames and calibration_frame_offset should be deprecated; ReCoDe should only accepts single frame calibration data
    4. keep_calibration_data should be renamed to append_calibration_data for clarity
    '''

    def __init__(self):
        self._param_map = {'reduction_level': -1, 'rc_operation_mode': -1, 'calibration_threshold_epsilon': -1,
                           'target_bit_depth': -1, 'source_bit_depth': -1, 'num_cols': -1, 'num_rows': -1,
                           'num_frames': -1, 'frame_offset': -1, 'num_calibration_frames': -1,
                           'calibration_frame_offset': -1, 'keep_part_files': -1, 'num_threads': -1,
                           'l2_statistics': -1, 'l4_centroiding': -1, 'compression_scheme': -1, 'compression_level': -1,
                           'source_file_type': -1, 'source_header_length': -1, 'keep_calibration_data': -1,
                           'calibration_file_type': -1, 'source_data_type': -1, 'target_data_type': -1,
                           'source_numpy_dtype': -1, 'target_numpy_dtype': -1}
        # 'source_numpy_dtype' and 'target_numpy_dtype' are not exposed externally, to be inferred from source_data_type
        # and source_bit_depth and target_data_type and target_bit_depth

    def load(self, params_filename):
        assert params_filename != '', 'Params filename missing'
        with open(params_filename) as fp:
            for line in fp:
                if line == '' or line == '\n' or line.startswith('#'):
                    continue
                else:
                    parts = line.split('=')
                    key = parts[0].strip().lower()
                    assert key in self._param_map, 'Unknown parameter: ' + key
                    self._param_map[key] = int(parts[1].strip().lower())

    def _validate_input_params(self):

        if self._param_map['reduction_level'] not in [1,2,3,4]:
            print('Reduction level must be 1, 2, 3 or 4')
            return False

        if self._param_map['rc_operation_mode'] not in [0,1] :
            print('RC Operation mode can be 0, 1 or 2')
            return False

        if self._param_map['calibration_threshold_epsilon'] == '':
            print('Calibration threshold (epsilon) cannot be empty')
            return False

        if self._param_map['source_bit_depth'] == -1 and self._param_map['source_file_type'] in [0,3]:
            print('Source bit depth cannot be empty when source filetype is binary/other')
            return False

        if self._param_map['num_cols'] == -1 and self._param_map['source_file_type'] in [0,3]:
            print('Number of columns cannot be empty when source filetype is binary/other')
            return False

        if self._param_map['num_rows'] == -1 and self._param_map['source_file_type'] in [0,3]:
            print('Number of rows cannot be empty when source filetype is binary/other')
            return False

        if self._param_map['num_frames'] == -1 and self._param_map['source_file_type'] in [0,3]:
            print('Number of frames cannot be empty when source filetype is binary/other')
            return False

        if not isinstance(self._param_map['frame_offset'], int):
            print('Frame offset should be an integer')
            return False

        if not isinstance(self._param_map['num_calibration_frames'], int):
            print('Number of calibration should be an integer')
            return False

        if not isinstance(self._param_map['calibration_frame_offset'], int):
            print('Calibration frame offset should be an integer')
            return False

        if self._param_map['keep_part_files'] not in [0,1]:
            print('Keep part files must be 0 or 1')
            return False

        if not isinstance(self._param_map['num_threads'], int):
            print('Number of threads should be an integer')
            return False

        if self._param_map['l2_statistics'] not in [0,1,2]:
            print('L2 statistics must be 0, 1 or 2')
            return False

        if self._param_map['l4_centroiding'] not in [0,1,2,3]:
            print('L4 centroiding must be 0, 1, 2 or 3')
            return False

        if self._param_map['compression_scheme'] not in [0,1,2,3,4,5,6,7,8,9,10,11]:
            print('Compression scheme must be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 or 11')
            return False

        if int(self._param_map['compression_level']) < 0 or int(self._param_map['compression_level']) > 22:
            print('Compression level can be from 0 - 22')
            return False

        if self._param_map['keep_calibration_data'] not in [0,1]:
            print('Keep dark data cannot be either 0 or 1')
            return False

        if self._param_map['source_file_type'] not in [0,1,2,3]:
            print('Source file type must be 0, 1, 2 or 3')
            return False

        if self._param_map['source_file_type'] in [0,3] and (self._param_map['source_header_length'] == -1 or not isinstance(self._param_map['source_header_length'], int)):
            print('Source Header Length cannot be empty or non-integer when source filetype is binary/other')
            return False

        if self._param_map['calibration_file_type'] not in [0,1,2,3]:
            print('Calibration filetype must be 0, 1, 2 or 3')
            return False

        '''
        if self._param_map['calibration_file_type'] in [0,3] and (self._param_map['dark_header_length'] == -1 or not isinstance(self._param_map['dark_header_length'], int)):
            print('Dark Header Length cannot be empty or non-integer when dark filetype is binary/other')
            return False
        '''

        if self._param_map['frame_offset'] < 0:
            self._param_map['frame_offset'] = 0

        if self._param_map['num_threads'] < 1:
            self._param_map['num_threads'] = 1

        if self._param_map['source_data_type'] not in [0,1,2]:
            print('Source data type must be 0, 1, or 2')
            return False

        if self._param_map['target_data_type'] not in [0,1,2]:
            print('Target data type must be 0, 1, or 2')
            return False

        if self._param_map['target_bit_depth'] == -1:
            self._param_map['target_bit_depth'] = self._param_map['source_bit_depth']
            
        self._param_map['source_numpy_dtype'] = map_dtype(self._param_map['source_data_type'],
                                                          self._param_map['source_bit_depth'])
        self._param_map['target_numpy_dtype'] = map_dtype(self._param_map['target_data_type'],
                                                          self._param_map['target_bit_depth'])

        # check user provided dtype can hold source bit and target bit depths

        # source_header_length is set by caller

        return True

    def serialize(self, filename):
        with open(filename, 'w') as f:
            for key in self._param_map:
                f.write(key + ' = ' + str(self._param_map[key]) + '\n')

    @property
    def reduction_level(self):
        """Returns the reduction level
        """
        return self._param_map['reduction_level']

    @reduction_level.setter
    def reduction_level(self, value):
        self._param_map['reduction_level'] = value

    @property
    def rc_operation_mode(self):
        """Returns the rc operation mode
        """
        return self._param_map['rc_operation_mode']

    @property
    def calibration_threshold_epsilon(self):
        """Returns the dark threshold epsilon
        """
        return self._param_map['calibration_threshold_epsilon']

    @property
    def target_bit_depth(self):
        """Returns the bit depth 
        """
        return self._param_map['target_bit_depth']

    @property
    def source_bit_depth(self):
        """Returns the source bit depth 
        """
        return self._param_map['source_bit_depth']

    @property
    def num_cols(self):
        """Returns the number of columns
        """
        return self._param_map['num_cols']

    @property
    def num_rows(self):
        """Returns the number of rows
        """
        return self._param_map['num_rows']

    @property
    def num_frames(self):
        """Returns the number of frames
        """
        return self._param_map['num_frames']

    @property
    def nx(self):
        """Returns the number of columns
        """
        return self._param_map['num_cols']

    @property
    def ny(self):
        """Returns the number of rows
        """
        return self._param_map['num_rows']

    @property
    def nz(self):
        """Returns the number of frames
        """
        return self._param_map['num_frames']

    @property
    def frame_offset(self):
        """Returns the frame offset
        """
        return self._param_map['frame_offset']

    @property
    def num_calibration_frames(self):
        """Returns number of dark frames
        """
        return self._param_map['num_calibration_frames']

    @property
    def calibration_frame_offset(self):
        """Returns dark frame offset
        """
        return self._param_map['calibration_frame_offset']

    @property
    def keep_part_files(self):
        """Returns keep part files
        """
        return self._param_map['keep_part_files']

    @property
    def num_threads(self):
        """Returns number of threads
        """
        return self._param_map['num_threads']

    @property
    def l2_statistics(self):
        """Returns L2 statistics
        """
        return self._param_map['l2_statistics']

    @property
    def l4_centroiding(self):
        """Returns L4 centroiding
        """
        return self._param_map['l4_centroiding']

    @property
    def L2_statistics(self):
        """Returns L2 statistics
        """
        return self._param_map['l2_statistics']

    @property
    def L4_centroiding(self):
        """Returns L4 centroiding
        """
        return self._param_map['l4_centroiding']

    @property
    def compression_scheme(self):
        """Returns compression scheme
        """
        return self._param_map['compression_scheme']

    @property
    def compression_level(self):
        """Returns compression level
        """
        return self._param_map['compression_level']

    @property
    def keep_calibration_data(self):
        """Returns indicator showing if calibration data is retained
        """
        return self._param_map['keep_calibration_data']

    @property
    def source_file_type(self):
        """Returns source file type
        """
        return self._param_map['source_file_type']

    @property
    def source_header_length(self):
        """Returns source header length
        """
        return self._param_map['source_header_length']

    @property
    def calibration_file_type(self):
        """Returns calibration file type
        """
        return self._param_map['calibration_file_type']

    '''
    @property
    def dark_header_length(self):
        """Returns dark header length
        """
        return self._param_map['dark_header_length']
    '''

    @property
    def source_data_type(self):
        """Returns numpy data type
        """
        return self._param_map['source_data_type']

    @property
    def target_data_type(self):
        """Returns numpy data type
        """
        return self._param_map['target_data_type']

    def validate(self):
        return self._validate_input_params()

    @num_cols.setter
    def num_cols(self, value):
        self._param_map['num_cols'] = value

    @num_rows.setter
    def num_rows(self, value):
        self._param_map['num_rows'] = value

    @num_frames.setter
    def num_frames(self, value):
        self._param_map['num_frames'] = value

    @nx.setter
    def nx(self, value):
        self._param_map['num_cols'] = value

    @ny.setter
    def ny(self, value):
        self._param_map['num_rows'] = value

    @nz.setter
    def nz(self, value):
        self._param_map['num_frames'] = value

    @source_data_type.setter
    def source_data_type(self, value):
        self._param_map['source_data_type'] = value

    @target_data_type.setter
    def target_data_type(self, value):
        self._param_map['target_data_type'] = value

    @property
    def source_numpy_dtype(self):
        return self._param_map['source_numpy_dtype']

    @property
    def target_numpy_dtype(self):
        return self._param_map['target_numpy_dtype']


if __name__== "__main__":

    ip = InputParams()
    ip.load('../../config/recode_params_3.txt')
    print(ip._param_map)
    # print(ip._validate_input_params())
    ip.reduction_level = 4
    ip.serialize('../../config/recode_params_L1.txt')