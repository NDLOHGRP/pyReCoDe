import numpy as np
from pyrecode.recode_server import ReCoDeServer
from pyrecode.recode_reader import ReCoDeReader, merge_parts
from pyrecode.params import InputParams, InitParams
from pyrecode.utils.converters import recalibrate_l1, l1_to_l4_converter

if __name__ == "__main__":

    """Create Mock Data for recalibration and conversion"""
    _dtype = np.uint16
    _shape = (9, 512, 512)
    _tag = 'minimal_read_write_test'

    # create mock data
    _data = np.random.randint(0, high=4096, size=_shape)

    # make data sparse
    _data = _data - 3500
    _data[_data < 0] = 0
    _data = _data.astype(_dtype)

    # create mock calibration data
    calib_frame = np.zeros((_shape[1], _shape[2]), dtype=_dtype)

    init_params = InitParams('batch', '../scratch', image_filename='test_data',
                             validation_frame_gap=2, log_filename='../scratch/recode.log',
                             run_name=_tag, verbosity=0, use_c=False)

    _input_params = InputParams()
    _input_params.load('../config/recode_params_' + _tag + '.txt')
    _input_params.nx = _shape[1]
    _input_params.ny = _shape[2]
    _input_params.nz = _shape[0]
    _input_params.source_data_type = 0  # unsigned int
    _input_params.target_data_type = 0  # unsigned int

    server = ReCoDeServer('batch')
    run_metrics = server.run(init_params, input_params=_input_params, dark_data=calib_frame, data=_data)

    # Merge intermediate files
    merge_parts('../scratch', 'test_data.rc1', 3)

    """Mock Data for recalibration and conversion is ready"""

    """Recalibration - One frame at a time"""

    # create a new more stringent calibration frame
    new_calib_frame = np.random.randint(0, high=10, size=(_shape[1], _shape[2]))

    # open file and read header
    recode_file_name = '../scratch/test_data.rc1'
    reader = ReCoDeReader(recode_file_name, is_intermediate=False)
    reader.open()
    header = reader.get_header().as_dict()
    frame_shape = (header['ny'], header['nx'])

    frames_dict = {}
    for i in range(_data.shape[0]):

        # load frame data
        frame_data = reader.get_next_frame()

        # recalibrate frame
        # use n_frames=1 to process one frame at a time
        re_calibrated_frame = recalibrate_l1(frame_data, n_frames=1, original_calibration_frame=calib_frame,
                                             new_calibration_frame=new_calib_frame, epsilon=1.0)
        # convert to L4
        # use n_frames=1 to process one frame at a time
        # use area_threshold=1 to remove 1 pixel puddles, only puddle with pixels > area_threshold are kept
        l4_frame = l1_to_l4_converter(re_calibrated_frame, frame_shape, n_frames=1, area_threshold=1, verbosity=1)

        # store frame in 'frames_dict' for next example
        frame_id = list(frame_data.keys())[0]
        frames_dict.update(frame_data)

    reader.close()

    """Recalibration - All frames together"""

    # recalibrate frames
    re_calibrated_frames = recalibrate_l1(frames_dict, original_calibration_frame=calib_frame,
                                          new_calibration_frame=new_calib_frame, epsilon=1.0)
    # convert to L4
    l4_frames = l1_to_l4_converter(re_calibrated_frames, frame_shape, area_threshold=1, verbosity=1)

