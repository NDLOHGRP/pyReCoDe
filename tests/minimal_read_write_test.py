from pyrecode.recode_writer import ReCoDeWriter, print_run_metrics
from pyrecode.recode_reader import ReCoDeReader, merge_parts
from pyrecode.params import InputParams
import numpy as np

if __name__ == "__main__":

    """Write"""
    print("==========")
    print("Write Test")
    print("==========\n")
    _dtype = np.uint16
    _shape = (3, 512, 512)
    _tag = 'minimal_read_write_test'

    _data = np.random.randint(0, high=4096, size=_shape).astype(_dtype)
    calib_frame = np.zeros((_shape[1], _shape[2]), dtype=_dtype)

    _input_params = InputParams()
    _input_params.load('../config/recode_params_' + _tag + '.txt')
    _input_params.nx = _shape[1]
    _input_params.ny = _shape[2]
    _input_params.nz = _shape[0]
    _input_params.source_data_type = 0    # unsigned int
    _input_params.target_data_type = 0    # unsigned int

    writer = ReCoDeWriter(
        'test_data',
        dark_data=calib_frame,
        output_directory='../scratch',
        input_params=_input_params,
        mode='batch',
        validation_frame_gap=-1,
        log_filename='recode.log',
        run_name=_tag,
        verbosity=0, use_c=False, max_count=-1, chunk_time_in_sec=0, node_id=0)

    writer.start()
    run_metrics = writer.run(_data)
    writer.close()
    print("Run Metrics")
    print("-----------")
    print_run_metrics(run_metrics)

    """Read and Validate Intermediate Files"""
    print("\n===========================")
    print("Read Test: Intermediate Files")
    print("===========================\n")

    intermediate_file_name = '../scratch/test_data.rc1_part000'

    reader = ReCoDeReader(intermediate_file_name, is_intermediate=True)
    reader.open()
    test_passed = True
    for i in range(3):
        frame_data = reader.get_next_frame()
        if frame_data is not None:
            if np.sum(_data[i, :, :] - frame_data[i]['data'].todense()) > 0:
                test_passed = False
    reader.close()

    """Merge intermediate files"""
    print("\n========================")
    print("Merge Intermediate Files ")
    print("========================\n")
    merge_parts('../scratch', 'test_data.rc1', 1)

    """Read and Validate Merged Data"""
    print("\n======================")
    print("Read Test: ReCoDe File ")
    print("======================\n")

    recode_file_name = '../scratch/test_data.rc1'
    reader = ReCoDeReader(recode_file_name, is_intermediate=False)
    reader.open()
    test_passed = True
    for i in range(3):
        frame_data = reader.get_next_frame()
        if frame_data is not None:
            if np.sum(_data[i, :, :] - frame_data[i]['data'].todense()) > 0:
                test_passed = False
    reader.close()

    if test_passed:
        print("\nMinimal Read/Write Test Passed!!!")
    else:
        print("\nMinimal Read/Write Test Failed")
