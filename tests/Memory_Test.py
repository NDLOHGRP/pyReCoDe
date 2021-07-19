import psutil
import numpy as np
from pyrecode.params import InputParams, InitParams
from pyrecode.recode_server import ReCoDeServer
from pyrecode.recode_reader import ReCoDeReader, merge_parts
from pyrecode.utils.converters import l1_to_l4_converter

if __name__ == "__main__":
    
    tag = 'test'
    similar = [False]*12
    mem_increase = [0.]*12
    shape = (1, 2048, 2048)
    calib_frame = np.zeros((shape[1], shape[2]), dtype=np.uint16)

    # Make randomised data sparse with only ones and zeroes
    data = np.random.randint(10000, size=shape, dtype=np.uint16)
    data[data != 1] = 0
    
    # Prepare original data in format that can be compared to RC4 
    original_data = np.transpose(data[0, :, :])

    # Run ReCoDe for each compression scheme
    for scheme in range(12):
        if scheme not in [0, 2, 5]: # The compression schemes outside this list may have errors - so skip them
            continue
         
        p0 = psutil.virtual_memory()[3]

        # ReCoDe files for each compression scheme is put into its own folder c0 to c11
        init_params = InitParams('batch', '../Memory_Test/c' + str(scheme), image_filename='test_data',
                                     validation_frame_gap=2, log_filename='../Memory_Test/c' + str(scheme) + '/recode.log',
                                     run_name=tag, verbosity=0, use_c=False)

        input_params = InputParams()
        input_params.load('../Memory_Test/recode_params_' + tag + '.txt')
        input_params.nx = shape[1]
        input_params.ny = shape[2]
        input_params.nz = shape[0]
        input_params.source_data_type = 0 # unsigned int
        input_params.target_data_type = 0 # unsigned int
        input_params._param_map['compression_scheme'] = scheme

        server = ReCoDeServer('batch')
        run_metrics = server.run(init_params, input_params=input_params, dark_data=calib_frame, data=data)

        # Merge intermediate files
        merge_parts('../Memory_Test/c' + str(scheme), 'test_data.rc1', 3)

        # Read ReCoDe file
        recode_file_name = '../Memory_Test/c' + str(scheme) + '/test_data.rc1'
        reader = ReCoDeReader(recode_file_name, is_intermediate=False)
        reader.open()
        recoded_frame = reader.get_next_frame()

        if recoded_frame == None:
            print(f"Frame from ReCoDe (compression scheme {scheme} is missing.")
            similar[scheme] = False
        else:
            # Convert RC1 frame to RC4
            l4_frame = l1_to_l4_converter(recoded_frame, (shape[1], shape[2]), n_frames=1)
            
            frame_id = list(l4_frame.keys())[0]
            coo_frame = l4_frame[frame_id]['data']
            recovered_data = coo_frame.toarray().astype(np.uint8)

            # Compare recovered data with original data
            similar[scheme] = np.allclose(recovered_data, original_data)

        # Check for memory increase
        mem_increase[scheme] = (p0 - psutil.virtual_memory()[3]) / (10**9)

    # Summarise test results
    passed_test = True
    error = ''
    
    for i in range(12):
        if similar[i] == False and i in [0, 2, 5]:
            passed_test = False
            error += f"Compression scheme {i}: ReCoDe data differs from original data.\n"
        if mem_increase[i] != 0:
            passed_test = False
            error += f"Compression scheme {i}: (Virtual) Memory leak of {mem_increase[i]} GB.\n"
    
    # Return the test result
    result = "Passed" if (passed_test == True) else "Failed"
    print(f"\nMemory test results: {result}\n{error}")