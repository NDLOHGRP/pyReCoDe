### ReCoDe Input Params File
The input params file is a text file with param_name = value mappings for the following parameters (see recode_params.txt). <b>All of the parameters listed below must be present in the input params file<b>

Parameters

|Parameter	|Description	|Value
|-----------|---------------|-----
|num_frames	|No. of frames to be processed. When running ReCoDe Server in 'stream' mode, this parameter is ignored. In 'batch' mode, if source file is MRCS/Sequence: if the user provided value is -1, the number of frames as per source file header is used, otherwise the user specified number of frames are read. If source file is not MRCS/Sequence this value must be accurately provided	|Integer
|num_rows	|Number of rows	|Integer
|num_cols	|Number of columns	|Integer
|rc_operation_mode	|Indicates if data should be only reduced or reduced and compressed	|0 = reduction only <br> 1 = reduction and compression
|reduction_level	|Data reduction level to be used	|1 = Level 1 <br> 2 = Level 2 <br> 3 = Level 3  <br> 4 = Level 4
|compression_scheme	|Compression algorithm to be used	|0 = zlib (default) <br> 1 = zstandard <br> 2 = lz4 <br> 3 = snappy <br> 4 = bzip <br> 5 = lzma <br> 6 = blosc + zlib <br> 7 = blosc + zstd <br> 8 = blosc + lz4 <br> 9 = blosc + snappy <br> 10 = blosclz <br> 11 = blosc_lz4hc
|compression_level	|Internal optimization level of compression algorithms	|Integer in range 0-22 (both inclusive). See individual compression algorithm documentation for details.
|source_bit_depth	|The number of bits needed to store the maximum intensity value in the source file (input to ReCode). This can be different from the bit-depth used in the source dat. For instance, a 16-bit dataset may store values only in the range (0-4095). In this case, the source_bit_depth can be 12.	|If source data type is 0 or 1, integer in the range (1-64). If source data type is 2, then 32 or 64.
|target_bit_depth	|The number of bits needed to store the maximum intensity value in the reduced compressed (ReCoDe) file. If this is different from source_bit_depth, the intensities are re-scaled to fit the target range.	|If target data type is 0 or 1, integer in the range (1-64). If target data type is 2, then 32 or 64.
|source_data_type	|The data type in the source file (input to ReCoDe)	|0 = Unsigned Integer <br> 1 = Signed Integer <br> 2 = Floating Point
|target_data_type	|The data type to be used in the output ReCoDe file	|Same codes as used for source_data_type
|num_threads	|The number of threads to be used by ReCoDe Server. Num_Threads = 1 implies no parallelism.	|Integer
|l4_centroiding	|The centroiding scheme to be used for L4. Only used if reduction_level is 4, ignored otherwise.	|0 = Use default (Weighted Centroid) <br> 1 = Weighted Centroids <br> 2 = Max. Pixel <br> 3 = Unweighted Centroids
|l2_statistics	|The summary statistic to be stored for L2. Only used if reduction_level is 2, ignored otherwise.	|0 = Use default (Max) <br> 1 = Max pixel intensity <br> 2 = Sum of pixel intensities
|calibration_threshold_epsilon	|Signal-Noise Calibration Parameter (S). See signal-noise calibration algorithm for details. Assumed to have the same data type as source data. 	|Numeric with same data type as source_data_type
|source_file_type	|The source (input) file type	|0 = Binary <br> 1 = MRCS <br> 2 = Sequence <br> 255 = Others
|source_header_length	|The source (input) file's header size in bytes. if source_file_type is 1 (MRCS file) or 2 (Sequence file) this input is ignored and the header size is automatically set to 1024 bytes	|Integer
|frame_offset	|Index of first frame to be read from the source data. If a non-zero value (N) is specified, ReCoDe skips the first N frames when reading the source file.	|Integer
|calibration_file_type	|The calibration file type	|0 = Binary <br> 1 = MRCS <br> 2 = Sequence <br> 255 = Others
|calibration_frame_offset	|Index of first used frame when generating calibration data. The value is stored in ReCoDe header for book keeping purposes. If the information is not available specify 0.	|Integer
|num_calibration_frames	|Number of frames used when generating calibration data. The value is stored in ReCoDe header for book keeping purposes. If the information is not available specify 0.	|Integer
|keep_calibration_data	|Indicates if the calibration frame should be stored at the end of the ReCoDe data stream.	|1 = store calibration frame at the end of the stream <br> 0 = do not store calibration frame
|keep_part_files	|Indicates whether the part files should be deleted after the merged .rcX file has been created (X=reduction_level).	|Reserved for future use. In the current implementation part files are not deleted by ReCoDe to maintain data redundancy. Users may manually delete part files if required.