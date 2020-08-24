import os
import math
import numpy as np
import sys
from scipy.sparse import coo_matrix
from collections import deque

from pyrecode.misc import get_dtype_string, map_dtype
import pyrecode.recode_compressors as compressors
from .recode_header import ReCoDeHeader
from .structures import ReCoDeStructures
import c_recode


class ReCoDeReader:
    def __init__(self, file, is_intermediate=False):
        self._source_filename = file
        self._current_frame_index = 0
        self._c_reader = c_recode.Reader()
        if is_intermediate:
            self._is_intermediate = 1
        else:
            self._is_intermediate = 0
        self._file_size = None
        self._header = None
        self._frame_metadata = None
        self._seek_table = None
        self._rc_header = None  # initialized in _load_header
        self._frame_data_start_position = 0  # the byte where frame data for frame 0 starts; initialized in _load_header
        self._frame_sz = 0  # size of frames in bytes; initialized in _load_header
        self._frame_buffer = None  # initialized in _load_header
        self._sz_frame_metadata = None  # initialized in _load_header
        self._n_elements_frame_metadata = None  # initialized in _load_header
        self._fp = None
        self._structures = None
        self._numpy_dtype = None
        self._decompressor_context = None

    def open(self, print_header=True):

        # load header (closes file after reading header)
        self._load_header(print_header)
        compressors.import_checks(self._header)

        # open file pointer again and get file size
        self._fp = open(self._source_filename, "rb")
        self._fp.seek(0, 2)
        self._file_size = self._fp.tell()
        self._fp.seek(0, 0)

        # initialize
        self._initialize()
        self._create_read_buffers()
        self._load_seek_table()

        self._numpy_dtype = map_dtype(self._header['target_dtype'], self._header['target_bit_depth'])

        # zstandard (zstd)
        if self._header['compression_scheme'] == 1:
            import zstandard as zstd
            self._decompressor_context = zstd.ZstdCompressor()

    '''
    def _open(self):
        self._c_reader._open_file(self._source_filename)
    '''

    def _load_header(self, print_header=True):
        self._rc_header = ReCoDeHeader()
        self._rc_header.load(self._source_filename)
        self._header = self._rc_header.as_dict()
        if print_header:
            self._rc_header.print()

    def _initialize(self):
        self._structures = ReCoDeStructures(self._header)

        # description of standard metadata (list of dicts)
        sm = self._structures.standard_frame_metadata_structure_for(self._header['reduction_level'],
                                                                    self._header['rc_operation_mode'])

        # list of non-standard metadata
        nsm = list(self._rc_header.non_standard_metadata_sizes.values())

        # size of metadata per frame
        non_std_metadata_size = np.sum(nsm)
        self._sz_frame_metadata = self._structures.get_standard_frame_metadata_size(self._header['reduction_level'],
                                                                                    self._header['rc_operation_mode']) \
                                  + non_std_metadata_size

        # number of elements in metadata per frame
        self._n_elements_frame_metadata = len(nsm) + len(sm)

        # get the starting position of frame data (start of frame 0 metadata for intermediate,
        # start of frame 0 data for recode file)
        self._frame_data_start_position = self._rc_header.get_frame_data_offset(self._is_intermediate,
                                                                                self._sz_frame_metadata)

        # size of sparse frames
        if self._header['reduction_level'] in [1, 2]:
            # for reduction levels 1 and 2, in the sparse frame each foreground pixel is associated with 3 values:
            # x,y,intensity
            n_elements = 3
        elif self._header['reduction_level'] in [3, 4]:
            # for reduction levels 1 and 2, in the sparse frame each foreground pixel is associated with 2 values:
            # x,y (coordinates)
            n_elements = 2

        # buffer where unpacked sparse format data is stored. the maximum bit depth of 64-bit is used to support all
        # data types. The data should be cast to the appropriate dtype (self._header['target_dtype']) after reading
        self._frame_sz = np.uint64(self._header["ny"]) * np.uint64(self._header["nx"]) * np.uint64(n_elements) * \
                         np.uint64(8)

        # buffer to hold sparse frame data
        self._frame_buffer = memoryview(bytes(self._frame_sz))
        if self._header['reduction_level'] == 2:
            self._summary_stat_buffer = memoryview(bytes(
                np.uint64(self._header["ny"]) * np.uint64(self._header["nx"]) * np.uint64(8)))

        return self._header

    def _create_read_buffers(self):
        if 'nz' not in self._header:
            raise ValueError('Attempting to set persistent variables before reading header')
        self._c_reader.create_buffers(self._header["ny"], self._header["nx"], self._header["target_bit_depth"])

    def _load_seek_table(self):

        """loads frame metadata for fast seeking of ReCoDe files not for intermediate files"""

        if not self._is_intermediate:

            # description of standard metadata (list of dicts)
            sm = self._structures.standard_frame_metadata_structure_for(self._header['reduction_level'],
                                                                        self._header['rc_operation_mode'])

            # dictionary of non-standard metadata
            nsm = self._rc_header.non_standard_metadata_sizes.values()

            n_metadata_elements = len(sm) + len(nsm)

            if 'nz' not in self._header:
                raise ValueError('Attempting to read seek table before reading header')

            self._frame_metadata = []

            # move to start of frame metadata
            # here we want to move to the start of metadata and not frame data,
            # so we explicitly set is_intermediate as True
            self._fp.seek(self._rc_header.get_frame_data_offset(True, self._sz_frame_metadata), 0)

            for _frame_index in range(self._header['nz']):
                d = {}
                for field in sm:
                    d[field['name']] = np.frombuffer(self._fp.read(field['bytes']), dtype=field['dtype'])[0]
                for field in nsm:
                    d_type = get_dtype_string(nsm[field])
                    d[field['name']] = np.frombuffer(self._fp.read(d_type.itemsize), dtype=d_type)
                self._frame_metadata.append(d)

            # create a seek table, that maintains the offset of each frame relative to self._frame_data_start_position
            # only available for ReCoDe files (not intermediate files)
            self._seek_table = np.zeros((self._header['nz'], 2), dtype=np.uint32)
            for _frame_index in range(self._header['nz']):
                self._seek_table[_frame_index, 0] = self._structures.get_frame_data_size(
                                                    self._header['reduction_level'], self._header['rc_operation_mode'],
                                                    self._frame_metadata[_frame_index])
            self._seek_table[1:, 1] = np.cumsum(self._seek_table[:-1, 0])

    def get_header(self):
        return self._rc_header

    def get_source_header(self):
        return self._rc_header.source_header

    def get_true_shape(self):
        return tuple([self._header["nz"], self._header["ny"], self._header["nx"]])

    def get_shape(self):
        return tuple([self._header["nz"], self._header["ny"], self._header["nx"]])

    def get_dtype(self):
        return self._header['target_dtype']

    def get_sub_volume(self, slice_z, slice_y, slice_x):
        raise NotImplementedError

    def get_frame(self, z):

        if self._is_intermediate:
            raise ValueError("Random acceess is not available for intermediate files")

        if z >= self._header['nz']:
            raise ValueError('Requested frame index is greater than number of frames in dataset')

        # move to start of requested frame's data
        s = self._frame_data_start_position
        self._fp.seek(s + self._seek_table[z, 1], 0)

        # safety check
        if self._file_size - self._fp.tell() == 0:
            sparse_d = None
        else:
            # load and decompress (if necessary)
            if self._header['reduction_level'] == 2:
                sparse_d, summary_stats = self._get_frame_sparse(self._frame_metadata[z])
            else:
                sparse_d = self._get_frame_sparse(self._frame_metadata[z])

        # pack and return
        if sparse_d is None:
            self._header['nz'] = self._current_frame_index
            return None
        else:
            if self._header['reduction_level'] == 2:
                frame_dict = {'metadata': self._frame_metadata[z], 'data': sparse_d, 'summary_stats': summary_stats}
            else:
                frame_dict = {'metadata': self._frame_metadata[z], 'data': sparse_d}

            self._current_frame_index = z + 1
            return {z: frame_dict}

    def get_next_frame(self):

        if self._current_frame_index == 0:
            self._fp.seek(self._frame_data_start_position, 0)

        # safety check
        if self._file_size - self._fp.tell() == 0:
            return None

        if not self._is_intermediate:
            if self._current_frame_index >= self._header['nz']:
                raise ValueError('Requested frame index is greater than number of frames in dataset')

        # if is an intermediate file, load metadata. For ReCoDe file metadata has been loaded in _load_seek_table
        if self._is_intermediate:

            # description of standard metadata (list of dicts)
            sm = self._structures.standard_frame_metadata_structure_for(self._header['reduction_level'],
                                                                        self._header['rc_operation_mode'])
            # dictionary of non-standard metadata
            nsm = self._rc_header.non_standard_metadata_sizes.values()

            d = {}
            frame_id = np.frombuffer(self._fp.read(4), dtype=np.uint32)[0]
            for field in sm:
                d[field['name']] = np.frombuffer(self._fp.read(field['bytes']), dtype=field['dtype'])[0]
            for field in nsm:
                d_type = get_dtype_string(nsm[field])
                d[field['name']] = np.frombuffer(self._fp.read(d_type.itemsize), dtype=d_type)[0]
        else:
            frame_id = self._current_frame_index
            d = self._frame_metadata[frame_id]

        # load and decompress (if necessary)
        if self._header['reduction_level'] == 2:
            sparse_d, summary_stats = self._get_frame_sparse(d)
        else:
            sparse_d = self._get_frame_sparse(d)

        # pack and return
        if sparse_d is None:
            self._header['nz'] = self._current_frame_index
            return None
        else:
            if self._header['reduction_level'] == 2:
                frame_dict = {'metadata': d, 'data': sparse_d, 'summary_stats': summary_stats}
            else:
                frame_dict = {'metadata': d, 'data': sparse_d}

            self._current_frame_index += 1
            return {frame_id: frame_dict}

    def get_next_frame_raw(self, read_data=True):

        if self._current_frame_index == 0:
            self._fp.seek(self._frame_data_start_position, 0)

        if not self._is_intermediate:
            if self._current_frame_index >= self._header['nz']:
                raise ValueError('Requested frame index is greater than number of frames in dataset')

        # if is an intermediate file, load metadata. For ReCoDe file metadata has been loaded in _load_seek_table
        if self._is_intermediate:

            # description of standard metadata (list of dicts)
            sm = self._structures.standard_frame_metadata_structure_for(self._header['reduction_level'],
                                                                        self._header['rc_operation_mode'])
            # dictionary of non-standard metadata
            nsm = self._rc_header.non_standard_metadata_sizes.values()

            d = {}
            if self._file_size - self._fp.tell() == 0:
                return None

            frame_id = np.frombuffer(self._fp.read(4), dtype=np.uint32)[0]
            for field in sm:
                d[field['name']] = np.frombuffer(self._fp.read(field['bytes']), dtype=field['dtype'])[0]
            for field in nsm:
                d_type = get_dtype_string(nsm[field])
                d[field['name']] = np.frombuffer(self._fp.read(d_type.itemsize), dtype=d_type)[0]

        else:
            frame_id = self._current_frame_index
            d = self._frame_metadata[frame_id]

        # load raw data
        raw_d = self._get_frame_raw(d, read_data=read_data)

        # pack and return
        if raw_d is None:
            self._header['nz'] = self._current_frame_index
            return None
        else:
            frame_dict = {'metadata': d, 'data': raw_d}
            self._current_frame_index += 1
            return {frame_id: frame_dict}

    def close(self):
        self._fp.close()

    def seek_to_frame_data(self):
        # get the starting position of frame data (start of frame 0 metadata for intermediate,
        # start of frame 0 data for recode file)
        self._frame_data_start_position = self._rc_header.get_frame_data_offset(self._is_intermediate,
                                                                                self._sz_frame_metadata)
        self._fp.seek(self._frame_data_start_position, 0)

    def _get_frame_raw(self, frame_metadata, read_data=True):

        sz_binary_map = 0
        if self._header['rc_operation_mode'] == 0:
            sz_binary_map = self._structures.binary_image_sz_bytes
        elif self._header['rc_operation_mode'] == 1:
            sz_binary_map = frame_metadata['bytes_in_compressed_binary_map']

        if read_data:
            binary_map = self._fp.read(sz_binary_map)
        else:
            binary_map = None
            self._fp.seek(sz_binary_map, 1)

        if self._header['reduction_level'] in [1, 2]:
            sz_pixvals = 0
            if self._header['reduction_level'] == 1 and self._header['rc_operation_mode'] == 0:
                sz_pixvals = frame_metadata['bytes_in_packed_pixvals']
            elif self._header['reduction_level'] == 1 and self._header['rc_operation_mode'] == 1:
                sz_pixvals = frame_metadata['bytes_in_compressed_pixvals']
            elif self._header['reduction_level'] == 2 and self._header['rc_operation_mode'] == 0:
                sz_pixvals = frame_metadata['bytes_in_packed_summary_stats']
            elif self._header['reduction_level'] == 2 and self._header['rc_operation_mode'] == 1:
                sz_pixvals = frame_metadata['bytes_in_compressed_summary_stats']

            if read_data:
                pixvals = self._fp.read(sz_pixvals)
            else:
                pixvals = None
                self._fp.seek(sz_pixvals, 1)

            return {'binary_map': binary_map, 'pixvals': pixvals}

        else:
            return {'binary_map': binary_map}

    def _get_frame_sparse(self, frame_metadata):

        if self._header['reduction_level'] == 1 and self._header['rc_operation_mode'] == 0:
            binary_map = self._fp.read(self._structures.binary_image_sz_bytes)
            pixvals = self._fp.read(frame_metadata['bytes_in_packed_pixvals'])

            n = self._c_reader.get_frame_sparse(self._header['reduction_level'], binary_map, pixvals, self._frame_buffer)

            if n != 0:
                sparse_d = self._make_coo_frame(n)
                return sparse_d
            else:
                return None

        if self._header['reduction_level'] == 1 and self._header['rc_operation_mode'] == 1:
            compressed_binary_map = self._fp.read(frame_metadata['bytes_in_compressed_binary_map'])
            compressed_pixvals = self._fp.read(frame_metadata['bytes_in_compressed_pixvals'])

            decompressed_binary_map = compressors.de_compress(self._header['compression_scheme'], compressed_binary_map,
                                                              self._decompressor_context)
            decompressed_pixvals = compressors.de_compress(self._header['compression_scheme'], compressed_pixvals,
                                                           self._decompressor_context)
            n = self._c_reader.get_frame_sparse(self._header['reduction_level'],
                                                decompressed_binary_map, decompressed_pixvals, self._frame_buffer)
            '''
            if n != 0:
                sparse_d = self._make_coo_frame(n)
                return sparse_d
            else:
                return None
            '''
            sparse_d = self._make_coo_frame(n)
            return sparse_d

        if self._header['reduction_level'] == 2 and self._header['rc_operation_mode'] == 0:
            binary_map = self._fp.read(self._structures.binary_image_sz_bytes)
            summary_stats = self._fp.read(frame_metadata['bytes_in_packed_summary_stats'])
            n = self._c_reader.get_frame_sparse(self._header['reduction_level'], binary_map, None, self._frame_buffer)

            if n != 0:
                sparse_d = self._make_coo_frame(n)
                unpacked_summary_stats = self._make_summary_stats_array(frame_metadata, summary_stats)
                return sparse_d, unpacked_summary_stats
            else:
                return None, None

        if self._header['reduction_level'] == 2 and self._header['rc_operation_mode'] == 1:
            compressed_binary_map = self._fp.read(frame_metadata['bytes_in_compressed_binary_map'])
            compressed_summary_stats = self._fp.read(frame_metadata['bytes_in_compressed_summary_stats'])
            decompressed_binary_map = compressors.de_compress(self._header['compression_scheme'], compressed_binary_map,
                                                              self._decompressor_context)
            decompressed_summary_stats = compressors.de_compress(self._header['compression_scheme'],
                                                                 compressed_summary_stats, self._decompressor_context)
            n = self._c_reader.get_frame_sparse(self._header['reduction_level'],
                                                decompressed_binary_map, None, self._frame_buffer)

            if n != 0:
                sparse_d = self._make_coo_frame(n)
                unpacked_summary_stats = self._make_summary_stats_array(frame_metadata, decompressed_summary_stats)
                return sparse_d, unpacked_summary_stats
            else:
                return None, None

        if self._header['reduction_level'] in [3, 4] and self._header['rc_operation_mode'] == 0:
            binary_map = self._fp.read(self._structures.binary_image_sz_bytes)
            n = self._c_reader.get_frame_sparse(binary_map, None, self._frame_buffer)

            if n != 0:
                sparse_d = self._make_coo_frame(n)
                return sparse_d
            else:
                return None

        if self._header['reduction_level'] in [3, 4] and self._header['rc_operation_mode'] == 1:
            compressed_binary_map = self._fp.read(frame_metadata['bytes_in_compressed_binary_map'])
            decompressed_binary_map = compressors.de_compress(self._header['compression_scheme'], compressed_binary_map,
                                                              self._decompressor_context)
            n = self._c_reader.get_frame_sparse(decompressed_binary_map, None, self._frame_buffer)

            if n != 0:
                sparse_d = self._make_coo_frame(n)
                return sparse_d
            else:
                return None

    def _make_coo_frame(self, n):
        a = np.frombuffer(self._frame_buffer, dtype=np.uint64, count=n * 3)
        d = np.array(a, dtype=np.uint64)
        d = np.transpose(np.reshape(d[:n * 3], [n, 3]))
        # cast d[2], d[0] and d[1] ti the appropriate dtypes, if required
        sparse_d = coo_matrix((d[2], (d[0], d[1])), shape=(self._header['ny'], self._header['nx']),
                              dtype=self._numpy_dtype)
        return sparse_d

    def _make_summary_stats_array(self, frame_metadata, packed_summary_stats):

        num_puddles = math.floor(frame_metadata['bytes_in_packed_summary_stats'] / self._header['target_bit_depth'])
        self._c_reader.bit_unpack_pixel_intensities(np.uint64(num_puddles), packed_summary_stats,
                                                    self._summary_stat_buffer)

        a = np.frombuffer(self._summary_stat_buffer, dtype=self._header['target_dtype'], count=num_puddles)
        d = np.array(a, dtype=self._header['target_dtype'])
        return d

    def copy_headers_to(self, target_fp, source_header_length):
        self._fp.seek(0, 0)
        rc_header = self._fp.read(self._rc_header.recode_header_length)
        src_header = self._fp.read(source_header_length)
        target_fp.write(rc_header)
        target_fp.write(src_header)

    @property
    def sz_frame_metadata(self):
        return self._sz_frame_metadata


def merge_parts(folder_path, base_filename, num_parts):

    # get number of frames
    intermediate_files = []
    num_frames = np.zeros(num_parts, dtype=np.uint64)
    for index in range(num_parts):
        inter_file_name = os.path.join(folder_path, base_filename + '_part' + '{0:03d}'.format(index))
        intermediate_files.append(inter_file_name)
        recode_reader = ReCoDeReader(inter_file_name, is_intermediate=True)
        recode_reader.open(print_header=False)
        nz = 0
        f = recode_reader.get_next_frame_raw(read_data=False)
        while f:
            nz += 1
            f = recode_reader.get_next_frame_raw(read_data=False)
        num_frames[index] = nz
        recode_reader.close()

    total_frames = np.sum(num_frames)

    # prepare target (merged) ReCoDe file
    # serialize ReCoDe header
    # Note: copy_header_to() moves file pointer of source to end of ReCoDe header, therefore close() and reopen()
    _target_file = open(os.path.join(folder_path, base_filename), 'wb')
    recode_reader_0 = ReCoDeReader(intermediate_files[0], is_intermediate=True)
    recode_reader_0.open(print_header=False)
    header = recode_reader_0.get_header().as_dict()
    recode_reader_0.copy_headers_to(_target_file, header['source_header_length'])
    recode_reader_0.close()

    # initialize readers and populate queues
    readers = []
    queues = []
    orders = np.ones(num_parts, dtype=np.uint64)
    for part_index in range(num_parts):
        recode_reader = ReCoDeReader(intermediate_files[part_index], is_intermediate=True)
        recode_reader.open(print_header=False)
        readers.append(recode_reader)
        d = deque(maxlen=1)
        f = readers[part_index].get_next_frame_raw()
        if f is not None:
            d.append(f)
            queues.append(d)
            orders[part_index] = list(f.keys())[0]
        else:
            orders[part_index] = np.iinfo(np.uint64).max

    # skip space for metadata
    sz_metadata = int(readers[0].sz_frame_metadata * total_frames)
    _target_file.seek(sz_metadata, 1)

    # merge frame data
    metadata = []
    for frame_index in range(total_frames):

        # search frames in queues, to find the one with the lowest frame_id
        _select = np.argmin(orders)

        # pop that frame, serialize data to file, hold metadata in list
        f = queues[_select].popleft()
        frame_id = list(f.keys())[0]
        metadata.append(f[frame_id]['metadata'])
        for field in f[frame_id]['data']:
            _target_file.write(f[frame_id]['data'][field])

        # load next raw frame into the emptied queue
        f = readers[_select].get_next_frame_raw()
        if f is not None:
            queues[_select].append(f)
            orders[_select] = frame_id
        else:
            orders[_select] = np.iinfo(np.uint64).max

        # if all part files are done, then break loop
        all_are_done = True
        for part_index in range(num_parts):
            if not orders[part_index] == np.iinfo(np.uint64).max:
                all_are_done = False

        if all_are_done:
            break

    # to serialize metadata, seek to the start of metadata
    _target_file.seek(readers[0]._rc_header._rc_header_length + readers[0]._header['source_header_length'], 0)

    # serialize metadata. Use len(metadata) and not total_frames, as the actual number of frames may be different from
    # the information in the intermediate headers
    for frame_index in range(len(metadata)):
        for field in metadata[frame_index]:
            if not field == 'frame_id':
                _target_file.write(metadata[frame_index][field])

    # update the number of frames based on the actual number of frames found: len(metadata)
    rc_header = readers[0].get_header()
    nz_position = rc_header.get_field_position_in_bytes('nz')
    _target_file.seek(nz_position, 0)
    _target_file.write(len(metadata).to_bytes(rc_header.get_definition('nz')['bytes'], sys.byteorder))
    _target_file.close()

    for part_index in range(num_parts):
        readers[part_index].close()


if __name__ == "__main__":

    file_name = 'D:/cbis/GitHub/ReCoDe/scratch/400fps_dose_43.rc1'
    intermediate_file_name = 'D:/cbis/GitHub/ReCoDe/scratch/400fps_dose_43.rc1_part000'

    reader = ReCoDeReader(file_name, is_intermediate=False)
    reader.open()
    frame_data = reader.get_frame(5)
    for i in range(3):
        frame_data = reader.get_next_frame()
    print(frame_data)
    reader.close()
