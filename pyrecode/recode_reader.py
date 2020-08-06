import math
import numpy as np
from scipy.sparse import coo_matrix

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

    def open(self):
        self._load_header()
        compressors.import_checks(self._header)
        self._fp = open(self._source_filename, "rb")
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

    def _load_header(self):
        self._rc_header = ReCoDeHeader()
        self._rc_header.load(self._source_filename)
        self._header = self._rc_header.as_dict()
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
        offset = self._rc_header.get_frame_data_offset()
        if self._is_intermediate:
            self._frame_data_start_position = offset
        else:
            self._frame_data_start_position = offset + self._header['nz'] * self._sz_frame_metadata

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

            self._frame_metadata = np.zeros((self._header['nz'], n_metadata_elements), dtype=np.uint32)

            # move to start of frame data
            self._fp.seek(self._rc_header.get_frame_data_offset(), 0)

            for _frame_index in range(self._header['nz']):
                d = {}
                for field in sm:
                    d[field] = np.frombuffer(self._fp.read(field['bytes']), dtype=field['dtypes'])[0]
                for field in nsm:
                    d_type = get_dtype_string(nsm[field])
                    d[field] = np.frombuffer(self._fp.read(d_type.itemsize), dtype=d_type)
                self._frame_metadata[_frame_index] = d

            # create a seek table, that maintains the offset of each frame relative to self._frame_data_start_position
            # only available for ReCoDe files (not intermediate files)
            self._seek_table = np.zeros((self._header['nz'], 2), dtype=np.uint32)
            for _frame_index in range(1, self._header['nz']):
                self._seek_table[_frame_index, 0] = self._structures.get_frame_data_size(
                    self._header['reduction_level'], self._header['rc_operation_mode'],
                    self._frame_metadata[_frame_index])
            self._seek_table[:, 1] = np.cumsum(self._seek_table[:, 0])

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
        self._fp.seek(s + self._seek_table[z, 1], int(0))

        # load and decompress (if necessary)
        if self._header['reduction_level'] == 2:
            sparse_d, summary_stats = self._get_frame_sparse(self._frame_metadata[z])
        else:
            sparse_d = self._get_frame_sparse(self._frame_metadata[z])

        # pack and return
        if sparse_d is None:
            print('Reached EoF after ' + str(z) + ' frames: Quitting')
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

        z = self._current_frame_index
        if z == 0:
            self._fp.seek(self._frame_data_start_position, 0)

        if z >= self._header['nz']:
            raise ValueError('Requested frame index is greater than number of frames in dataset')

        # if is an intermediate file, load metadata. For ReCoDe file metadata has been loaded in _load_seek_table
        if self._is_intermediate:

            # description of standard metadata (list of dicts)
            sm = self._structures.standard_frame_metadata_structure_for(self._header['reduction_level'],
                                                                        self._header['rc_operation_mode'])
            # dictionary of non-standard metadata
            nsm = self._rc_header.non_standard_metadata_sizes.values()

            d = {}
            z = np.frombuffer(self._fp.read(4), dtype=np.uint32)[0]
            for field in sm:
                d[field['name']] = np.frombuffer(self._fp.read(field['bytes']), dtype=field['dtype'])[0]
            for field in nsm:
                d_type = get_dtype_string(nsm[field])
                d[field['name']] = np.frombuffer(self._fp.read(d_type.itemsize), dtype=d_type)[0]
        else:
            d = self._frame_metadata[z]

        # load and decompress (if necessary)
        if self._header['reduction_level'] == 2:
            sparse_d, summary_stats = self._get_frame_sparse(d)
        else:
            sparse_d = self._get_frame_sparse(d)

        # pack and return
        if sparse_d is None:
            print('Reached EoF after ' + str(z) + ' frames: Quitting')
            self._header['nz'] = self._current_frame_index
            return None
        else:
            if self._header['reduction_level'] == 2:
                frame_dict = {'metadata': d, 'data': sparse_d, 'summary_stats': summary_stats}
            else:
                frame_dict = {'metadata': d, 'data': sparse_d}

            self._current_frame_index = z + 1
            return {z: frame_dict}

    def close(self):
        self._fp.close()

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
            if n != 0:
                sparse_d = self._make_coo_frame(n)
                return sparse_d
            else:
                return None

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
