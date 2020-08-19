import sys
import numpy as np
from .misc import get_dtype_code, get_dtype_string


class ReCoDeHeader:

    """
    ToDo:
    """

    def __init__(self, version=0.2):

        self._version = version

        self._rc_header = {}
        self._rc_header_field_defs = []
        self._get_rc_field_defs()
        self._source_header = None
        self._non_standard_frame_metadata_sizes = {}

    def _get_rc_field_defs(self):

        self._rc_header_field_defs = []

        if self._version < 0.2:
            self._rc_header_field_defs.append({'name': 'uid', 'bytes': 8, 'dtype': np.uint64})
            self._rc_header_field_defs.append({'name': 'version_major', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'version_minor', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'reduction_level', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'rc_operation_mode', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'target_bit_depth', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'nx', 'bytes': 2, 'dtype': np.uint16})
            self._rc_header_field_defs.append({'name': 'ny', 'bytes': 2, 'dtype': np.uint16})
            self._rc_header_field_defs.append({'name': 'nz', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'L2_statistics', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'L4_centroiding', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'compression_scheme', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'compression_level', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_file_type', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_header_length', 'bytes': 2, 'dtype': np.uint16})
            self._rc_header_field_defs.append({'name': 'source_header_position', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_file_name', 'bytes': 100, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'calibration_file_name', 'bytes': 100, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'calibration_threshold_epsilon', 'bytes': 2, 'dtype': np.uint16})
            self._rc_header_field_defs.append({'name': 'has_calibration_data', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'frame_offset', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'calibration_frame_offset', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'num_calibration_frames', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'source_bit_depth', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_dtype', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'target_dtype', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'checksum', 'bytes': 32, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'futures', 'bytes': 42, 'dtype': np.uint8})

            self._rc_header_length = 321
        else:
            self._rc_header_field_defs.append({'name': 'uid', 'bytes': 8, 'dtype': np.uint64})
            self._rc_header_field_defs.append({'name': 'version_major', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'version_minor', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'is_intermediate', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'reduction_level', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'rc_operation_mode', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'is_bit_packed', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'target_bit_depth', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'nx', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'ny', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'nz', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'frame_metadata_size', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'num_non_standard_frame_metadata', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'L2_statistics', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'L4_centroiding', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'compression_scheme', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'compression_level', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_file_type', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_header_length', 'bytes': 2, 'dtype': np.uint16})
            self._rc_header_field_defs.append({'name': 'source_header_position', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_file_name', 'bytes': 100, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'calibration_file_name', 'bytes': 100, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'calibration_threshold_epsilon', 'bytes': 8, 'dtype': np.uint64})
            self._rc_header_field_defs.append({'name': 'has_calibration_data', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'frame_offset', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'calibration_frame_offset', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'num_calibration_frames', 'bytes': 4, 'dtype': np.uint32})
            self._rc_header_field_defs.append({'name': 'source_bit_depth', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'source_dtype', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'target_dtype', 'bytes': 1, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'checksum', 'bytes': 32, 'dtype': np.uint8})
            self._rc_header_field_defs.append({'name': 'futures', 'bytes': 219, 'dtype': np.uint8})

            # self._rc_header_length = 512
            self._rc_header_length = 0
            for field in self._rc_header_field_defs:
                self._rc_header_length += field['bytes']

    def create(self, init_params, input_params, is_intermediate):

        if self._version < 0.2:
            self._rc_header['uid'] = 158966344846346
            self._rc_header['version_major'] = 0
            self._rc_header['version_minor'] = 1
            self._rc_header['reduction_level'] = input_params.reduction_level
            self._rc_header['rc_operation_mode'] = input_params.rc_operation_mode
            self._rc_header['target_bit_depth'] = input_params.target_bit_depth
            self._rc_header['nx'] = input_params.nx
            self._rc_header['ny'] = input_params.ny
            self._rc_header['nz'] = input_params.nz
            self._rc_header['L2_statistics'] = input_params.L2_statistics
            self._rc_header['L4_centroiding'] = input_params.L4_centroiding
            self._rc_header['compression_scheme'] = input_params.compression_scheme
            self._rc_header['compression_level'] = input_params.compression_level
            self._rc_header['source_file_type'] = input_params.source_file_type
            self._rc_header['source_header_length'] = input_params.source_header_length
            self._rc_header['source_header_position'] = 0
            self._rc_header['source_file_name'] = init_params.image_filename
            self._rc_header['calibration_file_name'] = init_params.calibration_filename
            self._rc_header['calibration_threshold_epsilon'] = input_params.calibration_threshold_epsilon
            self._rc_header['has_calibration_data'] = input_params.keep_calibration_data
            self._rc_header['frame_offset'] = input_params.frame_offset
            self._rc_header['calibration_frame_offset'] = input_params.calibration_frame_offset
            self._rc_header['num_calibration_frames'] = input_params.num_calibration_frames
            self._rc_header['source_bit_depth'] = input_params.source_bit_depth
            self._rc_header['source_dtype'] = 0     # version 0.1 only supports unsigned ints
            self._rc_header['target_dtype'] = 0     # version 0.1 only supports unsigned ints
            self._rc_header['checksum'] = np.zeros(32, dtype=np.uint8)
            self._rc_header['futures'] = np.zeros(42, dtype=np.uint8)
        else:
            self._rc_header['uid'] = 158966344846346
            self._rc_header['version_major'] = 0
            self._rc_header['version_minor'] = 2
            # is_intermediate
            self._rc_header['is_intermediate'] = is_intermediate
            self._rc_header['reduction_level'] = input_params.reduction_level
            self._rc_header['rc_operation_mode'] = input_params.rc_operation_mode
            # is_bit_packed
            self._rc_header['is_bit_packed'] = 1
            self._rc_header['target_bit_depth'] = input_params.target_bit_depth
            self._rc_header['nx'] = input_params.nx
            self._rc_header['ny'] = input_params.ny
            self._rc_header['nz'] = input_params.nz
            # frame_metadata_size
            self._rc_header['frame_metadata_size'] = 0
            # num_non_standard_frame_metadata
            self._rc_header['num_non_standard_frame_metadata'] = 0
            self._rc_header['L2_statistics'] = input_params.L2_statistics
            self._rc_header['L4_centroiding'] = input_params.L4_centroiding
            self._rc_header['compression_scheme'] = input_params.compression_scheme
            self._rc_header['compression_level'] = input_params.compression_level
            self._rc_header['source_file_type'] = input_params.source_file_type
            self._rc_header['source_header_length'] = input_params.source_header_length
            self._rc_header['source_header_position'] = 0
            self._rc_header['source_file_name'] = init_params.image_filename
            self._rc_header['calibration_file_name'] = init_params.calibration_filename
            self._rc_header['calibration_threshold_epsilon'] = input_params.calibration_threshold_epsilon
            self._rc_header['has_calibration_data'] = input_params.keep_calibration_data
            self._rc_header['frame_offset'] = input_params.frame_offset
            self._rc_header['calibration_frame_offset'] = input_params.calibration_frame_offset
            self._rc_header['num_calibration_frames'] = input_params.num_calibration_frames
            self._rc_header['source_bit_depth'] = input_params.source_bit_depth
            self._rc_header['source_dtype'] = input_params.source_data_type
            self._rc_header['target_dtype'] = input_params.target_data_type
            self._rc_header['checksum'] = np.zeros(32, dtype=np.uint8)
            self._rc_header['futures'] = np.zeros(219, dtype=np.uint8)

    @property
    def recode_header_length(self):
        return self._rc_header_length

    def as_dict(self):
        return self._rc_header

    def get(self, field_name):
        if field_name not in self._rc_header:
            raise ValueError('The requested field does not exist in recode header')
        return self._rc_header[field_name]

    def get_definition(self, name):
        for field_def in self._rc_header_field_defs:
            if field_def['name'] == name:
                return field_def
        raise ValueError('The requested field does not exist in recode header')

    def set(self, field_name, value):
        if field_name not in self._rc_header:
            raise ValueError('The requested field does not exist in recode header')
        self._rc_header[field_name] = value

    def load(self, rc_filename, is_intermediate=False):

        if rc_filename == '':
            raise ValueError('ReCoDe filename missing')

        with open(rc_filename, 'rb') as fp:

            # first load uid and version to decide which format to load
            for i in range(3):
                field = self._rc_header_field_defs[i]
                b = fp.read(field['bytes'])
                value = np.frombuffer(b, dtype=field['dtype'])
                self._rc_header[field['name']] = value[0]

            # get writer version
            self._version = int(self._rc_header['version_major']) \
                          + int(self._rc_header['version_minor']) / 10.0

            # choose the appropriate header structure for the version
            self._get_rc_field_defs()

            # go back to the start and load the full header
            fp.seek(0, 0)

            for field in self._rc_header_field_defs:
                b = fp.read(field['bytes'])
                value = np.frombuffer(b, dtype=field['dtype'])
                if field['name'] is 'calibration_file_name':
                    formatted_value = self._to_string(value)
                elif field['name'] is 'source_file_name':
                    formatted_value = self._to_string(value)
                else:
                    if len(value) == 1:
                        formatted_value = value[0]
                    else:
                        formatted_value = value
                self._rc_header[field['name']] = formatted_value

            # for version 0.1, set the missing fields
            if self._version < 0.2:
                if is_intermediate:
                    self._rc_header['is_intermediate'] = 0
                else:
                    self._rc_header['is_intermediate'] = 1
                self._rc_header['is_bit_packed'] = 1
                self._rc_header['frame_metadata_size'] = 0
                self._rc_header['num_non_standard_frame_metadata'] = 0
                self._rc_header['source_header_length'] = 0
                # version 0.1 only supports unsigned ints, override anything header says
                self._rc_header['source_dtype'] = 0
                self._rc_header['target_dtype'] = 0

            # load non-standard metadata sizes
            for i in range(self._rc_header['num_non_standard_frame_metadata']):
                b = fp.read(100)
                value = np.frombuffer(b, dtype=np.uint8)
                name = self._to_string(value[:-1])
                self._non_standard_frame_metadata_sizes[name] = value[99]

            # load source header
            b = fp.read(self._rc_header['source_header_length'])
            self._source_header = b

    def serialize(self, rc_filename):
        if rc_filename == '':
            raise ValueError('ReCoDe filename missing')
        with open(rc_filename, 'wb') as fp:
            self.serialize_to(fp)

    def serialize_to(self, fp):
        for field in self._rc_header_field_defs:
            _d_type = field['dtype']
            _n_bytes = field['bytes']
            _name = field['name']
            _value = self._rc_header[field['name']]
            if _d_type == np.uint8 and _n_bytes != 1:
                if _name in ['calibration_file_name', 'source_file_name']:
                    n = str(_value)
                    if len(n) > _n_bytes:
                        n = n[:_n_bytes]
                    elif len(n) < _n_bytes:
                        n = n.ljust(_n_bytes, ' ')
                    t = n.encode('utf-8')
                else:
                    t = (_value[:_n_bytes]).tobytes()
            else:
                t = _value.to_bytes(_n_bytes, sys.byteorder)
            fp.write(t)

    def skip_header(self, rc_fp):
        rc_fp.seek(self._rc_header_length)
        return rc_fp

    def get_frame_data_offset(self):
        if self._rc_header['version_major'] == 0 and self._rc_header['version_minor'] == 1:
            return self._rc_header_length
        else:
            return self._rc_header_length + self._rc_header['source_header_length'] \
                   + len(self._non_standard_frame_metadata_sizes)*100

    @property
    def source_header(self):
        return self._source_header

    @property
    def non_standard_metadata_sizes(self):
        return self._non_standard_frame_metadata_sizes

    def update(self, name, value):
        self._rc_header[name] = value

    def get_field_position_in_bytes(self, name):
        position = 0
        for field_def in self._rc_header_field_defs:
            if field_def['name'] == name:
                return position
            else:
                position += field_def['bytes']
        print(name)
        raise ValueError("The requested field is not defined in the header")

    def print(self):
        print("ReCoDe Header")
        print("-------------")
        for field in self._rc_header_field_defs:
            print(field['name'], '=', self._rc_header[field['name']])
            '''
            if field['name'] is 'calibration_file_name':
                print(field['name'], '=', self._to_string(self._rc_header[field['name']]))
            elif field['name'] is 'source_file_name':
                print(field['name'], '=', self._to_string(self._rc_header[field['name']]))
            else:
                if len(self._rc_header[field['name']]) == 1:
                    print(field['name'], '=', self._rc_header[field['name']][0])
                else:
                    print(field['name'], '=', self._rc_header[field['name']])
            '''
            
    def validate(self):
        # check that no fields are missing in the header, does not check the validity of their values
        for field in self._rc_header_field_defs:
            if field['name'] not in self._rc_header:
                print('ReCoDe Header Validation Failed: ' + field['name'] + ' is missing.')
                return False
        return True

    def _load_metadata(self, rc_filename):
        assert rc_filename != '', 'ReCoDe filename missing'
        # nCompressedSize_BinaryImage, nCompressedSize_Pixvals, bytesRequiredForPacking
        with open(rc_filename, 'wb') as fp:
            buffer = fp.read(self._rc_header['nz']*3*4)
            values = np.frombuffer(buffer, dtype=np.uint32)
            self._metadata = np.reshape(values, [self._rc_header['nz'], 3])

    def _to_string(self, arr):
        return ''.join([chr(x) for x in arr])


if __name__ == "__main__":

    rc_header = ReCoDeHeader()
    rc_header.load('D:/cbis/GitHub/ReCoDe/scratch/400fps_dose_43.rc1')
    rc_header.print()