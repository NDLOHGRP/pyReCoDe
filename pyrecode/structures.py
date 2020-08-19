import math
import numpy as np


class ReCoDeStructures:

    """
    ToDo:
    """

    def __init__(self, recode_header):

        self._recode_header = recode_header
        self._standard_frame_metadata_structure = {}
        self._binary_image_sz_bytes = int(math.ceil(
            (float(self._recode_header['nx']) * float(self._recode_header['ny']))/8.))

        for reduction_level in range(5):
            for rc_operation_mode in range(2):

                if reduction_level == 1 and rc_operation_mode == 0:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] \
                        = [{'name': 'bytes_in_packed_pixvals', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True}]

                if reduction_level == 1 and rc_operation_mode == 1:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] \
                        = [{'name': 'bytes_in_compressed_binary_map', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True},
                           {'name': 'bytes_in_compressed_pixvals', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True},
                           {'name': 'bytes_in_packed_pixvals', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': False}]

                if reduction_level == 2 and rc_operation_mode == 0:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] \
                        = [{'name': 'bytes_in_packed_summary_stats', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True}]

                if reduction_level == 2 and rc_operation_mode == 1:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] \
                        = [{'name': 'bytes_in_compressed_binary_map', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True},
                           {'name': 'bytes_in_compressed_summary_stats', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True},
                           {'name': 'bytes_in_packed_summary_stats', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': False}]

                if reduction_level in [3, 4] and rc_operation_mode == 0:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] = []

                if reduction_level in [3, 4] and rc_operation_mode == 1:
                    self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)] \
                        = [{'name': 'bytes_in_compressed_binary_map', 'bytes': 4, 'dtype': np.uint32, 'is_frame_size': True}]

    def get_standard_frame_metadata_size(self, reduction_level, rc_operation_mode):
        """
        computes the total size of standard frame metadata elements per frame (non-standard metadata elements are not included)
        :param reduction_level: int reduction level (1, 2, 3, or 4)
        :param rc_operation_mode: int reduction only (0) or reduction and compression (1)
        :return: int size of standard frame metadata elements per frame in bytes
        """
        sz_bytes = 0
        for field in self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)]:
            sz_bytes += np.dtype(field['dtype']).itemsize
        return sz_bytes

    def get_frame_data_size(self, reduction_level, rc_operation_mode, metadata):
        """
        computes the size of a frame data (excluding metadata) based on its metadata
        :param reduction_level: int reduction level (1, 2, 3, or 4)
        :param rc_operation_mode: int reduction only (0) or reduction and compression (1)
        :param metadata: dictionary metadata values for this frame
        :return: int size of frame data in bytes
        """

        if reduction_level == 1 and rc_operation_mode == 0:
            return self._binary_image_sz_bytes + metadata['bytes_in_packed_pixvals']

        if reduction_level == 1 and rc_operation_mode == 1:
            return metadata['bytes_in_compressed_binary_map'] + metadata['bytes_in_compressed_pixvals']

        if reduction_level == 2 and rc_operation_mode == 0:
            return self._binary_image_sz_bytes + metadata['bytes_in_packed_summary_stats']

        if reduction_level == 2 and rc_operation_mode == 1:
            return metadata['bytes_in_compressed_binary_map'] + metadata['bytes_in_compressed_summary_stats']

        if reduction_level == 3 and rc_operation_mode == 0:
            return self._binary_image_sz_bytes

        if reduction_level == 3 and rc_operation_mode == 1:
            return metadata['bytes_in_compressed_binary_map']

        if reduction_level == 4 and rc_operation_mode == 0:
            return self._binary_image_sz_bytes

        if reduction_level == 4 and rc_operation_mode == 1:
            return metadata['bytes_in_compressed_binary_map']

    @property
    def binary_image_sz_bytes(self):
        return self._binary_image_sz_bytes

    @property
    def standard_frame_metadata_structure(self):
        return self._standard_frame_metadata_structure

    def standard_frame_metadata_structure_for(self, reduction_level, rc_operation_mode):
        return self._standard_frame_metadata_structure[(reduction_level, rc_operation_mode)]
