import numpy as np


class rc_cfg():

    REQ_TYPE_QUERY = 0
    REQ_TYPE_COMMAND = 1

    FILE_TYPE_BINARY = 0
    FILE_TYPE_MRC = 1
    FILE_TYPE_SEQ = 2
    FILE_TYPE_OTHER = 255

    STATUS_CODE_BUSY = 0            # Processing a requets, can't listen but not dead
    STATUS_CODE_AVAILABLE = 1       # Listening
    STATUS_CODE_ERROR = -1          # Dead due to exception
    STATUS_CODE_NOT_READY = -2      # Hasn't stated yet
    STATUS_CODE_IS_CLOSED = -3      # Is closed
    '''
    Expected status flow: -2, 1, 0, 1, 0, 1, 0, -3
    '''
    STATUS_CODES = {}
    STATUS_CODES['STATUS_CODE_BUSY'] = STATUS_CODE_BUSY
    STATUS_CODES['STATUS_CODE_AVAILABLE'] = STATUS_CODE_AVAILABLE
    STATUS_CODES['STATUS_CODE_ERROR'] = STATUS_CODE_ERROR
    STATUS_CODES['STATUS_CODE_NOT_READY'] = STATUS_CODE_NOT_READY
    STATUS_CODES['STATUS_CODE_IS_CLOSED'] = STATUS_CODE_IS_CLOSED

    MESSAGE_TYPE_INFO = 0
    MESSAGE_TYPE_ERROR = -1
    MESSAGE_TYPE_STATUS = 1
    MESSAGE_TYPE_ACK = 2

    MESSAGE_TYPES = {}
    MESSAGE_TYPES['MESSAGE_TYPE_INFO'] = MESSAGE_TYPE_INFO
    MESSAGE_TYPES['MESSAGE_TYPE_ERROR'] = MESSAGE_TYPE_ERROR
    MESSAGE_TYPES['MESSAGE_TYPE_STATUS'] = MESSAGE_TYPE_STATUS
    MESSAGE_TYPES['MESSAGE_TYPE_ACK'] = MESSAGE_TYPE_ACK


def map_dtype(type, bit_depth):

    if type == 0:
        # unsigned ints
        if bit_depth <= 8:
            return np.uint8
        elif bit_depth <= 16:
            return np.uint16
        elif bit_depth <= 32:
            return np.uint32
        elif bit_depth <= 64:
            return np.uint64
    elif type == 1:
        # signed ints
        if bit_depth <= 8:
            return np.int8
        elif bit_depth <= 16:
            return np.int16
        elif bit_depth <= 32:
            return np.int32
        elif bit_depth <= 64:
            return np.int64
    elif type == 2:
        # floats
        if bit_depth <= 32:
            return np.float32
        elif bit_depth <= 64:
            return np.float64
        
    error_msg = 'Unable to match a numpy dtype for type = ' + str(type) + ' (0=unsigned int, 1=signed int, 2=float) with bit depth = ' + str(bit_depth)
    raise ValueError(error_msg)


def get_dtype_code(dtype):
    dtype_code_map = {np.uint8: 0, np.uint16: 1, np.uint32: 2, np.uint64: 3,
                      np.int8: 4, np.int16: 5, np.int32: 6, np.int64: 7,
                      np.float32: 8, np.float64: 9}

    if dtype in dtype_code_map:
        return dtype_code_map[dtype]
    else:
        raise ValueError('Unknown dtype')


def get_dtype_string(dtype):
    if not isinstance(dtype, int):
        dtype = int(dtype)
    dtype_code_map = {0: 'uint8', 1: 'uint16', 2: 'uint32', 3: 'uint64',
                      4: 'int8', 5: 'int16', 6: 'int32', 7: 'int64',
                      8: 'float32', 9: 'float64'}

    if dtype in dtype_code_map:
        return dtype_code_map[dtype]
    else:
        raise ValueError('Unknown dtype')
