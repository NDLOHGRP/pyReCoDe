import json
from abc import ABC, abstractmethod
import numpy as np
from .misc import rc_cfg as rc


# use st_blksize whenever we can
DEFAULT_BUFFER_SIZE = 8 * 1024  # bytes


def emfile (file, file_type=None, mode="r", buffering=-1):

    def _open (file, source_type=None, buffering=-1):
        
        try:
            if (source_type == rc.FILE_TYPE_MRC):
                reader = MRCReader(file)
            elif (source_type == rc.FILE_TYPE_SEQ):
                reader = SEQReader(file)
            elif (source_type == rc.FILE_TYPE_BINARY):
                raise NotImplementedError
            else:
                print("Source type: " + source_type + " is not supported.")
                raise ValueError
            return reader
            
        except:
            raise

    if (mode == "r"):
        return _open(file, file_type, buffering)
    else:
        print("emfile supports only 'r' mode.")
        raise NotImplementedError

class EMReaderBase():
    """
    The Base class for all emlab.io.*Reader classes
    Any implementing sub class must implement the seven abstract methods
    """
    
    def __init__(self, file, source_type='', fast_random_access=False, buffer_size=DEFAULT_BUFFER_SIZE):
        """file is a path-like object giving the pathname (absolute or relative to the current working directory) of the file to be opened.
        source_type is a string describing the type of file. It can be on of the following: "mrc", "mrcs", "seq", "recode", "rc", "dm3", and "dm4".
        fast_random_access is a boolean indicating if fast slicing can be performed. It is True for MRC/MRCS and ReCoDe files and False for SEQ, DM3 and DM4 files.
        buffer_size is an optional parameter for future use.
        """
        self._source_filename = file
        self._source_type = source_type
        self._open()
        self._header = self._load_header()
        self._shape = self._get_shape()
        self._dtype = self._get_dtype()
        self.buffer_size = buffer_size
        self._fast_random_access = fast_random_access
        self._current_z = 0
        super().__init__()
    
    @property
    def source_type(self):
        """Returns the source file type, e.g. mrcs, seq, etc.
        """
        return self._source_type
        
    @property
    def shape(self):
        """Returns the shape of data as per header. The actual data size in file is not checked.
        """
        return self._shape
        
    @property
    def header(self):
        """Returns a dictionary of header fields
        """
        return self._header
        
    @property
    def dtype(self):
        """Returns the data type
        """
        return self._dtype
        
    @property
    def fast_random_access(self):
        """Returns True if fast slicing can be performed, False otherwise
        True for MRC/MRCS and ReCoDe files
        False for SEQ, DM3 and DM4 files
        """
        return self._fast_random_access
        
    @abstractmethod
    def get_true_shape(self):
        """ Returns the actual shape of data in the file. 
        This function is not implemented for DM3 and DM4
        """
        return None
    
    @abstractmethod
    def _load_header(self):
        """Reads the header and returns a dictionary of header fields and values
        """
        return {}
        
    @abstractmethod
    def _get_shape(self):
        """Returns the shape of the data (as per header information) as a numpy array
        """
        return {}
        
    @abstractmethod
    def _get_sub_volume(self, slice_z, slice_y, slice_x):
        """Extracts subvolume from data and returns an EMData object. 
        Slices for the z,y and x dimensions are given as Slice objects slice_z, slice_y and slice_x respectively.
        """
        return None
        
    @abstractmethod
    def _get_frame(self, z_index):
        """Extracts frame no z_index from data and returns an EMData object. 
        """
        return None
    
    @abstractmethod
    def _open(self):
        """Opens file object and prepares for reading in header and data. 
        Any calls to get data (such as _load_header(), _get_sub_volume(), etc) after calling this method should be able to access the file and return requested data.
        Requests to read data before calling this function should raise an error.
        """
        return None
    
    @abstractmethod
    def close(self):
        """Closes any open file objects.
        Requests to read data after calling this function should raise an error.
        """
        return None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_z > self.shape[0]:
            raise StopIteration
        else:
            self._current_z += 1
            return self._get_frame(self._current_z - 1)
    
    def __getitem__(self, key):
        
        if isinstance(key, tuple):
            if (len(key) == 3):
                # return a subvolume
                return self._get_sub_volume(key[0], key[1], key[2])
            elif(len(key) == 2):
                # return a subvolume
                return self._get_sub_volume(key[0], key[1], slice(0,self._shape[2]))
            elif(len(key) == 1):
                # return a bunch of frames
                return self._get_sub_volume(key[0], slice(0,self._shape[1]), slice(0,self._shape[2]))
            
        elif isinstance(key, slice):
            # return a bunch of frames
            return self._get_sub_volume(key, slice(0,self._shape[1]), slice(0,self._shape[2]))
            
        elif isinstance(key, int):
            # return one frame
            return self._get_frame(key)
            
        else:
            raise TypeError
    
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close()
    
    def print_header(self):
        for field in self._header:
            print(field + ":\t" + str(self._header[field]))

    @abstractmethod
    def serialize_header(self, fp):
        return None
    
    
class MRCReader(EMReaderBase):
    def __init__(self, file):
        try:
            import mrcfile
            self._mrcfile = mrcfile
        except ImportError:
            print("Reading MRC files requires mrcfile to be installed")
            return
        EMReaderBase.__init__(self, file, 'mrc', False)

    def _open(self):
        try:
            self._file_handle = self._mrcfile.open(self._source_filename, mode="r")
        except ValueError:
            self._file_handle = self._mrcfile.open(self._source_filename, mode="r", permissive=True)
        self._stack = self._file_handle.data
        
    def _load_header(self):
        record = self._file_handle.header
        names = record.dtype.names
        return dict(zip(names, [record[item] for item in names]))

    def get_true_shape(self):
        return self._stack.shape

    def _get_shape(self):
        return tuple([self._header["nz"], self._header["ny"], self._header["nx"]])

    def _get_dtype(self):
        return (self._stack.dtype)

    def _get_sub_volume(self, slice_z, slice_y, slice_x):
        if self._file_handle.is_image_stack():
            container = self._stack[slice_z, slice_y, slice_x]
        elif self._file_handle.is_single_image():
            container = self._stack[np.newaxis,slice_y,slice_x]
        else:
            raise NotImplementedError
        return container

    def _get_frame(self, z_index):
        if self._file_handle.is_single_image():
            container = self._stack[np.newaxis,:,:]
        elif self._file_handle.is_image_stack():
            container = self._stack[z_index][np.newaxis,:,:]
        else:
            raise NotImplementedError
        return container

    def close(self):
        self._file_handle.close()

    def serialize_header(self, fp):
        fp.write(self._file_handle.header)


class SEQReader(EMReaderBase):
    
    def __init__(self, file, buffer_size=DEFAULT_BUFFER_SIZE):
        try:
            import pims
            self._pims = pims
        except ImportError:
            print("Reading Sequence files requires PIMS to be installed")
            return
        EMReaderBase.__init__(self, file, 'seq', False, buffer_size)
        
    def _open(self):
        self._stack = self._pims.open(self._source_filename)
        self._is_open = True
        
    def _get_shape(self):
        shape = tuple([self._stack.header_dict['allocated_frames'], self._stack.header_dict['height'], self._stack.header_dict['width']])
        return shape
    
    def _load_header(self):
        return self._stack.header_dict
        
    def get_true_shape(self):
        frame = self._stack[0]
        frame_dims = frame.shape
        shape = tuple([len(images), frame_dims[0], frame_dims[1]])
        return shape
        
    def _get_frame(self, z_index):
        container = np.zeros((1, self._shape[1], self._shape[2]), dtype = self._dtype)
        container[0,:,:] = self._stack[z_index]
        return container
        
    def _get_sub_volume(self, slice_z, slice_y, slice_x):
        z_indices = range(*slice_z.indices(self._shape[0]))
        nz = len(z_indices)
        ny = len(range(*slice_y.indices(self._shape[1])))
        nx = len(range(*slice_x.indices(self._shape[2])))
        container = np.zeros((nz, ny, nx), dtype = self._dtype)
        for index, z_index in enumerate(z_indices):
            container[index] = self._stack[z_index][slice_y.start:slice_y.stop:slice_y.step, slice_x.start:slice_x.stop:slice_x.step]
        return container
    
    def _get_dtype(self):
        if (self._header['bit_depth'] == 8):
            return np.uint8
        elif (self._header['bit_depth'] == 16):
            return np.int16
        else:
            print("Sequence datasets with bit-depth {0:d} is not supported.".format(self._header['bit_depth']))
            raise TypeError    
    
    def close(self):
        self._stack.close()
        self._is_open = False
        return

    def serialize_header(self, fp):
        # s = json.dumps(self._stack.header)
        # binary = ' '.join(format(ord(letter), 'b') for letter in s)
        binary = np.zeros(1024, dtype=np.uint8).tobytes()
        fp.write(binary[:1024])


if __name__ == "__main__":

    reader = MRCReader('', 'mrcs')