import os
import numpy as np
from pyrecode.recode_reader import ReCoDeReader


class ReCoDeViewer:

    def __init__(self, folder_path, base_filename, num_parts, fractionation):
        
        self._num_parts = num_parts
        self._fractionation = fractionation

        self._readers = {}
        for index in range(num_parts):
            intermediate_file_name = os.path.join(folder_path, base_filename + '_part' + '{0:03d}'.format(index))
            self._readers[index] = ReCoDeReader(intermediate_file_name, is_intermediate=True)
            self._readers[index].open()
        self._ny = self._readers[0]._get_shape()[1]
        self._nx = self._readers[0]._get_shape()[2]

        self._view = None
        self._frame_start = 0
        self._buffer = {}
        for index in range(fractionation):
            self._buffer[index] = queue.Queue(maxsize=fractionation)

    def _get_next_frame_safely (self, reader_index):
        '''
        performs a get only if there is enough data ahead, ensuring that EoF is not reached after get
        returns None if enough data is not available. In this case, pointer is not moved, and another attempt to read the frame may be made in the next pass
        '''
        
        # check if there is enough data ahead, based on current estimate of file size and current pointer position
        # if pointer is too close to EoF, update current estimate of file size and try again
        # if pointer is still too close to EoF, return None
        
        d = self._readers[reader_index]._get_next_frame()
        return d

    def get_next_view (self):

        # attempt to fill queues
        for index in range(self._num_parts):
            for t in range(self._fractionation - self._buffer[index].qsize()):
                d = self._get_next_frame_safely(index)
                if d is not None:
                    self._buffer[index].put(d)
                else:
                    break

        # temporary holder of fractionation frames
        temp = {}

        # try to fill temp with 'fractionation' frames
        for _fid in range(self._frame_start, self._frame_start+self._fractionation):
            for index in range(self._num_parts):
                for i in range(self._buffer[index].qsize()):
                    if _fid in self._buffer[index].queue[0]:            # Assumes frames are in ascending order within a part file
                        temp.update(self._buffer[index].get())          # Assumes there are no duplicate frames

        # warn if enough frames are not available
        if len(temp) < self._fractionation:
            print("Warning: read fewer frames (" + str(len(temp)) + ") than requested (" + str(self._fractionation) + ").")

        self._view = np.zeros((self._nx,self._ny))
        for frame_id in temp:
            if temp[frame_id] is not None:
                self._view = np.add(self._view, temp[frame_id].toarray())

        ret_val = {'start': self._frame_start, 'n_Frames': len(temp), 'view': self._view}
        
        # next frame_start
        self._frame_start = np.max(list(temp.keys())) + 1

        return ret_val
    
    def close(self):
        for index in range(self._num_parts):
            self._readers[index].close()
