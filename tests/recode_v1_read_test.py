import numpy as np
import matplotlib.pyplot as plt
from pyrecode.recode_reader import ReCoDeReader, merge_parts

if __name__ == "__main__":

    intermediate_file_name = '/scratch/loh/abhik/24-Oct-2019/gold_nano_dose_0.022/gold_nano_on_cgrid.rc1_part000'

    reader = ReCoDeReader(intermediate_file_name, is_intermediate=True)
    reader.open()

    summed_frame = np.zeros((4096, 4096))
    frame_data = reader.get_next_frame()
    while frame_data:
        frame_id = list(frame_data.keys())[0]
        a = frame_data[frame_id]['data'].toarray()
        summed_frame = np.add(summed_frame, a)
        print(frame_id, frame_data[frame_id]['metadata'])
        frame_data = reader.get_next_frame()
    reader.close()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(summed_frame)
    plt.savefig('sum.png')

    merge_parts('/scratch/loh/abhik/24-Oct-2019/gold_nano_dose_0.022/', 'gold_nano_on_cgrid.rc1', 12)

    recode_file_name = '/scratch/loh/abhik/24-Oct-2019/gold_nano_dose_0.022/gold_nano_on_cgrid.rc1'
    reader = ReCoDeReader(recode_file_name, is_intermediate=False)
    reader.open()

    """
    frame_data = reader.get_next_frame()
    while frame_data:
        frame_id = list(frame_data.keys())[0]
        a = frame_data[frame_id]['data'].toarray()
        print(frame_id, frame_data[frame_id]['metadata'])
        frame_data = reader.get_next_frame()
    """

    for frame_id in [0,1,2]:
        frame_data = reader.get_frame(frame_id)
        print(frame_id, frame_data[frame_id]['metadata'])

    reader.close()
