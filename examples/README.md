### ReCoDe examples
Ipython notebooks showcasing examples of pyReCoDe and its related applications.

---

The `Fine_Calibration_with_Backscattering` notebook describes in detail the fine calibration process, including a simple model for estimating the number of backscattering events. 

__To view data:__

1. From ReCoDe part files: See `ReCoDe_Live_View`/`ReCoDe_Live_View_MT`/`ReCoDe_Live_View_example` notebooks.

The `ReCoDe_Live_View` notebook demonstrates a visualisation of ReCoDe data by superimposing all available frames to discern possible areas of interest. As data is captured in intervals and converted into the L1 format by ReCoDe, each part file generated will be read separately by ReCoDeViewer, then combined. Motion blur is not accounted for.

The `ReCoDe_Live_View_MT` notebook achieves the same visualisation as `ReCoDe_Live_View` but allows for multiprocessing. See `ReCoDe_Live_View_example` notebook for an updated version.

2. From v0.1 intermediate/ReCoDe files: See `Reading_ReCoDe_v0.1_Files` notebook.

The `Reading_ReCoDe_v0.1_Files` notebook introduces the process of reading data from intermediate files and ReCoDe files, as well as the process of merging intermediate files into a ReCoDe file. (ReCoDe files are designed for fast random access, while intermediate files only allow sequential access.)