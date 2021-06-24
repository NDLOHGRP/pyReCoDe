### ReCoDe examples
Ipython notebooks showcasing examples of pyReCoDe and its related applications.

---

__To view ReCoDed data:__

__1. From ReCoDe part files: See `ReCoDe_Live_View`/`ReCoDe_Live_View_MT`/`ReCoDe_Live_View_example` notebooks.__

The `ReCoDe_Live_View` notebook demonstrates a visualisation of ReCoDe data by superimposing all available frames to discern possible areas of interest. As data is captured in intervals and converted into the L1 format by ReCoDe, each part file generated will be read separately by ReCoDeViewer, then combined. Motion blur is not accounted for.

The `ReCoDe_Live_View_MT` notebook achieves the same visualisation as `ReCoDe_Live_View` but allows for multiprocessing. See `ReCoDe_Live_View_example` notebook for an updated version.

__2. From v0.1 intermediate/ReCoDe files: See `Reading_ReCoDe_v0.1_Files` notebook.__

The `Reading_ReCoDe_v0.1_Files` notebook introduces the process of reading data from intermediate files and ReCoDe files, as well as the process of merging intermediate files into a ReCoDe file. (ReCoDe files are designed for fast random access, while intermediate files only allow sequential access.)

---

__For calibration/conversion:__

The `Fine_Calibration_with_Backscattering` notebook describes in detail the fine calibration process, including a simple model for estimating the number of backscattering events. 

The `recalibration_and_conversion.py` script may be used to calibrate L1 ReCoDe data and convert L1 to L4. (Refer to paper published at https://doi.org/10.1038/s41467-020-20694-z for the 4 possible data reduction levels.)

---

Refer to `recode_server.py` for an example on initialising the ReCoDe Server.