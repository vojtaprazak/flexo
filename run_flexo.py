from flexo_pipeline import *

#Essential inputs
rec_dir = '' # Str. Path to IMOD reconstruction directory
out_dir = '' # Str. Directory where output files are written out
mol_1_prm1 = '' # Str. Path to PEET parameter file. If it contains more than tomogram, use mol_1_prm_tomogram_number to specify which (numbered from 1)
#mol_1_prm2 = '' # Str. Specify if the data was already split into two
mol_1_average_volume = '' # Str. Path to volume to be backplotted. White on black (typically Chimera segmented average)

#Recommended
pre_exposure = 200 # Int. Useful range ~100-250. Applies low-pass filter corresponding to exposure (e/A^2) according to Grant+Grigorieff. 
#mdoc = '' # Str. Path to SerialEM .mdoc file. Looks for [raw stack name].mdoc in rec_dir by default.
#machines = 1 # Int. Specify numner of cores for parallelisation. Will use IMOD_CALIB_DIR shell variable if it exists (to get node names), otherwise uses local machine.
#mol_1_box_size = [40, 40] #List of two ints. 





#Optional

#No mdoc? use this:
#order = '' # Str. Path to order file. Use Empiar 10045 format.
#tilt_dose = 3.0 # Float. Flux per tilt (e/A^2)
#flux = 12.0 # Float. Electron flux (e/A^2/s). Read from .mdoc by default, use this if not calibrated.


#Imaging parameters (default values are listed)
#V = 300 # Int Accelerating voltage (kV).
#Cs = 2.7 # Int Spherical abberation (mm).
