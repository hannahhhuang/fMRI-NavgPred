# fMRI-NavgPred
## Data
This is the link to our [processed navigation data](https://www.dropbox.com/scl/fi/pbojewluy0tz95jsmtfd5/processed-navigation-data.zip?rlkey=186anxv8wzj13im3eadpnonzo&dl=0).

The data is in the following format:
processed navigation data
    |-- subject (s02): each participant
    | |-- behavioral data
    | | |-- run number: direction information for each experiment
    | |
    | |-- imagingData_processed: processed into nifti format
    | | |-- run number: imaging data for each experiment
    | | | |-- xxx01.nii: 3D brain image for each sampling time (1.5s)
    | |
    | |-- imagingData_tifffile: processed into .tif format
    | | |-- runnumber.tif: each file is a 4D array of (Time, X, Y, Z),
    | | | each XYZ is the Entorhinal cortex after masking, with a sampling interval of 1.5s
    | |
    | |-- LEC.nii: Left Entorhinal cortex mask
    | |-- rLEC.nii: Left Entorhinal cortex mask resliced to the same resolution as the images
    |
    |-- s03
    | ...
