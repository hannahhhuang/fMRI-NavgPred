# fMRI-NavgPred
## Data
This is the [raw dataset](https://osf.io/swav5/files/osfstorage).

This is the link to our [processed navigation data](https://www.dropbox.com/scl/fi/pbojewluy0tz95jsmtfd5/processed-navigation-data.zip?rlkey=186anxv8wzj13im3eadpnonzo&dl=0).

### File Structure:
```
processed navigation data
|-- subject (s02): Each participant
| |-- behavioral data
| | |-- run_001: Direction information for each experiment
| | |-- run_002
| | |-- ...
| |
| |-- ImagingData_processed: Processed into nifti format
| | |-- run_001: Imaging data for each experiment
| | | |-- xxx01.nii: 3D brain image for each sampling time (1.5s)
| | | |-- ...
| |
| |-- imagingData_tifffile: Processed into .tif format
| | |-- run_001.tif: Each file is a 4D array of size (Time, X, Y, Z). Each (X,Y,Z) slice is the Entorhinal cortex imaging after masking, with a sampling interval of 1.5s
| | |-- ...
| |
| |-- LEC.nii: Left Entorhinal cortex mask
| |-- rLEC.nii: Left Entorhinal cortex mask resliced to the same resolution as the images
|
|-- s03
| ...
```
# Video link: 
[dropbox link to our video](https://www.dropbox.com/scl/fi/xiroqzoz66apn635cqc3h/Screen-Recording-2023-11-07-at-23.46.18.mov?rlkey=6j6vr1mqn8sxduegfviysw447&dl=0)
