# fMRI-NavgPred
## Data
This is the link to our [processed navigation data](https://www.dropbox.com/scl/fi/pbojewluy0tz95jsmtfd5/processed-navigation-data.zip?rlkey=186anxv8wzj13im3eadpnonzo&dl=0).

The data is in the following format:
processed navigation data
- **Behavioral Data**: Contains information related to the participant's behavior during each experimental run.
- **Imaging Data**: Two formats are provided here:
  - **Processed NIfTI**: Standard format for storing neuroimaging data with `.nii` extension.
  - **TIFF Files**: Each `.tif` file contains a 4D array with masked brain region data, with a 1.5-second interval between each time point.
- **Cortical Masks**:
  - `LEC.nii`: Mask file for the Left Entorhinal Cortex.
  - `rLEC.nii`: Resliced version of the Left Entorhinal Cortex mask to align with the resolution of the imaging data.

