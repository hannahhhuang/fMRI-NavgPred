import nibabel as nib
import numpy as np
import tifffile as tiff
import os

class NIfTIConverter:
    def __init__(self, folder_path, mask_path, output_path):
        """
        folder_path: path to folder containing run_00x... folders
        mask_path: path to mask file
        output_path: path to output TIFF file 
        
        example usage:
        NIfTIConverter('./Navigation data/s02/ImagingData_s02_processed','/Users/hannah/GT/PSYC4803/My4803Folder/Navigation data/s02/rLEC.nii','./Navigation data/s02/ImagingData_s02_tifffile').convert()  
        """

        self.folder_path = folder_path
        self.mask_path = mask_path
        self.output_path =output_path

    def convert(self):
        for folder_name in os.listdir(self.folder_path):
            if folder_name.startswith("."):
                continue
            subfolder_path = os.path.join(self.folder_path, folder_name)
            time_img_data = []
            for filename in os.listdir(subfolder_path):
                if folder_name.startswith(".") or folder_name not in ["run_001", "run_002", "run_003", "run_004", "run_005", "run_006", "run_007", "run_008", "run_009", "run_010" , "run_011", "run_012"]:
                    continue
                img_path= os.path.join(subfolder_path, filename)
                img_data = nib.load(img_path).get_fdata()
                mask = nib.load(self.mask_path).get_fdata()
                img_data_masked = img_data * mask
                time_img_data.append(img_data_masked) # T, X, Y, Z
            time_img_data = np.array(time_img_data)
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            output_path = os.path.join(self.output_path,folder_name + ".tif")
            tiff.imsave(output_path, time_img_data)
            print(f"Saved TIFF file at {output_path}")

if __name__ == "__main__":
    for i in range(3,23):
        if i==11:
            continue
        if i < 10:
            subject = "s0" + str(i)
        else:
            subject = "s" + str(i)
        NIfTIConverter('./Navigation data/' + subject + '/ImagingData_' + subject + '_processed','./Navigation data/' + subject + '/rREC.nii','./Navigation data/' + subject + '/ImagingData_' + subject + '_tifffile_right').convert()