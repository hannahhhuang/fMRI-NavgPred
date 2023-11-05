# fMRI-NavgPred
## Data
This is the link to our [processed navigation data](https://www.dropbox.com/scl/fi/pbojewluy0tz95jsmtfd5/processed-navigation-data.zip?rlkey=186anxv8wzj13im3eadpnonzo&dl=0).

The data is in the following format:
processed navigation data
    --subject (s02): 每个participant
        -- behavioral data
            -- run number: 每次实验的 direction 信息
        -- imagingData_processed: 处理成了nifti 格式
            -- run number: 每一次实验 的imaging data
                -- xxx01.nii: 每一个采样时间(1.5s)的3d 大脑图像
        -- imagingData_tifffile: 处理成了.tif 格式
            -- runnumber.tif: 每个file是 (Time, X, Y, Z) 的 4D array，每个XYZ是经过mask之后的Entorhinal cortex, time 间隔1.5s 采样时间
        -- LEC.nii: Left Entorhinal cortex mask
        -- rLEC.nii: Left Entorhinal cortex mask after reslicing to same resolution as the images
    -- s03
    ...
