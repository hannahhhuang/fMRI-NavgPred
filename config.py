class Config:
    def __init__(self):
        self.data_config = {
            # path
            "folder_path": "/Users/hannah/GT/PSYC4803/My4803Folder/Navigation data/s02/ImagingData_s02_tifffile",
            "target_path": "/Users/hannah/GT/PSYC4803/My4803Folder/processedOutput",
            # classes
            "classes": [50,110,170,230,290,350]
        }
        self.model_config = {
            "feats": [1,  16,  64, 256],
            "use_res": True
        }
        self.trainer_config = {
            "max_epochs": 100,
            "accumu_steps": 1,
            "eval_frequency": 10,
            "ckpt_save_folder": "/Users/hannah/GT/PSYC4803/My4803Folder/ckpt",
            "ckpt_load_path": None,
            "ckpt_load_lr": False,
            "train_val_ratio": 0.8,
            "batch_size": 2,
            "lr": 1e-3,
            "gamma": 0.95
        }