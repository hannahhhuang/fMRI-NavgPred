class Config:
    def __init__(self):
        self.data_config = {
            # path
            "folder_path": "/home/hhuang474/navigation_predict/binary_classification/s02/ImagingData_s02_tifffile",
            "target_path": "/home/hhuang474/navigation_predict/processedOutput",
            # classes
            "classes": [27,87,147,207,267,327]
        }
        self.model_config = {
            "feats": [1,  16,  64, 256],
            "use_res": True
        }
        self.trainer_config = {
            "max_epochs": 10000,
            "accumu_steps": 1,
            "eval_frequency": 2,
            "ckpt_save_folder": "/home/hhuang474/navigation_predict/binary_classification/ckpt",
            "ckpt_load_path": None,#"/home/hhuang474/navigation_predict/binary classification/ckpt/99.ckpt",
            "ckpt_load_lr": True,
            "train_val_ratio": 0.8,
            "batch_size": 128,
            "lr": 1e-3,
            "gamma": 0.95
        }