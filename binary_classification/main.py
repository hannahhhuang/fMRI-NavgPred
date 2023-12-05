import torch.backends.cudnn

import config, dataset, model, trainer


if __name__=="__main__":
    cfg=config.Config()
    navdata = dataset.NavigationDataset(**cfg.data_config)

    resmodel = model.ResNet(**cfg.model_config)

    runner=trainer.Trainer(
        dataset=navdata,
        model=resmodel,
        **cfg.trainer_config

    )
    runner.fit()