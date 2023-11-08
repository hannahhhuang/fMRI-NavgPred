import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm


class Trainer:
    def __init__(
            self, max_epochs, accumu_steps, eval_frequency,
            ckpt_save_folder, ckpt_load_path, ckpt_load_lr,
            dataset, train_val_ratio, batch_size,
            model, lr, gamma
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.accumu_steps = accumu_steps
        self.eval_frequency = eval_frequency

        # Paths
        self.ckpt_save_folder = ckpt_save_folder
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_load_lr = ckpt_load_lr

        # Split dataset
        train_num = int(len(dataset) * train_val_ratio)
        train_set, eval_set = random_split(dataset, [train_num, len(dataset) - train_num])
        train_set.dataset.train()
        eval_set.dataset.eval()

        # DataLoaders
        self.trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                      pin_memory=True)
        self.evaluloader = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                                      pin_memory=True)

        # Model
        self.model = model.to(self.device)

        # Loss function integrated into the model, assuming the forward returns loss
        # Optimizer and Scheduler
        self.scaler = amp.GradScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)

        # Recorder for TensorBoard
        self.writer = SummaryWriter()

        # Index
        self.epoch = 1  # Start from epoch 1 or load from checkpoint

        #loss function
        self.loss_fn=nn.CrossEntropyLoss()

        # Model info
        para_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {para_num:,} trainable parameters.")

    def fit(self):
        self._load_ckpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epochs+1), 
            total=self.max_epochs, desc=self.ckpt_save_folder, smoothing=0.0,
            unit="epoch", initial=self.epoch
                ):
            self._train_epoch()
            if (self.epoch+1) % self.eval_frequency == 0:
                self._eval_epoch()
                self._update_lr()
                self._save_ckpt() # Save checkpoint every eval_frequency epochs

    def _train_epoch(self):
        self.model.train()
        train_loss = []

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0
        )

        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with amp.autocast():
                pred = self.model(inputs)
                loss= self.loss_fn(pred, labels)
                loss /= self.accumu_steps

            self.scaler.scale(loss).backward()
            train_loss.append(loss.item())

            if (i + 1) % self.accumu_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        self.writer.add_scalar('Loss/Train', sum(train_loss) / len(train_loss), self.epoch)

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        eval_loss = []
        correct_pred=0
        total_pred=0

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.evaluloader)/self.accumu_steps), 
            desc="valid_epoch", leave=False, unit="steps", smoothing=1.0
        )

        for inputs, labels in self.evaluloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            pred= self.model(inputs)
            loss= self.loss_fn(pred, labels)
            eval_loss.append(loss.item())
            correct_pred += torch.sum(torch.argmax(pred,dim=1) == torch.argmax(labels,dim=1)).item()
            total_pred += len(labels)
        accuracy=correct_pred/total_pred

        self.writer.add_scalar('Loss/Eval', sum(eval_loss) / len(eval_loss), self.epoch)
        self.writer.add_scalar('Accuracy/Eval', accuracy, self.epoch)

    @torch.no_grad()
    def _update_lr(self):
        if self.scheduler.get_last_lr()[0] > 1e-8:
            self.scheduler.step()

    @torch.no_grad()
    def _save_ckpt(self):
        if not os.path.exists(self.ckpt_save_folder):
            os.makedirs(self.ckpt_save_folder)

        ckpt_path = os.path.join(self.ckpt_save_folder, f"{self.epoch}.ckpt")
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, ckpt_path)

    @torch.no_grad()
    def _load_ckpt(self):
        if self.ckpt_load_path and os.path.isfile(self.ckpt_load_path):
            ckpt = torch.load(self.ckpt_load_path)
            self.epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            if self.ckpt_load_lr:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"Checkpoint loaded: {self.ckpt_load_path} at epoch {self.epoch}")
        else:
            print("No checkpoint found, starting training from scratch.")