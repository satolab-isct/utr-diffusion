from typing import Any
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from src.data.dataloader import SequenceDataset
from src.utils.sample_util import inference
from src.utils.utils import write_to_fasta
from src.utils.utils import get_warmup_flatten_cosine_schedule as lr_schedule
from .train_loop_basic import BasicTrainLoop

class TrainLoop_multi_gpu(BasicTrainLoop):
    def __init__(
            self,
            data: dict[str, Any],
            model: torch.nn.Module,
            accelerator: Accelerator,
            start_epoch: int = 1,
            end_epoch: int = 10000,
            log_step: int = 50,
            valid_epoch: int = 5,
            sample_epoch: int = 500,
            save_epoch: int = 500,
            save_name: str = '',
            batch_size: int = 960,
            num_workers: int = 4,
            learning_rate: float = 1e-3,
            tgt_values=None,  # if None discrete else continuous
            with_condition=True,
    ):
        super().__init__(model=model, accelerator=accelerator, start_epoch=start_epoch, end_epoch=end_epoch, log_step=log_step,
                         valid_epoch=valid_epoch, sample_epoch=sample_epoch, save_epoch=save_epoch, save_name=save_name,
                         batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate,
                         num_classes=data['Classes'] if 'Classes' in data else 3)

        # some setting params
        self.tgt_values = tgt_values
        self.with_condition = with_condition

        # Dataloader and Learning schedule
        self.train_dl, self.valid_dl = self._prepare_data_loader(data, batch_size, num_workers)
        self.schedule = lr_schedule(
            optimizer=self.optimizer,
            num_training_steps=len(self.train_dl) * self.end_epoch if self.train_dl is not None else 1,
            warmup_rate=0.05,
            flatten_rate=0.7,
        )


    def _prepare_data_loader(self, data, batch_size, num_workers):
        if data != {}:  # case "data={}" for sample only
            seq_train = SequenceDataset(seqs=data["Train"], c=data['Train_label'])
            seq_valid = SequenceDataset(seqs=data["Valid"], c=data['Valid_label'])
            train_dl = DataLoader(seq_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            valid_dl = DataLoader(seq_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
            return train_dl, valid_dl
        else:
            print('Training data not provided, running in sampling mode!!')
            return None, None


    def train_loop(self):
        # Multi-GPU prepare Model, Optimizer and Accelerator
        self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule)

        self.log_update(mode='init')

        for epoch in tqdm(range(self.start_epoch, self.end_epoch + 1), disable=not self.accelerator.is_main_process):
            # training
            self.train(epoch=epoch)

            # validation
            if epoch % self.valid_epoch == 0:
                self.validation(epoch=epoch)

            # sampling (only rank=0)
            if epoch % self.sample_epoch == 0 and self.accelerator.is_main_process:
                self.sample(epoch=epoch)

            # checkpoint (only rank=0)
            if epoch % self.save_epoch == 0 and self.accelerator.is_main_process:
                self.save_checkpoint(epoch=epoch, multi_gpu_enabled=True)

        self.accelerator.end_training()
        return self.valid_loss


    def train(self, epoch):
        self.model.train()
        for step, batch in enumerate(self.train_dl):
            x, y = batch
            with self.accelerator.autocast():  # Mixed precision on 混合精度オンにする
                y = y if self.with_condition else None
                loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.accelerator.wait_for_everyone()
            self.optimizer.step()
            self.schedule.step()
            self.global_step += 1

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

            if self.global_step % self.log_step == 0: # logging
                avg_loss = self.accelerator.gather(loss.detach()).mean()
                if self.accelerator.is_main_process:
                    self.train_loss = avg_loss.item()
                    self.log_update(mode='train', epoch=epoch)


    def validation(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with self.accelerator.autocast():
                for batch in self.valid_dl:
                    x, y = batch
                    y = y if self.with_condition else None
                    loss = self.model(x, y)
                    total_loss += loss.detach()

            self.valid_loss = total_loss / len(self.valid_dl) # batch average
            self.valid_loss = self.accelerator.reduce(self.valid_loss, reduction="mean") # multi-GPUs average
        if self.valid_loss <= self.best_valid_loss:
            self.best_valid_loss = self.valid_loss

        if self.accelerator.is_main_process:
            self.log_update(mode='valid', epoch=epoch)


    def sample(self, epoch, write_fasta=True):
        # use EMA model
        device = self.accelerator.device
        model_for_sampling = self.ema_model if hasattr(self, "ema_model") else self.model
        model_for_sampling = self.accelerator.unwrap_model(model_for_sampling)
        model_for_sampling.to(device)
        model_for_sampling.eval()

        sample_fn = partial(
            inference,
            diffusion_model=model_for_sampling,
            class_num=self.num_classes,
            cond_weight=getattr(model_for_sampling, "cond_weight", 0.0),
            target_values=self.tgt_values,
            device=device,
            with_condition=self.with_condition,
        )

        with torch.no_grad():
            with self.accelerator.autocast():
                print("\nGenerating synthetic sequences...")
                if epoch == self.end_epoch:
                    seqs, all_images = sample_fn(output_all_steps=True)
                    torch.save({k: v.cpu().numpy() for k, v in all_images.items()},
                               os.path.join('save', self.save_name, "all_images_denoising_process.pt"))
                    print("all_images_denoising_process.pt saved!")
                else:
                    seqs = sample_fn()

            if write_fasta:
                print('Saving fasta file...')
                write_to_fasta(sequences=seqs, folder_name=os.path.join('save', self.save_name), epoch=epoch)