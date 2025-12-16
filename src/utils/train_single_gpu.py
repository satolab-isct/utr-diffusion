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
from src.data.dataloader_diy_data import gumbel_softmax
from .train_loop_basic import BasicTrainLoop

class TrainLoop_single_gpu(BasicTrainLoop):
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
        do_gumbel_softmax: bool = False,
        tgt_values = None, # if None discrete else continueous
    ):
        super().__init__(model=model, accelerator=accelerator, start_epoch=start_epoch, end_epoch=end_epoch, log_step=log_step,
                         valid_epoch=valid_epoch, sample_epoch=sample_epoch, save_epoch=save_epoch, save_name=save_name,
                         batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate,
                         num_classes= data['Classes'] if 'Classes' in data else 3)

        # some setting params
        self.do_gumbel_softmax = do_gumbel_softmax
        self.tgt_values = tgt_values

        # Dataloader and Learning schedule
        self.train_dl, self.valid_dl = self._prepare_data_loader(data, batch_size, num_workers)
        self.schedule = lr_schedule(optimizer=self.optimizer,
                                    num_training_steps=len(self.train_dl) * self.end_epoch if self.train_dl is not None else 1,
                                    warmup_rate=0.05, flatten_rate=0.7,)

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
        # Prepare for training
        self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule)

        self.log_update(mode='init')
        for epoch in tqdm(range(self.start_epoch, self.end_epoch + 1)):
            # training
            self.train(epoch=epoch)

            #validation
            if epoch % self.valid_epoch == 0:
                self.validation(epoch=epoch)

            # Sampling
            if epoch % self.sample_epoch == 0:
                self.sample(epoch=epoch)

            # Saving checkpoint
            if epoch % self.save_epoch == 0:
                self.save_checkpoint(epoch=epoch)

        self.accelerator.end_training()
        return self.valid_loss ## validation modal is not finished yet


    def train(self, epoch):
        self.model.train()  # shift to train mode
        for step, batch in enumerate(self.train_dl):
            x, y = batch
            x = gumbel_softmax(x, scale=3, tau=0.8, hard=False) if self.do_gumbel_softmax else x
            with self.accelerator.autocast():  # Mixed precision on 混合精度オンにする
                loss = self.model(x, y)

            # update optimizer, scheduler and global step
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.schedule.step()
            self.global_step += 1

            # logging
            if self.global_step % self.log_step == 0:
                self.train_loss = loss.item()
                self.log_update(mode='train', epoch=epoch)


    def validation(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with self.accelerator.autocast():
                for batch in self.valid_dl:
                    x, y = batch
                    x = gumbel_softmax(x, scale=1, tau=1, hard=True) if self.do_gumbel_softmax else x
                    loss = self.model(x, y)
                    total_loss += loss.item()

        self.valid_loss = total_loss / len(self.valid_dl)
        if self.valid_loss <= self.best_valid_loss:
            self.best_valid_loss = self.valid_loss

        # logging
        self.log_update(mode='valid', epoch=epoch)



    def sample(self, epoch, write_fasta=True):
        self.model.eval()
        sample_fn = partial(
            inference,
            diffusion_model=self.model,
            class_num=self.num_classes,
            cond_weight=self.model.cond_weight,
            target_values=self.tgt_values,
            device=self.accelerator.device
        )
        with torch.no_grad():
            with self.accelerator.autocast():
                print("\nGenerating synthetic sequences...")
                if epoch==self.end_epoch and self.is_save_process:
                    seqs, all_images = sample_fn(output_all_steps=True)
                    torch.save({k: v.cpu().numpy() for k, v in all_images.items()},
                               os.path.join(self.save_name, "all_images_denoising_process.pt"))
                    print("all_images_denoising_process.pt saved!")
                else:
                    seqs = sample_fn()

            if write_fasta:
                print('Saving fasta file...')
                write_to_fasta(sequences=seqs, folder_name=self.save_name, epoch=epoch)

    
    def load_checkpoint_then_do_sample(self, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        self.model = self.accelerator.prepare(self.model)
        self.sample(epoch=self.start_epoch, write_fasta=True)
        



