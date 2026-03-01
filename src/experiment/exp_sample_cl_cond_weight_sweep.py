from os import rename

from tensorflow.python.keras.saving.saving_utils import raise_model_input_error
from src.experiment.exp_target_labels import mrl_target_values_sweep, mfe_target_values_sweep
from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CL
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CL
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
from src.utils.utils import rename_fasta_cond_to_condition_weight, collect_fasta_from_subdirs

import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
def sample_continuous_single_label_condition_weight_sweep(mode=None):
    unet = UNet_CL(
        dim=200,  # 200
        channels=1,
        dim_mults=(1, 2, 4),  # (1,2,4)
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2,
        num_label=1,
    )

    if mode == 'MFE':
        model_save_folder = "save/MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch_sample_sweep_cond_sweep"
        checkpoint_path = "checkpoints/MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch.pt"
        target_values = mfe_target_values_sweep
    elif mode == 'MRL':
        model_save_folder = "save/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch_sample_sweep_cond_sweep"
        checkpoint_path = "checkpoints/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch.pt"
        target_values = mrl_target_values_sweep
    else:
        raise ValueError('Wrong label type')

    cond_weights = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 100]
    for cond_weight in cond_weights:
        diffusion = Diffusion_CL(
            model=unet,
            timestep=200,
            beta_last=0.01,
            condition_weight=cond_weight,
            uncondition_prop=0.2,
        )

        accelerator = Accelerator()
        model_save_name = os.path.join(ROOT, model_save_folder, f'cond_{cond_weight}')
        checkpoint_path = os.path.join(ROOT, checkpoint_path)

        TrainLoop(
            data={},
            model=diffusion,
            accelerator=accelerator,
            end_epoch=2000,
            log_step=8,   # how many steps to show the log トレーニングログを表示するステップ数
            valid_epoch=5,
            sample_epoch=200,
            save_epoch=2000,
            save_name=model_save_name,
            batch_size=5200,
            num_workers = 16,
            learning_rate=1e-4,
            do_gumbel_softmax=False,
            tgt_values = target_values,
        ).load_checkpoint_then_do_sample(checkpoint_path)

    collect_fasta_from_subdirs(model_save_folder)


if __name__ == "__main__":
    #sample_continuous_single_label_condition_weight_sweep(mode='MRL')
    target_dir = os.path.join(ROOT, 'save/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch_sample_sweep_cond_sweep')
    #collect_fasta_from_subdirs(target_dir=target_dir)
    rename_fasta_cond_to_condition_weight(target_dir=target_dir, old="cond", new="condition_weight")