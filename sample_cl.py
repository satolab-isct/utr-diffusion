from src.experiment.exp_target_labels import mrl_target_values_sweep, mfe_target_values_sweep
from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CL
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CL
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def sample_continuous_single_label():
    unet = UNet_CL(
        dim=200,  # 200
        channels=1,
        dim_mults=(1, 2, 4),  # (1,2,4)
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2,
        num_label=1,
    )

    diffusion = Diffusion_CL(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=1,
        uncondition_prop=0.2,
    )

    accelerator = Accelerator()
    model_save_name = "save/MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch_sample_sweep"
    checkpoint_path = "checkpoints/MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch.pt"

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
        tgt_values = mfe_target_values_sweep,
    ).load_checkpoint_then_do_sample(checkpoint_path)

if __name__ == "__main__":
    sample_continuous_single_label()