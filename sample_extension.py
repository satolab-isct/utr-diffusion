from accelerate import Accelerator
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.models.repaint.repaint_codon import RePaintSampler
import torch
import warnings
import os
import numpy as np
from tqdm import tqdm
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample(checkpoint_path):
    accelerator = Accelerator()

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2
    )

    diffusion = Diffusion(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=1,
        uncondition_prop=0.2,
    )
    checkpoint_dict = torch.load(checkpoint_path)
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    repaint = RePaintSampler(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        num_class=3,
        return_all=False
    )
    repaint = accelerator.prepare(repaint)

    total_seq_length = 1040
    win_size = 50
    step = 45
    total_sequences = [], []
    save_path = os.path.join('extension_save' ,checkpoint_path.split('/')[-1].replace('.pt', ''), 'full_1040bp_step_45bp')
    os.makedirs(save_path, exist_ok=True)

    gt, mask = bulid_gt_and_mask_from_codons(codon_list=[], pos_list=[])
    for i, win_pos in enumerate(range(0, total_seq_length - win_size + 1, step)):
        tqdm.write(f'Extension times:{i}, Window pos:{win_pos}-{win_pos + win_size}')
        segments = repaint.p_resample(gt=gt, mask=mask)
        new_part = segments[:, :, :, -step:]
        if win_pos == 0: # first time
            total_sequences = segments #
        else:
            total_sequences = np.concatenate((total_sequences, new_part), axis=-1)
        gt_known = torch.tensor(total_sequences[:, :, :, -(win_size - step) :])
        gt_unknown = torch.zeros_like(torch.tensor(new_part))
        mask_known, mask_unknown = torch.ones_like(gt_known), torch.zeros_like(gt_unknown)
        gt = torch.concat((gt_known, gt_unknown), dim=-1)
        mask = torch.concat((mask_known, mask_unknown), dim=-1)

        # save the segments in current window
        save_name = os.path.join(save_path, f"segments_{win_pos}-{win_pos + win_size}bp")
        torch.save(segments, save_name + '.pt')
        write_fasta(segments, save_name + '.fasta', num_class=3, batch_bs=100)

    save_name = os.path.join(save_path, 'full_sequences')
    torch.save(total_sequences, save_name + '.pt')
    write_fasta(total_sequences, save_name + '.fasta', num_class=3, batch_bs=100)


if __name__ == "__main__":
    checkpoint_path = 'checkpoints/MFE_30k_cls_3_[-15, -7.5, 0.0]_ep_2k_ts_200_beta_0.01_con_1_uncon_0.2_drop_0.2_at_2000epoch.pt'
    sample(checkpoint_path)




