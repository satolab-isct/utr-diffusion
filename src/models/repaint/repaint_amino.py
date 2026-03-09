import torch
from einops import rearrange
from .scheduler import get_schedule_jump
from tqdm.auto import tqdm
from .utils import get_amino_images_for_alter_codons, build_gt_from_image_and_pos

schedule_jump_params = {
    'jump_length': 10,  # jump length for each jump
    'jump_n_sample': 10,  # jump frequency for every n steps
    'n_sample': 1,
    't_T': 200
}


# this repaint sampler works for continuous multi-label generation
class RePaint_Amino:
    def __init__(self,
                 diffusion,
                 sample_bs: int = 100,
                 seq_len: int = 50,
                 cond_weight: float = 2.0,
                 num_class: int = 3,
                 strategy='wasserstein',
                 stop_point=1.0,
                 return_all=True):
        self.model = diffusion
        self.device = diffusion.device
        self.sample_bs = sample_bs
        self.seq_len = seq_len
        self.cond_weight = cond_weight
        self.num_class = num_class
        self.return_all = return_all
        self.shape = [sample_bs * num_class] + [1, 4, seq_len]
        self.strategy = strategy
        self.stop_point = stop_point

    def p_resample(self, gt, mask, tgt_aminos, pos_list):
        labels = []
        for class_id in range(1, self.num_class + 1):
            labels += [class_id] * self.sample_bs
        labels = torch.tensor(labels, dtype=torch.float).to(self.device)
        gt = gt.expand(self.sample_bs * self.num_class, -1, -1, -1).to(self.device)
        mask = mask.expand(self.sample_bs * self.num_class, -1, -1, -1).to(self.device)
        all_amino_images = get_amino_images_for_alter_codons(tgt_aminos, with_padding=True).to(self.device)
        result = {
            'samples': [],
            'forward_steps': [],
            'backward_steps': []
        }

        for sample, forward_step, backward_step in self.p_resample_loop(gt, mask, labels, all_amino_images, pos_list):
            result['samples'].append(sample.detach().cpu().to(torch.float16).numpy())
            result['forward_steps'].append(forward_step)
            result['backward_steps'].append(backward_step)

        if self.return_all:
            return result
        else:
            return result['samples'][-1]

    def p_resample_loop(self, gt, mask, labels, tgt_amino_images, amino_pos):
        backward_step_count = 0
        forward_step_count = 0

        times = get_schedule_jump(**schedule_jump_params)
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = tqdm(time_pairs)

        n_sample = self.shape[0]
        image = torch.randn(self.shape, device=self.device)

        context_mask = torch.ones_like(labels).to(self.device)
        # double the batch and make 0 index unconditional
        labels = labels.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0

        for t_last, t_cur in time_pairs:
            if t_cur < t_last:  # reverse
                backward_step_count += 1
                # early stopping
                early_stop = backward_step_count + forward_step_count > self.stop_point * len(time_pairs)
                if not early_stop:
                    gt = self.apply_codon_flexibility(gt, image, mask, tgt_amino_images, amino_pos)

                gt_noised = self.noise_steps(x_0=gt, t=t_last)
                mixed_image = gt_noised * mask + image * (1 - mask)
                timesteps = torch.full((n_sample,), t_last, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    image = self.model.p_sample_guided(x=mixed_image, t=timesteps, classes=labels,
                                                       cond_weight=self.cond_weight,
                                                       context_mask=context_mask, t_index=t_last)
            else:
                forward_step_count += 1
                image = self.noise_step(x_t_m1=image, t=t_last)

            yield image, forward_step_count, backward_step_count

    def noise_step(self, x_t_m1, t):
        B = x_t_m1.shape[0]
        t = torch.full((B,), t, dtype=torch.long, device=self.device)
        x_t = self.model.q_sample_single_step(x_t_m1, t)
        return x_t

    def noise_steps(self, x_0, t):
        B = x_0.shape[0]
        t = torch.full((B,), t, dtype=torch.long, device=self.device)
        x_t = self.model.q_sample(x_0, t)
        return x_t

    def apply_codon_flexibility(self, gt, image, mask, all_amino_images, amino_pos) -> torch.Tensor:

        if self.strategy == 'init_random':
            return gt
        # mask, image [B, 1, C, L]
        mask_1d = mask[0, 0, 0, :]
        index = mask_1d.nonzero(as_tuple=False).flatten()
        fix_region = image[:, :, :, index]  # [B, 1, 4, 50] -> [B, 1, 4, 3 * N]

        new_codon_images = choose_codon_by_strategy(fix_region, all_amino_images, strategy=self.strategy)  # [B, N, 6]
        gt_image = build_gt_from_image_and_pos(codon_images=new_codon_images, pos_list=amino_pos, device=self.device)

        return gt_image


def choose_codon_by_strategy(query: torch.Tensor, candidates: torch.Tensor,
                             strategy: str = 'wasserstein') -> torch.Tensor:
    """
    query:     [B, 1, 4, 3 * N] — N: number of specified amino
    candidates:[N, 6, 4, 3] — all amino images, one amino mapping 6 codons at max
    return:    [B, N, 6] —
    """
    batch_size = query.shape[0]
    query = rearrange(query, 'b c h (n p) -> b (c n) h p', p=3)  # [B, 1, 4, 3 * N] -> [B, N, 4, 3]
    query = query.unsqueeze(2).expand(-1, -1, 6, -1, -1)  # [B, N, 4, 3] -> [B, N, 6, 4, 3]
    candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [N, 6, 4, 3] -> [B, N, 6, 4, 3]

    if strategy == 'euclidean':
        dist = euclidean_distance(query, candidates)
    elif strategy == 'wasserstein':
        dist = wasserstein_1d_distance(query, candidates)
    else:
        raise ValueError(f'Unknown strategy {strategy}')

    codon_idx = dist.argmin(-1)  # [B, N, 6] -> [B, N]
    codon_idx_exp = codon_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3)  # [B, N] -> [B, N, 4, 3]
    codon_images = torch.gather(candidates, 2, codon_idx_exp.unsqueeze(2)).squeeze(2)

    return codon_images  # [B, N, 4, 3]


def wasserstein_1d_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    # query, candidates [B, N, 6, 4, 3]
    # output: [B, N, 6]

    # CDF over base dimension
    query_cdf = torch.cumsum(query, dim=3)  # [B, N, 6, 4, 3]
    candidate_cdf = torch.cumsum(candidates, dim=3)

    # abs diff * support
    wasser = (query_cdf - candidate_cdf).abs()  # w = 1
    wasser = wasser.sum(dim=[3, 4])

    return wasser


def euclidean_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    # query, candidates [B, N, 6, 4, 3]
    # output: [B, N, 6]
    diff = (query - candidates).abs()
    dist = torch.sqrt(diff ** 2).sum(dim=[3, 4])

    return dist


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query = torch.randn(2, 2, 4, 3).to(device)
    candidates = torch.randn(2, 2, 6, 4, 3).to(device)
    wasser = wasserstein_1d_distance(query, candidates)