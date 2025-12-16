import os
from typing import Optional
import ipdb

import numpy as np
import pandas as pd
import torch
from pyparsing import with_class
from tqdm import tqdm
import itertools
from src.utils.utils import convert_to_seq

nucleotides = ["A", "C", "G", "T"]

def create_sample(
        diffusion_model,
        cell_types: list,
        conditional_numeric_to_tag: dict,
        number_of_samples: int = 1000,
        group_number: list | None = None,
        cond_weight_to_metric: int = 0,
        save_timesteps: bool = False,
        save_dataframe: bool = True,
        generate_attention_maps: bool = False,
    ):

    final_sequences = []
    for n_a in tqdm(range(number_of_samples)):
        sample_bs = 10
        
        if group_number:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        classes = sampled.float().to(diffusion_model.device)

        if generate_attention_maps:
            sampled_images, cross_att_values = diffusion_model.sample_cross(
                classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[group_number]}.npy", cross_att_values)

        else:
            sampled_images = diffusion_model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)
        
        if save_timesteps:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_sequences.append(pd.DataFrame(seqs_to_df))
        
        if save_dataframe:
            # Only using the last timestep
            for en, step in enumerate(sampled_images[-1]):
                final_sequences.append(convert_to_seq(step, nucleotides))
        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
                )
                final_sequences.append(seq_final)
    
    if save_timesteps:
        # Saving dataframe containing sequences for each timestep
        pd.concat(final_sequences, ignore_index=True).to_csv(
            f"final_{conditional_numeric_to_tag[group_number]}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return
    
    if save_dataframe:
        # Saving list of sequences to txt file
        print("Running Save Dataframe block")
        with open(f"final_{conditional_numeric_to_tag[group_number]}.txt", "w") as f:
            f.write("\n".join(final_sequences))
        return
    
    df_motifs_count_syn = extract_motifs(final_sequences)
    return df_motifs_count_syn

def inference(
    diffusion_model,
    class_num,             # int or list/tuple
    cond_weight: float = 0.0,
    sample_bs: int = 1000,
    seq_len: int = 50,
    output_all_steps: bool = False,
    target_values = None, # for continuous value generation
    device=None,
    with_condition=True,
):
    if target_values is None: # discrete value
        if isinstance(class_num, int):  # single-label
            return inference_single_label(
                diffusion_model=diffusion_model,
                class_num=class_num,
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
                with_condition=with_condition,
            )
        elif isinstance(class_num, (list, tuple)): # multi-label
            return inference_double_label(
                diffusion_model=diffusion_model,
                num_labels=len(class_num),
                num_classes=class_num[0],
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported class_num type: {type(class_num)}")
    else: # continuous value
        target_tensor = torch.tensor(target_values, dtype=torch.float32)
        if target_tensor.ndim == 1: #single label
            return inference_continuous_single_label(
                diffusion_model=diffusion_model,
                target_values=target_values,
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
            )
        else:
            return inference_continuous_double_label(
                diffusion_model=diffusion_model,
                target_values=target_values,
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
            )


def inference_single_label(
        diffusion_model,
        class_num: int = 3,
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps = False,
        device=None,
        with_condition=True
    ):
    final_sequences = []
    all_sampled_images = {}

    for label in range(1, class_num+1):
        labels = torch.full((sample_bs,),label, dtype=torch.float).to(device) if with_condition else None
        if output_all_steps:
            all_sampled_images[label] = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight, output_all_steps=True)
            sampled_image = all_sampled_images[label][-1]
            all_sampled_images[label] = torch.stack(all_sampled_images[label], dim=0).squeeze(2)
        else:
            sampled_image = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">class_{label}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_double_label(
    diffusion_model,
    num_labels: int = 2,
    num_classes: int = 3,
    cond_weight: float = 0.0,
    sample_bs: int = 1000,
    seq_len: int = 50,
    output_all_steps: bool = False,
    device=None,
):

    final_sequences = []
    all_sampled_images = {}

    # make all label combination e.g., [(1,1), (1,2), (1,3), ..., (3,3)]
    label_combinations = list(itertools.product(range(1, num_classes + 1), repeat=num_labels))
    for label_tuple in label_combinations:
        # 构建 label tensor, shape: [sample_bs, num_labels]
        labels = torch.tensor([label_tuple] * sample_bs, dtype=torch.float32).to(device)
        if output_all_steps:
            all_sampled = diffusion_model.sample(
                classes=labels,
                shape=(sample_bs, 1, 4, seq_len),
                cond_weight=cond_weight,
                output_all_steps=True
            )
            sampled_image = all_sampled[-1]  # 最后一帧
            all_sampled_images[str(label_tuple)] = torch.stack(all_sampled, dim=0).squeeze(2)  # [T, B, 4, L]
        else:
            sampled_image = diffusion_model.sample(
                classes=labels,
                shape=(sample_bs, 1, 4, seq_len),
                cond_weight=cond_weight
            )

        # decode to sequence
        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">class_{label_tuple}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_continuous_single_label(
        diffusion_model,
        target_values: list = [4.0, 6.0, 8.0],
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps = False,
        device=None,
    ):
    final_sequences = []
    all_sampled_images = {}

    for idx, tgt in enumerate(target_values):
        labels = torch.full((sample_bs,),tgt, dtype=torch.float).to(device)
        if output_all_steps:
            all_sampled_images[idx] = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight, output_all_steps=True)
            sampled_image = all_sampled_images[idx][-1]
            all_sampled_images[idx] = torch.stack(all_sampled_images[idx], dim=0).squeeze(2)
        else:
            sampled_image = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">target_{tgt}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_continuous_double_label(
        diffusion_model,
        target_values: list = [[4.0, -10.0], [6.0, -5.0], [8.0, -5.0]],
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps: bool = False,
        device=None,
    ):
    final_sequences = []
    all_sampled_images = {}

    for idx, (mrl, mfe) in enumerate(target_values):
        labels = torch.tensor([[mrl, mfe]] * sample_bs, dtype=torch.float32).to(device)
        if output_all_steps:
            all_sampled_images[idx] = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight, output_all_steps=True)
            sampled_image = all_sampled_images[idx][-1]
            all_sampled_images[idx] = torch.stack(all_sampled_images[idx], dim=0).squeeze(2)
        else:
            sampled_image = diffusion_model.sample(labels, (sample_bs, 1, 4, seq_len), cond_weight)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            header = f">target_MRL_{mrl}_MFE_{mfe}_seq_{n}"
            final_sequences.append(header + "\n" + "".join(seq) + "\n")

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def extract_motifs(sequence_list: list):
    """Extract motifs from a list of sequences"""
    motifs = open("synthetic_motifs.fasta", "w")
    motifs.write("\n".join(sequence_list))
    motifs.close()
    os.system("gimme scan synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 -n 20 > syn_results_motifs.bed")
    df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_syn["motifs"] = df_results_syn[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].groupby("motifs").count()
    return df_motifs_count_syn


def convert_sample_to_fasta(sample_path: list):
    """Convert cell specific samples to a fasta format"""
    sequences = []
    samples = pd.read_csv(sample_path, sep="\t", header=None)
    # Extract each line of the dataframe into a list
    samples_list = samples[0].tolist()
    # Convert into a fasta format
    for i, seq in enumerate(samples_list):
        sequences.append(f">sequence_{i}\n" + seq)
    return sequences
