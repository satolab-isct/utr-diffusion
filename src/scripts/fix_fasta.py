import os

if __name__ == '__main__':
    root = 'repaint_save/MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch/codon_CDS_exp'

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.fasta'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                fixed_lines = []
                for i, line in enumerate(lines):
                    if i % 2 == 0:  # 奇数行（从0开始计数）
                        if not line.startswith('>'):
                            line = '>_' + line.strip() + '\n'
                    fixed_lines.append(line)

                with open(filepath, 'w') as f:
                    f.writelines(fixed_lines)
                print(f"✅ fixed: {filepath}")