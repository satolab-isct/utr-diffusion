import random

def generation_N_random_seq(fastafile='random_100k_selected.fasta', num:int =100000, min_len=50, max_len=50):
    # generation num random sequences with length between min_len and max_len
    # Then save to fasta file
    nucleotides = ['A', 'C', 'G', 'T']
    with open(fastafile, 'w') as f:
        for idx in range(num):
            length = random.randint(min_len, max_len)
            seq = ''.join(random.choices(nucleotides, k=length))
            f.write(f'>seq_{idx}\n')
            f.write(seq+'\n')
    print(f'Saved {num} random sequences of length {min_len}-{max_len} to', fastafile)


if __name__ == '__main__':
    generation_N_random_seq(fastafile='./../../save/random_N_sequences/random_100k_selected.fasta')