AMINO_TO_CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],                  # Alanine           アラニン
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],    # Arginine          アルギニン
    'N': ['AAU', 'AAC'],                                # Asparagine        アスパラギン
    'D': ['GAU', 'GAC'],                                # Aspartic acid     アスパラギン酸
    'C': ['UGU', 'UGC'],                                # Cysteine          システイン
    'Q': ['CAA', 'CAG'],                                # Glutamine         グルタミン
    'E': ['GAA', 'GAG'],                                # Glutamic acid     グルタミン酸
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],                  # Glycine           グリシン
    'H': ['CAU', 'CAC'],                                # Histidine         ヒスチジン
    'I': ['AUU', 'AUC', 'AUA'],                         # Isoleucine        イソロイシン
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],    # Leucine           ロイシン
    'K': ['AAA', 'AAG'],                                # Lysine            リシン
    'M': ['AUG'],                                       # Methionine (START)メチオニン
    'F': ['UUU', 'UUC'],                                # Phenylalanine     フェニルアラニン
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],                  # Proline           プロリン
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],    # Serine
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],                  # Threonine         スレオニン
    'W': ['UGG'],                                       # Tryptophan        トリプトファン
    'Y': ['UAU', 'UAC'],                                # Tyrosine          チロシン
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],                  # Valine            バリン
    '*': ['UAA', 'UAG', 'UGA'],                         # Stop codons       終止コドン
}


CODON_TO_AMINO = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'AAU': 'N', 'AAC': 'N',
    'GAU': 'D', 'GAC': 'D',
    'UGU': 'C', 'UGC': 'C',
    'CAA': 'Q', 'CAG': 'Q',
    'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAU': 'H', 'CAC': 'H',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AAA': 'K', 'AAG': 'K',
    'AUG': 'M',
    'UUU': 'F', 'UUC': 'F',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'UGG': 'W',
    'UAU': 'Y', 'UAC': 'Y',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UAA': '*', 'UAG': '*', 'UGA': '*'
}


def rna_to_dna(seq: str) -> str:
    return seq.replace('U', 'T')


def dna_to_rna(seq: str) -> str:
    return seq.replace('T', 'U')


def get_codons_for_amino(amino: str) -> list:
    return [rna_to_dna(codon) for codon in AMINO_TO_CODONS[amino]]


def get_amino_for_codon(codon: str) -> str:
    return CODON_TO_AMINO[dna_to_rna(codon)]
