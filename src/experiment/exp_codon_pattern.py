import numpy as np

pattern_single_codon = {
    'name': 'single_codon',
    'codon': ['AGC'],
    'pos': [11],
}


pattern_double_codon = {
    'name': 'double_codon',
    'codon': ['TTG', 'ATG'],
    'pos': [5, 41],
}

pattern_triple_codon = {
    'name': 'triple_codon',
    'codon': ['AGG', 'GCG', 'TTA'],
    'pos': [14, 23, 35],
}

pattern_quadra_codon = {
    'name': 'quadra_codon',
    'codon': ['TGC', 'AAC', 'GGA', 'CTT'],
    'pos': [8, 17, 29, 38],
}

pattern_penta_codon = {
    'name': 'penta_codon',
    'codon': ['ATG', 'GCC', 'TAC', 'CAG', 'TTT'],
    'pos': [2, 20, 26, 32, 44],
}

pattern_half_sequence = {
    'name': 'half_sequence',
    'codon': ['CCAGCTTGGTGAACTCGGTGTTGGA'],
    'pos': [0]
}

pattern_codon_stripes = {
    'name': 'codon_stripes',
    'codon': ['AGC', 'GTG', 'TCG', 'TTG', 'TGT', 'GCC', 'CGT', 'CGA'],
    'pos': [2, 8, 14, 20, 26, 32, 38, 44],
}



Codon_Patterns = [
    pattern_single_codon,
    pattern_double_codon,
    pattern_triple_codon,
    pattern_quadra_codon,
    pattern_penta_codon
]

# Experiment: Fixed amino-acid with alternative codons

pattern_single_amino = {
    'name': 'single_amino',
    'amino': ['P'],
    'pos': [5]
}


pattern_double_amino = {
    'name': 'double_amino',
    'amino': ['R', 'L'],
    'pos': [8, 23]
}

pattern_triple_amino = {
    'name': 'triple_amino',
    'amino': ['K', 'S', 'V'],
    'pos': [14, 26, 41]
}

pattern_quadra_amino ={
    'name': 'quadra_amino',
    'amino': ['A', 'F', 'G', 'Y'],
    'pos': [10, 22, 35, 47]
}

pattern_penta_amino = {
    'name': 'penta_amino',
    'amino': ['R', 'T', 'L', 'H', 'V'],
    'pos':   [5, 14, 25, 36, 45]
}

pattern_dozen_amino = {
    'name': 'dozen_amino',
    'amino': ['M', 'F', 'D', 'A', 'W', 'E', 'Q', 'Y', 'G', 'N', 'H', 'T'],
    'pos':   [2,   5,   9,  12,  17,  22,  26,  29,  33,  37,  42,  47]
}



Amino_Patterns = [
    pattern_single_amino,
    pattern_double_amino,
    pattern_triple_amino,
    pattern_quadra_amino,
    pattern_penta_amino
]