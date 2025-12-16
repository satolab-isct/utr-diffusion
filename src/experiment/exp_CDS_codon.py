# CDS area from Bian-san
pattern_triple_codons = {
    'name': 'triple_codon',
    'codon': ['AGG', 'GCG', 'TTA'],
    'pos': [14, 23, 41],
}

CDS_Sample_31 = {
    'name': 'Sample_number_31',
    'codon': 'ATGGTGTTCACTTTGGAGGATTTTGTGGGTGATTGGAGACAGACAGCAGG'
}

CDS_Sample_70 = {
    'name': 'Sample_number_70',
    'codon': 'ATGGTGTTCACTCTGGAAGATTTTGTGGGAGATTGGAGGCAGACTGCAGG'
}

CDS_Sample_14 = {
    'name': 'Sample_number_14',
    'codon': 'ATGGTGTTCACCCTTGAGGATTTCGTGGGAGACTGGAGACAGACAGCCGG'
}

CDS_Sample_25 = {
    'name': 'Sample_number_25',
    'codon': 'ATGGTGTTCACATTGGAAGACTTTGTTGGTGATTGGAGACAGACTGCAGG'
}

CDS_Sample_52 = {
    'name': 'Sample_number_52',
    'codon': 'ATGGTGTTCACATTAGAAGATTTTGTTGGGGACTGGAGACAGACAGCTGG'
}

def get_codon_pattern_from_sample(cds_seq, cds_length:int, total_length: int = 50):
    name = cds_seq['name'] + f'_length_{cds_length}'
    pos = total_length - cds_length
    codon_truncated = cds_seq['codon'][:cds_length]
    pattern= {
        'name': name,
        'codon': [codon_truncated],   # must in list data format [str]
        'pos': [pos]                  # must in list data format [int]
    }
    return pattern

def get_amino_pattern_from_sample(cds_seq, cds_length:int, total_length: int = 50):
    from src.models.repaint.amino_codon_table import CODON_TO_AMINO, dna_to_rna
    name = cds_seq['name'] + f'_length_{cds_length}'
    pos_list = [pos for pos in range(total_length - cds_length, total_length, 3)]
    codon_truncated = dna_to_rna(cds_seq['codon'][:cds_length]) # replace T with U
    amino_list = [CODON_TO_AMINO.get(codon_truncated[i: i+3]) for i in range(0, len(codon_truncated), 3)] # we need a function to translate codons to amino based on codon_to_amino table
    pattern= {
        'name': name,
        'amino': amino_list,   # must in list data format [str]
        'pos': pos_list        # must in list data format [int]
    }
    return pattern

CDS_samples = [
    CDS_Sample_31,
    CDS_Sample_70,
    CDS_Sample_14,
    CDS_Sample_25,
    CDS_Sample_52
]

def build_codon_patterns_from_CDS_list(n):
    return [get_codon_pattern_from_sample(CDS, n) for CDS in CDS_samples]

def build_amino_patterns_from_CDS_list(n):
    return [get_amino_pattern_from_sample(CDS, n) for CDS in CDS_samples]


if __name__ == '__main__':
    patterns = build_codon_patterns_from_CDS_list(30)
    for pattern in patterns:
        print(pattern['name'], pattern['codon'], pattern['pos'])
