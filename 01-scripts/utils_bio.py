from dnachisel import *
from Bio.Seq import Seq
from Bio import SeqIO
from ete3 import NCBITaxa
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict
import os
import math
from collections import Counter

ncbi = NCBITaxa()

def csv_to_fasta(csv_file,out_dir):
    organism = csv_file.split('/')[-1].split('.')[0]
    csv_df = pd.read_csv(csv_file).drop_duplicates()
    with open(f'{out_dir}/{organism}.faa', 'w') as fasta_out:
        for idx, row in csv_df.iterrows():
            sequence_id, sequence = row[0], row[1]
            fasta_out.write(f">{organism}-{sequence_id}\n{sequence}\n")

def dict_to_fasta(faa_dict,out_file):
    with open(out_file,'w') as out_file:
        for id, seq in faa_dict.items():
            out_file.write(f'>{id}\n{seq}\n')

def fasta_to_df(fasta):
    fasta_dict = {}
    fasta_sequences = SeqIO.parse(open(fasta),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        fasta_dict[name] = sequence

    df = pd.DataFrame([fasta_dict]).T.reset_index()
    df.columns = ['tile_id','tile_seq']
    return df

def df_to_fasta(df,id_col,seq_col,out_file):
    with open(out_file,'w') as output_file:
        for _, row in df.iterrows():
            output_file.write(f'>{row[id_col]}\n{row[seq_col]}\n')
    return None

def chunks(data,size=10_000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in slice(it, size)}

def create_protein_tiles(protein_sequence,tile_size=40,overlap_size=10):
    tiles = []
    sequence_length = len(protein_sequence)
    for i in range(0, sequence_length - tile_size + 1, tile_size - overlap_size):
        tile = protein_sequence[i:i + tile_size]
        tiles.append(tile)
    # tiles.append(protein_sequence[-tile_size:])
    return tiles

def is_protein_sequence(sequence):
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.upper()
    return all(aa in valid_amino_acids for aa in sequence)

def slice_dictionary(original_dict, num_slices):
    sliced_dicts = []
    slice_size = len(original_dict) // num_slices
    extra_items = len(original_dict) % num_slices
    iter_dict = iter(original_dict.items())
    for _ in tqdm(range(num_slices),total=num_slices):
        current_slice_size = slice_size + (1 if extra_items > 0 else 0)
        extra_items -= 1
        sliced_dict = dict([next(iter_dict) for _ in range(current_slice_size)])
        sliced_dicts.append(sliced_dict)
    return sliced_dicts

def protein_to_dna(protein_sequence):
    codon_table = {
        'A': 'GCT', 
        'R': 'CGT', 
        'N': 'AAT', 
        'D': 'GAT', 
        'C': 'TGT',
        'Q': 'CAA', 
        'E': 'GAA', 
        'G': 'GGT', 
        'H': 'CAT', 
        'I': 'ATT',
        'L': 'TTA', 
        'K': 'AAA', 
        'M': 'ATG', 
        'F': 'TTT', 
        'P': 'CCT',
        'S': 'TCT', 
        'T': 'ACT', 
        'W': 'TGG', 
        'Y': 'TAT', 
        'V': 'GTT',
        '*': 'TAA'
    }
    dna_sequence = ''
    for amino_acid in protein_sequence:
        dna_sequence += codon_table.get(amino_acid, '')
    return dna_sequence

def codon_optimize(seq):
    problem = DnaOptimizationProblem(
        sequence=seq,
        constraints=[
            AvoidPattern(str(12) + "xA"),
            AvoidPattern(str(12) + "xT"),
            AvoidPattern(str(6) + "xC"),
            AvoidPattern(str(6) + "xG"),
            AvoidHairpins(stem_size=10, hairpin_window=200),
            EnforceGCContent(mini=0.35, maxi=0.75, window=200),
            EnforceTranslation(),
        ],
        objectives=[CodonOptimize(species='s_cerevisiae',method='use_best_codon')]
    )
    problem.max_random_iters = 50_000
    problem.resolve_constraints()
    problem.optimize()
    final_sequence = problem.sequence
    return final_sequence

def max_sequential_same_letter_count(seq):
    if not seq:
        return 0
    max_count = 1
    current_count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    return max_count

def parse_sort_results(sort_data, out_name, fig_out_dir='.', results_out_dir='.'):
    activities, total_reads, reads_per_bin, loss_table = sort_data
    activities_df = activities.reset_index().rename(columns={'level_0': 'DNAseq', 'level_1': 'BC'})
    reads_per_bin_df = (reads_per_bin.reset_index().rename(columns={'level_0': 'DNAseq', 'level_1': 'BC'}))
                    
    ad_count_df = reads_per_bin_df.drop(columns='BC').groupby('DNAseq').sum().sum(axis=1).reset_index().rename(columns={0:'num_AD_reads'})
    bc_count_df = reads_per_bin_df.drop(columns='DNAseq').groupby('BC').sum().sum(axis=1).reset_index().rename(columns={0:'num_BC_reads'})
    total_reads_df = total_reads.reset_index().rename(columns={'level_0': 'DNAseq', 'level_1': 'BC',0:'num_reads_per_BC'})

    total_sum = sum([total_reads_df['num_reads_per_BC'].sum(), loss_table['ad_preceder'], loss_table['design_file'], loss_table['thresh']])
    missing_barcodes_df = total_reads_df[total_reads_df['BC'].isna()]

    tmp_df = total_reads_df.groupby(['BC']).nunique()
    tmp_df = tmp_df[tmp_df['DNAseq'] > 1]

    merged_df = activities_df.merge(total_reads_df.drop(columns='BC').rename(columns={'DNAseq':'AD'}), left_index=True, right_index=True)
    assert merged_df['DNAseq'].equals(merged_df['AD'])

    merged_df = merged_df.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7, 'AD'])
    reads_per_bin_df = reads_per_bin.reset_index().drop(columns='level_1').rename(columns={'level_0':'DNAseq'})
    merged_df2 = merged_df.merge(reads_per_bin_df, on='DNAseq')
    merged_df2['AAseq'] = merged_df2['DNAseq'].apply(lambda x: str(Seq(x).translate()))
    merged_df2 = merged_df2.reindex(columns=['DNAseq', 'AAseq', 'numreads_per_AD', 0, 1, 2, 3, 4, 5, 6, 7, 'Activity'])

    print('Total reads with missing barcodes: ', missing_barcodes_df['num_reads_per_BC'].sum())
    print('Total reads with multiple ADs per BC: ', total_reads_df[total_reads_df['BC'].isin(tmp_df.index)]['num_reads_per_BC'].sum())
    print('sum included reads: ', total_reads_df['num_reads_per_BC'].sum())
    print('% included reads: ', 100*ad_count_df['num_AD_reads'].sum() / total_sum)
    print('% reads lost ad_preceder: ', 100*loss_table['ad_preceder'] / total_sum)
    print('% reads lost design_file: ', 100*loss_table['design_file'] / total_sum)
    print('% reads lost thresh: ', 100*loss_table['thresh'] / total_sum)

    activities_df.to_csv(f"{results_out_dir}/{out_name}_activities.csv", index=True)
    reads_per_bin_df.to_csv(f"{results_out_dir}/{out_name}_readsperbin.csv", index=True)
    total_reads_df.to_csv(f"{results_out_dir}/{out_name}_totalreads.csv", index=True)
    pd.DataFrame([loss_table]).T.to_csv(f'{results_out_dir}/{out_name}_loss.csv')
    merged_df2.to_csv(f'{results_out_dir}/{out_name}_data.csv', index=True)

    fig_params = dict(bbox_inches='tight',transparent=False,dpi=400)
    plt.figure(figsize=(4,4))
    g = sns.histplot(data=activities, x='Activity', bins=50, kde=False,element='step',edgecolor=None)
    g.set(xlabel='Activity Score',ylabel='Frequency',title='Histogram of Activity Scores')
    plt.savefig(f'{fig_out_dir}-{out_name}_ActivityScoreDist.png',**fig_params)

    plt.figure(figsize=(4,4))
    g = sns.histplot(data=bc_count_df, x='num_BC_reads', bins=400, log_scale=(True, True),element='step',edgecolor=None)
    g.set(xlabel='Reads Per Barcode',ylabel='Frequency',title='Included AD:BC Reads')
    plt.savefig(f'{fig_out_dir}-{out_name}_AD-BC_ReadsDist.png',**fig_params)

    plt.figure(figsize=(4,4))
    g = sns.histplot(data=ad_count_df, x='num_AD_reads', bins=400, log_scale=(True, True),edgecolor=None)
    g.set(xlabel='Reads Per AD',ylabel='Frequency',title='Included AD Reads')
    plt.savefig(f'{fig_out_dir}-{out_name}_AD_ReadsDist.png',**fig_params)

    plt.figure(figsize=(4,4))
    g = sns.histplot(total_reads_df.groupby('DNAseq')['BC'].nunique(), bins=35, log_scale=(False, True), element='step',edgecolor=None)
    g.set(xlabel='Barcodes per AD',ylabel='Frequency',title='Barcodes per AD for Included Reads')
    plt.savefig(f'{fig_out_dir}-{out_name}_BCperAD_ReadsDist.png')
    return None


def split_cog_categories(df, method='primary'):
    category_counts = defaultdict(float)

    for _, row in df.iterrows():
        cats = list(row['COG_category'])
        count = row['count']
        
        if method == 'primary':
            category_counts[cats[0]] += count
        elif method == 'fractional':
            for cat in cats:
                category_counts[cat] += count / len(cats)
        else:
            raise ValueError("Method must be 'primary', or 'fractional'")
    
    return dict(category_counts)

def get_taxonomy_data(taxon_id,level='phylum'):
    try:
        lineage = ncbi.get_lineage(taxon_id)
    except:
        return taxon_id

    names = ncbi.get_taxid_translator(lineage)
    ranks = ncbi.get_rank(lineage)

    for tax_id in lineage:
        if ranks[tax_id] == level:
            return names[tax_id]
    return taxon_id

def count_fasta_sequences(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                count += 1
    return count

def max_overlap(seq1, seq2):
    max_len = 0
    for i in range(len(seq1)):
        if seq2.startswith(seq1[i:]):
            max_len = len(seq1) - i
            break
    return max_len

def compute_overlap_matrix(sequences):
    n = len(sequences)
    overlap_matrix = [[0] * n for _ in range(n)]
    
    for i, seq1 in tqdm(enumerate(sequences),total=len(sequences)):
        for j, seq2 in enumerate(sequences):
            if i != j: # diagonal
                overlap_matrix[i][j] = max_overlap(seq1, seq2)
    
    return overlap_matrix

def calculate_positional_entropy(sequences):
    sequence_length = len(sequences[0])
    total_entropy = 0.0
    for position in range(sequence_length):
        column = [seq[position] for seq in sequences]
        counts = Counter(column)
        total = len(column)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        total_entropy += entropy

    return total_entropy

def calculate_entropy(sequence):
    freq = Counter(sequence)
    total_amino_acids = len(sequence)
    entropy = 0
    for count in freq.values():
        probability = count / total_amino_acids
        entropy -= probability * math.log(probability, 2)
    return entropy

def extract_promoter(in_fasta,out_fasta):
    awk_cmd = r"""
        awk '/>/{id=$0; next} /ACTTCATCGGCTAGC/ {
        gsub(/^[[:blank:]]+|[[:blank:]]+$/, "", $0);
        substring = substr($0, index($0, "ACTTCATCGGCTAGC")+15,159);
        if (length(substring) == 159) {
            print id;
            print substring;
        }
    }' %s > %s
    """ % (in_fasta, out_fasta)
    os.system(awk_cmd)
    return None

def extract_barcode(in_fasta,out_fasta):
    awk_cmd = r"""
        awk '/>/{id=$0; next} /TGATAACTAGCTGAGGGCCCG/ {
        gsub(/^[[:blank:]]+|[[:blank:]]+$/, "", $0);
        substring = substr($0, index($0, "TGATAACTAGCTGAGGGCCCG")+21,14);
        if (length(substring) == 14) {
            print id;
            print substring;
        }
    }' %s > %s
    """ % (in_fasta, out_fasta)
    os.system(awk_cmd)
    return None

def fasta_to_df(in_fasta):
    fasta_sequences = SeqIO.parse(open(in_fasta),'fasta')
    fasta_dict = {}
    for fasta in fasta_sequences:
        fasta_dict[fasta.id] = str(fasta.seq)
    df = pd.DataFrame([fasta_dict]).T.reset_index()
    df.columns = ['ID','seq']
    return df

def combine_fasta(fasta1, fasta2):
    sequences1 = {record.id: record for record in SeqIO.parse(fasta1, "fasta")}
    combined_sequences = {}
    for record in SeqIO.parse(fasta2, 'fasta'):
        id = record.id
        if id in sequences1:
            combined_sequences[id] = [str(sequences1[id].seq),str(record.seq)]
    df = pd.DataFrame(combined_sequences).T.reset_index()
    df.columns = ['ID','tile','barcode']
    return df

def map_tile_id_to_seq(df,
                       id_col):
    tile_ids = set(df[id_col].unique())
    faa_dict = {}
    for record in tqdm(SeqIO.parse('../../02-OUTPUT/09-tile_mycocosm/03-clustered_90/deduplicated_fungi_tiles_90.faa', "fasta"),total=72_057_539):
        if record.id in tile_ids:
            faa_dict[record.id] = str(record.seq)
    return faa_dict

def run_diamond(faa_file,out_file,diamond_db):
    os.system(f'diamond blastp -d {diamond_db} -q {faa_file} -o {out_file} --id 100')

def map_protein_id_to_seq(protein_ids,organism_ids,faa_dir):
    faa_dict = {}
    for id in tqdm(organism_ids):
        path = f'{faa_dir}/{id}.faa'
        try:
            assert os.path.isfile(path)
        except:
            print(path)
        for record in SeqIO.parse(path, "fasta"):
            if record.id in protein_ids:
                faa_dict[record.id] = str(record.seq)


    return faa_dict

def lookup_tiles(df,seq_col,tile_faa,dmnd_file,dmnd_db,db_dir,protein_faa):
    tile_dict = map_tile_id_to_seq(df,seq_col)
    dict_to_fasta(tile_dict,tile_faa)
    run_diamond(tile_faa,dmnd_file,dmnd_db)
    dmnd_file = 'paddle_wrong_tiles.tsv'
    dmnd_df = pd.read_csv(dmnd_file,sep='\t',header=None)
    dmnd_df['organism'] = dmnd_df[1].apply(lambda x: x.split('-')[0])
    organisms = set(dmnd_df['organism'].unique())
    protein_ids = set(dmnd_df[1])
    protein_dict = map_protein_id_to_seq(protein_ids,organisms,db_dir)
    dict_to_fasta(protein_dict,protein_faa)