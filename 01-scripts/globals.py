import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

DataSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]
SplitResult = Tuple[Tuple[DataSplit, DataSplit, DataSplit], DataSplit]

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.5
sns.set_context('paper', rc={'font.size':8,
                             'axes.titlesize':8,
                             'axes.labelsize':8,
                             'xtick.labelsize':6,
                             'ytick.labelsize':6,
                             'xtick.major.width':0.5,
                             'ytick.major.width':0.5,
                             'axes.linewidth':0.5})   


RANDOM_STATES = list(range(1,4))
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_I = {aa:i for i, aa in enumerate(AMINO_ACIDS)}
I_TO_AA = {i:aa for i, aa in enumerate(AMINO_ACIDS)}
COLORS = list(sns.color_palette('tab20'))
COLOR_PAIRS = [(i, i + 1) for i in range(2, len(COLORS), 2)]
DATASET_COLORS = {'mycocosm': '#69764d',
                  'hummel': '#8c564b',
                  'sanborn': '#d89b47',
                  'morffy': '#ee82ee'}
ESM_MODELS = {'esm1v_t33_650M_UR90S_1': 33,
              'esm1v_t33_650M_UR90S_2': 33,
              'esm1v_t33_650M_UR90S_3': 33,
              'esm1v_t33_650M_UR90S_4': 33,
              'esm1v_t33_650M_UR90S_5': 33,
              'esm2_t12_35M_UR50D': 12,
              'esm2_t30_150M_UR50D': 30,
              'esm2_t33_650M_UR50D': 33,
              'esm2_t36_3B_UR50D': 36,
              'esm2_t48_15B_UR50D': 48,
              'esm2_t6_8M_UR50D': 6}
FIGURE_PARAMS = dict(dpi=600,transparent=False,bbox_inches='tight')
    
COG_MAP = {'A': 'RNA processing and modification',
           'B': 'Chromatin structure and dynamics',
           'C': 'Energy production and conversion',
           'D': 'Cell cycle control, cell division, chromosome partitioning',
           'E': 'Amino acid transport and metabolism',
           'F': 'Nucleotide transport and metabolism',
           'G': 'Carbohydrate transport and metabolism',
           'H': 'Coenzyme transport and metabolism',
           'I': 'Lipid transport and metabolism',
           'J': 'Translation, ribosomal structure and biogenesis',
           'K': 'Transcription',
           'L': 'Replication, recombination and repair',
           'M': 'Cell wall/membrane/envelope biogenesis',
           'N': 'Cell motility',
           'O': 'Posttranslational modification, protein turnover, chaperones',
           'P': 'Inorganic ion transport and metabolism',
           'Q': 'Secondary metabolites biosynthesis, transport, and catabolism',
           'R': 'General function prediction only',
           'S': 'Function unknown',
           'T': 'Signal transduction mechanisms',
           'U': 'Intracellular trafficking and secretion',
           'V': 'Defense mechanisms',
           'W': 'Extracellular structures',
           'Y': 'Nuclear structure',
           'Z': 'Cytoskeleton'}

COG_GROUPED_MAP = {
    # Information storage and processing
    'J': 'Information processing',  # Translation, ribosomal structure and biogenesis
    'A': 'Information processing',     # RNA processing and modification
    'K': 'Information processing',  # Transcription
    'L': 'Information processing',  # Replication, recombination and repair
    'B': 'Information processing',     # Chromatin structure and dynamics

    # Cellular processes
    'D': 'Cellular processes',      # Cell cycle control, cell division, chromosome partitioning
    'Y': 'Cellular processes',      # Nuclear structure
    'V': 'Cellular processes',      # Defense mechanisms
    'T': 'Cellular processes',      # Signal transduction mechanisms
    'M': 'Cellular processes',      # Cell wall/membrane/envelope biogenesis
    'N': 'Cellular processes',      # Cell motility
    'Z': 'Cellular processes',      # Cytoskeleton

    # Metabolism
    'C': 'Metabolism',              # Energy production and conversion
    'G': 'Metabolism',              # Carbohydrate transport and metabolism
    'E': 'Metabolism',              # Amino acid transport and metabolism
    'F': 'Metabolism',              # Nucleotide transport and metabolism
    'H': 'Metabolism',              # Coenzyme transport and metabolism
    'I': 'Metabolism',              # Lipid transport and metabolism
    'P': 'Metabolism',              # Inorganic ion transport and metabolism
    'Q': 'Metabolism',              # Secondary metabolites biosynthesis, transport and catabolism

    # Poorly characterized
    'R': 'Poorly characterized',    # General function prediction only
    'S': 'Poorly characterized'     # Function unknown
}

IN_DIR = '../../01-INPUT'
METADATA_DIR = f'{IN_DIR}/01-metadata'
DATASET_DIR = f'{IN_DIR}/02-datasets'
SEQUENCING_DIR = f'{IN_DIR}/03-sequencing'

OUT_DIR = '../../02-OUTPUT'
TARGET_DIR = f'{OUT_DIR}/01-evaluate_targets'
ENCODING_DIR = f'{OUT_DIR}/02-evaluate_encodings'
EMBEDDING_DIR = f'{OUT_DIR}/03-evaluate_embeddings'
SIMPLE_ARCHITECTURE_DIR = f'{OUT_DIR}/04-evaluate_simple_architectures'
COMPLEX_ARCHITECTURE_DIR = f'{OUT_DIR}/05-evaluate_complex_architectures'
PARAMETER_DIR = f'{OUT_DIR}/06-optimize_parameters'
HAMMING_ENSEMBLE_DIR = f'{OUT_DIR}/07-ensemble_models_hamming'
INITIAL_GENERALIZABILITY_DIR = f'{OUT_DIR}/08-evaluate_initial_generalizability'
MYCOCOSM_DIR = f'{OUT_DIR}/09-tile_mycocosm'
LIBRARY_DIR = f'{OUT_DIR}/10-design_library'
BARCODE_DIR = f'{OUT_DIR}/11-analyze_barcodes'
SORTSEQ_DIR = f'{OUT_DIR}/12-analyze_sortseq'
HARMONIZE_DIR = f'{OUT_DIR}/13-harmonize_datasets'
UNCERTAINTY_DIR = f'{OUT_DIR}/14-evaluate_uncertainty'
INTERPRETABILITY_DIR = f'{OUT_DIR}/15-evaluate_updated_interpretability'
GENERALIZABILITY_DIR = f'{OUT_DIR}/16-evaluate_updated_generalizability'
UPDATED_DIR = f'{OUT_DIR}/17-analyze_updated_mycocosm'
EXPLORE_DIR = f'{OUT_DIR}/18-explore_sequence_space'
COMPARE_DIR = f'{OUT_DIR}/19-compare_datasets'
SANBORN_DIR = f'{OUT_DIR}/20-evaluate_dataset_sanborn'
MORFFY_DIR = f'{OUT_DIR}/21-evaluate_dataset_morffy'
ADHUNTER_DIR = f'{OUT_DIR}/22-compare_model_ADhunter'
PADDLE_DIR = f'{OUT_DIR}/23-compare_model_PADDLE'
TADA_DIR = f'{OUT_DIR}/24-compare_model_TADA'
SIMPLE_DIR = f'{OUT_DIR}/25-compare_model_simple'
SANDBOX_DIR = f'{OUT_DIR}/xx-sandbox'

FIGURE_DIR = '../../03-FIGURES'

BATCH_SIZE = 128
HIDDEN_SIZE = 64
KERNEL_SIZE = 5
DILATION = 3
MAX_EPOCHS = 100
NUM_RES_BLOCKS = 3
PATIENCE = 5
WEIGHT_DECAY = 1e-2
LEARNING_RATE = 1e-3
SEQ_LEN = 53

SAVE_FIGURES = True