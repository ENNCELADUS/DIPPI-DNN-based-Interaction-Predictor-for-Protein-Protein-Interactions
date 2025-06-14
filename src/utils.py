import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

def examine_dataframe(df):
    """Print the structure of the dataframe to identify column names"""
    print("DataFrame columns:", df.columns.tolist())
    print("First row sample:", df.iloc[0].to_dict())
    return df.columns.tolist()


def load_data():
    """Load the actual data from the project structure"""
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root (DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions)
    # From src/ go up one level to reach project root
    project_root = os.path.dirname(current_dir)
    
    print(f"Script directory: {current_dir}")
    print(f"Project root: {project_root}")
    print("Loading data...")
    
    # Construct full paths to data files
    data_dir = os.path.join(project_root, 'data')
    splits_dir = os.path.join(data_dir, 'splits')
    features_dir = os.path.join(data_dir, 'features')
    
    print(f"Looking for data in: {data_dir}")
    
    # Check if data directories exist
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load data files with full paths
    train_path = os.path.join(splits_dir, 'train_data.pkl')
    cv_path = os.path.join(splits_dir, 'validation_data.pkl')
    test1_path = os.path.join(splits_dir, 'test1_data.pkl')
    test2_path = os.path.join(splits_dir, 'test2_data.pkl')
    embeddings_path = os.path.join(features_dir, 'embeddings_standardized.pkl')
    
    # Check if all files exist
    for path, name in [(train_path, 'train_data.pkl'), 
                       (cv_path, 'validation_data.pkl'),
                       (test1_path, 'test1_data.pkl'), 
                       (test2_path, 'test2_data.pkl'),
                       (embeddings_path, 'embeddings_standardized.pkl')]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
        else:
            print(f"✅ Found: {name}")
    
    # Load the data
    train_data = pickle.load(open(train_path, 'rb'))
    cv_data = pickle.load(open(cv_path, 'rb'))
    test1_data = pickle.load(open(test1_path, 'rb'))
    test2_data = pickle.load(open(test2_path, 'rb'))

    # Examine structure of the first dataframe to understand its format
    print("\nExamining training data structure:")
    examine_dataframe(train_data)

    print("\nLoading protein embeddings...")
    protein_embeddings = pickle.load(open(embeddings_path, 'rb'))
    print(f"Loaded {len(protein_embeddings)} protein embeddings")

    print(train_data.head())
    for i, (key, value) in enumerate(protein_embeddings.items()):
        if i >= 5:
            break
        print(f"Protein ID: {key}, Embedding shape: {value.shape}")
    
    return train_data, cv_data, test1_data, test2_data, protein_embeddings


def load_data_v2():
    """Load data for v2 scripts from splits directory structure"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    print("Loading data...")
    
    # Look for data in the splits directory
    data_dir = os.path.join(project_root, 'data', 'splits')
    print(f"Looking for data in: {data_dir}")
    
    # Check if files exist
    files_to_check = [
        'train_data.pkl',
        'validation_data.pkl', 
        'test1_data.pkl',
        'test2_data.pkl',
        'embeddings_standardized.pkl'
    ]
    
    for file in files_to_check:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load datasets
    train_data = pd.read_pickle(os.path.join(data_dir, 'train_data.pkl'))
    val_data = pd.read_pickle(os.path.join(data_dir, 'validation_data.pkl'))
    test1_data = pd.read_pickle(os.path.join(data_dir, 'test1_data.pkl'))
    test2_data = pd.read_pickle(os.path.join(data_dir, 'test2_data.pkl'))
    
    # Load protein embeddings
    print("Loading protein embeddings...")
    protein_embeddings = pd.read_pickle(os.path.join(data_dir, 'embeddings_standardized.pkl'))
    print(f"Loaded {len(protein_embeddings)} protein embeddings")
    
    # Show some data structure info
    print("\nExamining training data structure:")
    print(f"DataFrame columns: {list(train_data.columns)}")
    if len(train_data) > 0:
        first_row = train_data.iloc[0].to_dict()
        print(f"First row sample: {first_row}")
    
    # Show embedding structure
    sample_proteins = list(protein_embeddings.keys())[:5]
    for protein_id in sample_proteins:
        emb_shape = protein_embeddings[protein_id].shape
        print(f"Protein ID: {protein_id}, Embedding shape: {emb_shape}")
    
    return train_data, val_data, test1_data, test2_data, protein_embeddings


def detect_column_names(data_df, embeddings_dict):
    """Automatically detect column names for protein IDs and interaction labels"""
    columns = data_df.columns.tolist()
    
    # Determine protein ID and interaction columns
    protein_a_col = None
    protein_b_col = None
    interaction_col = None
    
    # Common column name patterns
    protein_a_patterns = ['protein_a', 'protein_id_a', 'proteinA', 'proteinIDA', 'protein_A', 'protein_id_A']
    protein_b_patterns = ['protein_b', 'protein_id_b', 'proteinB', 'proteinIDB', 'protein_B', 'protein_id_B']
    interaction_patterns = ['isInteraction', 'is_interaction', 'interaction', 'label']
    
    # Find protein ID columns
    for col in columns:
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in protein_a_patterns):
            protein_a_col = col
        elif any(pattern.lower() in col_lower for pattern in protein_b_patterns):
            protein_b_col = col
        elif any(pattern.lower() in col_lower for pattern in interaction_patterns):
            interaction_col = col
    
    # If we still can't find the columns, look for any that might contain protein IDs
    if protein_a_col is None or protein_b_col is None:
        # Check the first row to see if any column contains values that match keys in embeddings_dict
        first_row = data_df.iloc[0].to_dict()
        for col, val in first_row.items():
            if isinstance(val, str) and val in embeddings_dict:
                if protein_a_col is None:
                    protein_a_col = col
                elif protein_b_col is None and col != protein_a_col:
                    protein_b_col = col
    
    if protein_a_col is None or protein_b_col is None or interaction_col is None:
        print("Column detection failed. Please specify column names manually.")
        print("Available columns:", columns)
        raise ValueError("Could not detect required columns")
    
    print(f"Using columns: Protein A = '{protein_a_col}', Protein B = '{protein_b_col}', Interaction = '{interaction_col}'")
    return protein_a_col, protein_b_col, interaction_col


class ProteinPairDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dict, protein_a_col=None, protein_b_col=None, interaction_col=None):
        """
        Dataset for protein pair interaction prediction
        
        Args:
            pairs_df: DataFrame with protein pair data
            embeddings_dict: Dict mapping uniprotID -> embedding tensor (seq_len, 960)
            protein_a_col: Column name for protein A IDs (auto-detected if None)
            protein_b_col: Column name for protein B IDs (auto-detected if None) 
            interaction_col: Column name for interaction labels (auto-detected if None)
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        
        # Auto-detect column names if not provided
        if protein_a_col is None or protein_b_col is None or interaction_col is None:
            self.protein_a_col, self.protein_b_col, self.interaction_col = detect_column_names(pairs_df, embeddings_dict)
        else:
            self.protein_a_col = protein_a_col
            self.protein_b_col = protein_b_col
            self.interaction_col = interaction_col
        
        # Filter valid pairs
        valid_indices = []
        for idx in range(len(self.pairs_df)):
            row = self.pairs_df.iloc[idx]
            if (row[self.protein_a_col] in self.embeddings_dict and 
                row[self.protein_b_col] in self.embeddings_dict):
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        print(f"Dataset: {len(valid_indices)} valid pairs out of {len(self.pairs_df)} total pairs")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        row = self.pairs_df.iloc[data_idx]
        
        # Get embeddings
        emb_a = self.embeddings_dict[row[self.protein_a_col]]
        emb_b = self.embeddings_dict[row[self.protein_b_col]]
        
        # Convert to tensors if needed
        if not isinstance(emb_a, torch.Tensor):
            emb_a = torch.from_numpy(emb_a).float()
        if not isinstance(emb_b, torch.Tensor):
            emb_b = torch.from_numpy(emb_b).float()
        
        # Get interaction label
        interaction = int(row[self.interaction_col])
        
        return {
            'emb_a': emb_a,           # (seq_len_a, 960)
            'emb_b': emb_b,           # (seq_len_b, 960)
            'interaction': interaction,
            'id_a': row[self.protein_a_col],
            'id_b': row[self.protein_b_col]
        }


def collate_fn(batch):
    """
    Collate function for protein pair batches with padding
    """
    # Extract components
    embs_a = [item['emb_a'] for item in batch]
    embs_b = [item['emb_b'] for item in batch]
    interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    max_len_a = max(emb.shape[0] for emb in embs_a)
    max_len_b = max(emb.shape[0] for emb in embs_b)
    
    # Create padded tensors and length masks
    batch_size = len(batch)
    padded_a = torch.zeros(batch_size, max_len_a, 960)
    padded_b = torch.zeros(batch_size, max_len_b, 960)
    lengths_a = torch.zeros(batch_size, dtype=torch.long)
    lengths_b = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
        len_a, len_b = emb_a.shape[0], emb_b.shape[0]
        padded_a[i, :len_a] = emb_a
        padded_b[i, :len_b] = emb_b
        lengths_a[i] = len_a
        lengths_b[i] = len_b
    
    return padded_a, padded_b, lengths_a, lengths_b, interactions


class ProteinPairDatasetV2(Dataset):
    """Dataset for protein pairs with embeddings (v2 compatible)"""
    def __init__(self, data, protein_embeddings):
        # Filter data to only include pairs where both proteins have embeddings
        valid_pairs = []
        for _, row in data.iterrows():
            protein_a = row['uniprotID_A']
            protein_b = row['uniprotID_B']
            if protein_a in protein_embeddings and protein_b in protein_embeddings:
                valid_pairs.append(row)
        
        self.data = pd.DataFrame(valid_pairs)
        self.protein_embeddings = protein_embeddings
        print(f"Dataset: {len(self.data)} valid pairs out of {len(data)} total pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        protein_a = row['uniprotID_A']
        protein_b = row['uniprotID_B']
        interaction = row['isInteraction']
        
        # Get embeddings
        emb_a = torch.tensor(self.protein_embeddings[protein_a], dtype=torch.float32)
        emb_b = torch.tensor(self.protein_embeddings[protein_b], dtype=torch.float32)
        
        return {
            'emb_a': emb_a,
            'emb_b': emb_b,
            'interaction': float(interaction)
        }


def collate_fn_v52(batch):
    """
    Collate function for v5.2 PPI classifier
    Returns embeddings with length information for v2 MAE compatibility
    """
    # Extract components
    embs_a = [item['emb_a'] for item in batch]
    embs_b = [item['emb_b'] for item in batch]
    interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.float)
    
    # Get original lengths before padding
    lengths_a = torch.tensor([emb.shape[0] for emb in embs_a])
    lengths_b = torch.tensor([emb.shape[0] for emb in embs_b])
    
    # Pad sequences to same length within batch
    max_len_a = max(emb.shape[0] for emb in embs_a)
    max_len_b = max(emb.shape[0] for emb in embs_b)
    
    # Create padded tensors
    batch_size = len(batch)
    padded_a = torch.zeros(batch_size, max_len_a, 960)
    padded_b = torch.zeros(batch_size, max_len_b, 960)
    
    for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
        len_a, len_b = emb_a.shape[0], emb_b.shape[0]
        padded_a[i, :len_a] = emb_a
        padded_b[i, :len_b] = emb_b
    
    return padded_a, padded_b, lengths_a, lengths_b, interactions