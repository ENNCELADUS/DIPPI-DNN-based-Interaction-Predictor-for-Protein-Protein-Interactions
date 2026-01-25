#!/bin/bash
# =============================================================================
# PRING Data Processing Pipeline
# =============================================================================
# This script automates the entire PRING data preprocessing pipeline.
# Please ensure:
#   1. Raw data is downloaded from https://huggingface.co/datasets/piaolaidangqu/PRING
#   2. Raw data is placed in ./data/PRING/data_process/raw_data
#   3. MMseqs2 is installed (required for sequence-similarity filtering)
#   4. Conda environment with required dependencies is activated
# =============================================================================

set -e  # Exit on any error

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/PRING/data_process"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_step() {
    local step_name="$1"
    local script_path="$2"
    
    log_info "Running: $step_name"
    if python "$script_path"; then
        log_info "Completed: $step_name"
    else
        log_error "Failed: $step_name"
        exit 1
    fi
}

# =============================================================================
# Pre-flight checks
# =============================================================================
log_info "Starting PRING Data Processing Pipeline"
log_info "Project root: $PROJECT_ROOT"
log_info "Data directory: $DATA_DIR"

# Check if raw_data directory exists
if [ ! -d "$DATA_DIR/raw_data" ]; then
    log_error "Raw data directory not found: $DATA_DIR/raw_data"
    log_error "Please download raw data from https://huggingface.co/datasets/piaolaidangqu/PRING"
    exit 1
fi

# Check if MMseqs2 is installed (required for step 1.5)
if ! command -v mmseqs &> /dev/null; then
    log_warn "MMseqs2 not found. Step 1.5 (Sequence-similarity Filtering) may fail."
    log_warn "Please install MMseqs2 before running the pipeline."
fi

# =============================================================================
# Part 1: Protein Sequence Processing
# =============================================================================
log_info "=========================================="
log_info "Part 1: Protein Sequence Processing"
log_info "=========================================="

SEQ_PROCESS_DIR="$DATA_DIR/seq_process"
cd "$SEQ_PROCESS_DIR"
log_info "Changed to directory: $SEQ_PROCESS_DIR"

# Step 1.1: Get the UniProt ID
run_step "1.1 Get UniProt IDs" "uniprot_id.py"

# Step 1.2: Download Protein Sequences
run_step "1.2 Download Protein Sequences" "download_fasta.py"

# Step 1.3: Length Filtering (threshold: 50-1000)
run_step "1.3 Length Filtering" "seq_len_filter.py"

# Step 1.4: Separate Species
run_step "1.4 Separate Species" "seperate_species.py"

# Step 1.5: Sequence-similarity Filtering (threshold: 0.4)
run_step "1.5 Sequence-similarity Filtering" "seq_sim.py"

# Step 1.6: Similar Function Protein Filtering
run_step "1.6 Similar Function Protein Filtering" "similar_function_remove.py"

# Step 1.7: Organism Mapping
run_step "1.7 Organism Mapping" "organism_mapping.py"

# Step 1.8: Separate PPIs
run_step "1.8 Separate PPIs" "seperate_ppis.py"

# Step 1.9: Move Files
run_step "1.9 Move Files" "move_files.py"

# =============================================================================
# Part 2: Graph Generation
# =============================================================================
log_info "=========================================="
log_info "Part 2: Graph Generation"
log_info "=========================================="

GRAPH_GEN_DIR="$DATA_DIR/graph_gen"
cd "$GRAPH_GEN_DIR"
log_info "Changed to directory: $GRAPH_GEN_DIR"

# Step 2.1: Graph Construction
run_step "2.1 Graph Construction" "graph_cons.py"

# Step 2.2: Graph Split for HUMAN
run_step "2.2 Graph Split for HUMAN" "graph_split.py"

# Step 2.3: Sample Negative PPI Samples for HUMAN
run_step "2.3 Sample Negative PPI Samples (HUMAN)" "negative_sample.py"

# Step 2.4: Sample Negative PPI Samples for Other Species
run_step "2.4 Sample Negative PPI Samples (Other Species)" "otherspecies_negative_sample.py"

# Step 2.5: Simplify Fasta File
run_step "2.5 Simplify Fasta File" "fasta_simplify.py"

# Step 2.6: Generate All Against All Pairs
run_step "2.6 Generate All Against All Pairs" "all_against_all_pair.py"

# Step 2.7: Sample Subgraphs for Graph-level Testing
run_step "2.7 Sample Subgraphs" "graph_sample.py"

# Step 2.8: Rename Final Dataset Folder
log_info "Running: 2.8 Rename Final Dataset Folder"
cd "$DATA_DIR"
if [ -d "species_processed_data" ]; then
    mv species_processed_data pring_dataset
    log_info "Completed: Renamed species_processed_data to pring_dataset"
else
    log_warn "Directory 'species_processed_data' not found, skipping rename"
fi

# =============================================================================
# Pipeline Complete
# =============================================================================
log_info "=========================================="
log_info "PRING Data Processing Pipeline Complete!"
log_info "Output directory: $DATA_DIR/pring_dataset"
log_info "=========================================="
