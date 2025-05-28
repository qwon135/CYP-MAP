## CYP-MAP: Multi-Level GNN Model for Site of Metabolism Prediction
![back ground](https://github.com/user-attachments/assets/38367d5d-f568-40f8-891a-0b942973a3e8)

### Overview
CYP-MAP is a multi-level graph neural network (GNN) model designed to predict sites of metabolism (SoM) by Cytochrome P450 (CYP) enzymes, which play a key role in drug metabolism. By analyzing the molecular structure of drug candidates, CYP-MAP identifies potential metabolic sites, reaction types, and CYP450 subtypes. Developed using experimentally validated metabolic data from DrugBank and other sources, the model integrates global, bond, and atom-level representations to capture both local reactivity and broader molecular properties. This predictive capability supports drug optimization for improved efficacy and safety.

## Supported Environment
- **Operating Systems**: Linux (Ubuntu 18.04 or higher recommended), macOS
- **Programming Language**: Python 3.8 or higher
- **GPU Support**: CUDA 11.8 or higher (multi-GPU support)

## Installation Guide
Required Dependencies
```bash
pip install -r requirements.txt
```

Essential packages:
- PyTorch 2.0.1
- RDKit 2022.03.2
- NumPy 1.20.0+
- torch_geometric 2.3.1
- dgl 1.1.2+cu117

## Estimated Installation Time
Typical installation time: approximately 5-10 minutes (may vary depending on environment and internet speed)

## Usage

1. Pretrain: Graph Contrastive Learning

   Option A: Multi-GPU (Distributed Data Parallel)
   If you have multiple GPUs (e.g., 4), run the following command for distributed pretraining:

   ```bash
   cd pretrain   
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --master_port 12312 run_pretrain.py
   ```

   Option B: Single GPU
   If you have only one GPU, run the following simpler command:

   ```bash
   cd pretrain   
   python run_pretrain.py
   ```

   Skip Pretraining (Optional)
   A pretrained GNN model is already provided at:

   ```bash
   pretrain/ckpt_pretrain/gnn_pretrain.pt

   ```

3. Prepare Fine-tuning Graph
   ```bash
   python save_graph.py
   ```

4. Train Model
   ```bash
   python train.py --seed 42
   ```

5. Inference
   ```bash
   python -u infer.py --ckpt ckpt/42.pt --th 0.15
   ```

## Demo Execution
How to run the model with example data:
```bash
python -u infer.py --demo --ckpt ckpt/42.pt --th 0.15
```
- Typical execution time: approximately 0.5 seconds per molecule (CPU), 0.1 seconds (GPU)
- Batch processing time for 1000 molecules: approximately 2 minutes (on GPU)

## Command Line Usage

### When the input is SMILES:
```bash
python output_module --smiles 'CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)N' --subtype sub9 --base_dir "./output_dir/"
```

### When the input is Structural Data File (SDF):
```bash
python output_module --sdf "./example_molecules.sdf" --subtype sub9 --base_dir "./output_dir/"
python output_module --sdf "./example_molecules.sdf" --subtype sub9 --mode broad --base_dir "./output_dir/"
```

### Input Type Options:
- `--smiles`: SMILES string of the molecule
- `--sdf`: Path to the SDF file

### CYP450 Mode (--subtype):
- `--subtype`: Subtype to predict. Options: 'sub9', 'all'
  - **all (All)**: A model trained on molecules metabolized by all 9 CYP450 subtypes without distinction
  - **sub9 (9 Subtypes)**: A model trained to distinguish between the 9 CYP450 subtypes, providing detailed predictions for each subtype

### Prediction Window (--mode):
- `--mode`: Prediction mode. Options: 'default', 'broad'
  - **Default**: An optimized option without coverage constraints
  - **Broad**: An option optimized under the condition of 80% SOM (Site of Metabolism) coverage

### Output Type:
- `--output_type`: Type of output to generate (e.g., default, only-som, json)

### Output Path:
- `--base_dir`: Base directory to save outputs

### Algorithm Description
- CYP-MAP uses molecular graph representation and graph neural networks to predict drug metabolism sites. Key features:

### Self-supervised graph contrastive learning
- Utilization of atomic and bond properties of molecular structures
- Prediction of metabolism sites for various CYP isoforms

## Example Dataset
- `data/` directory contains the following files:
  - `cyp_map_train.sdf`: Training data for known CYP reactions
  - `Decoys_cypreact_Drug_like.sdf`: Training data with molecules that have no known CYP reactions
  - `cyp_map_test.sdf`: Specially curated test dataset for model evaluation
  - `cyp_map_train_with_decoy.sdf`: Combined dataset containing both CYP reactions (from cyp_map_train) and decoy molecules
