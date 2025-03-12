## CYP-MAP: Site-of-Metabolism Prediction Tool
### Overview
CYP-MAP is a graph-based deep learning tool for predicting sites of metabolism (SoM) by CYP enzymes. This tool analyzes the structure of drug molecules to identify locations with high potential for metabolism.
### Supported Environment

Operating Systems: Linux (Ubuntu 18.04 or higher recommended), macOS
Programming Language: Python 3.8 or higher
GPU Support: CUDA 11.8 or higher (multi-GPU support)

### Installation Guide
Required Dependencies
bashCopypip install -r requirements.txt
Essential packages:

PyTorch 2.0.1
RDKit 2022.03.2
NumPy 1.20.0+
torch_geometric 2.3.1
dgl 1.1.2+cu117

### Estimated Installation Time
Typical installation time: approximately 5-10 minutes (may vary depending on environment and internet speed)
Usage
1. Prepare Pretraining Data
- tar -zxvf pretrain/pretrain_data.tar.gz
- python pretrain/save_graph_pretrain.py
2. Graph Contrastive Learning
- CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --master_port 12312 pretrain/run_pretrain.py
3. Prepare Fine-tuning Graph
- python save_graph.py
4. Train Model
- python train.py --seed 42
5. Inference
- python -u infer.py --ckpt ckpt/42.pt --th 0.15
### Demo Execution
How to run the model with example data:
- python -u infer.py --demo --ckpt ckpt/42.pt --th 0.15

- Typical execution time: approximately 0.5 seconds per molecule (CPU), 0.1 seconds (GPU)
- Batch processing time for 1000 molecules: approximately 2 minutes (on GPU)

### Algorithm Description
- CYP-MAP uses molecular graph representation and graph neural networks to predict drug metabolism sites. Key features:

### Self-supervised graph contrastive learning
- Utilization of atomic and bond properties of molecular structures
- Prediction of metabolism sites for various CYP isoforms

## Example Dataset
- `data/` directory contains the following files:
  - `train_0819.sdf`: Training data for known CYP reactions
  - `train_decoy_0819.sdf`: Training data with molecules that have no known CYP reactions
  - `test_0819.sdf`: Specially curated test dataset for model evaluation
