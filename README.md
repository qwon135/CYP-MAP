# Site-of-metabolism

### Prepare pratrain graph
- python save_graph_pretrain.py

### Pretrain(Graph Contrastive Learning)
- CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --master_port 12312 run_pretrain.py

### Prepare finetune graph
- python save_graph.py

### Train

- python train.py --seed 42

### Inference

 - python -u infer.py --ckpt ckpt/42.pt --th 0.15
