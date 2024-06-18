# Site-of-metabolism

### Prepare graph
- python save_graph.py

### Train

- python train.py --seed 42 --save_name 'cyp_som'

### Inference

 - python -u infer.py --ckpt ckpt/0.pt --add_H 1 --th 0.1
