## how to run

```
uv run main.py
```

It runs a benchmark of the core of the umap algorithm: attraction along edges using the edge weights, repulsion along randomly selected pairs of nodes, 5x as many repulsion pairs as there are edges.

By default it will run on GPU if it can and fall back to CPU otherwise. It will run for 5 epochs, 20 million nodes, and use atomic adds for adding up the position deltas for each node.

It generates fake data to run all this on.

More options:

```
uv run main.py --help
usage: main.py [-h] [--device {auto,mps,cuda,cpu}] [--num-nodes NUM_NODES] [--degree DEGREE] [--epochs EPOCHS] [--learning-rate LEARNING_RATE]
               [--negative-learning-rate NEGATIVE_LEARNING_RATE] [--num-negatives NUM_NEGATIVES] [--seed SEED] [--use-int64] [--use-segment-reduce]

UMAP-style graph optimization benchmark

options:
  -h, --help            show this help message and exit
  --device {auto,mps,cuda,cpu}
                        Device to use for computation (default: auto)
  --num-nodes NUM_NODES
                        Number of nodes in the graph (default: 20,000,000)
  --degree DEGREE       Average degree per node (default: 10)
  --epochs EPOCHS       Number of training epochs (default: 5)
  --learning-rate LEARNING_RATE
                        Learning rate for attractive forces (default: 0.001)
  --negative-learning-rate NEGATIVE_LEARNING_RATE
                        Learning rate for repulsive forces (default: 0.0001)
  --num-negatives NUM_NEGATIVES
                        Number of negative samples per epoch (default: 5)
  --seed SEED           Random seed for reproducibility (default: 42)
  --use-int64           Use int64 instead of int32 for edge indices (default: int32)
  --use-segment-reduce  Use segment_reduce instead of scatter_add (requires sorting)
```

## caveats

I used an LLM to help hack this together quickly, so there are probably bugs and mistakes in here.

I haven't tested `--device cuda` yet.

I haven't tested `--use-segment-reduce` yet. 

## example output on my laptop

I'm using a 2021 M1 MacBook Pro, 16GB unified memory, 16 GPU cores.

```
> uv run main.py
Random seed set to: 42
Using Apple Metal (MPS) device.
Generating realistic data for 20,000,000 nodes...
Using int32 for edge indices
Dynamically set NUM_EDGES to 100,000,000
Data generation complete.

--- Tensor Sizes ---
  node_positions: shape=[20000000, 2], size=152.59 MB
  edges: shape=[100000000, 2], size=762.94 MB
  edge_weights: shape=[100000000, 1], size=381.47 MB
  Total (base tensors): 1.27 GB

--- Verifying Node Degrees ---
Mean degree:   10.00
Median degree: 10.00
Variance of degree: 5.00
Min degree: 5.00
Max degree: 26.00

Starting benchmark for 5 epochs...
Each epoch: 1 attractive pass + 5 repulsive passes
Benchmark:   0%|                                                                                                                                                    | 0/5 [00:00<?, ?epoch/s]
--- Attractive Phase Tensor Sizes (first epoch) ---
  pos_u (gathered): shape=[100000000, 2], size=762.94 MB
  pos_v (gathered): shape=[100000000, 2], size=762.94 MB
  delta (computed): shape=[100000000, 2], size=762.94 MB
  Peak intermediate memory (attractive): 2.24 GB

--- Repulsive Phase ---
Sampling 100,000,000 random pairs per repulsive pass
Total repulsive samples per epoch: 500,000,000

Benchmark: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:25<00:00, 17.10s/epoch]

--- Benchmark Results ---
Total time for 5 epochs: 85.5202 seconds
Average time per epoch: 17.1040 seconds

--- Phase Breakdown ---
Attractive phase: 16.1531 seconds (18.89%)
Repulsive phase:  69.2937 seconds (81.03%)

--- Attractive Phase Operations ---
gather         : 4.8155 seconds (29.81%)
compute        : 1.5795 seconds (9.78%)
scatter_add    : 9.7581 seconds (60.41%)

--- Repulsive Phase Operations ---
sample         : 0.3497 seconds (0.50%)
gather         : 18.5202 seconds (26.73%)
compute        : 1.6466 seconds (2.38%)
scatter_add    : 48.7772 seconds (70.39%)

--- Result Checksum (for verification) ---
Sum:  19999044.0000000000
Mean: 0.4999760985
Std:  0.2872487903
Min:  -0.0125121158
Max:  1.0116773844

--- Estimated Time for 2000 Epochs ---
Estimated time: 9.50 hours
```
