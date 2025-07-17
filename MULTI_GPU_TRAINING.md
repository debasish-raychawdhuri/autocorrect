# Multi-GPU Training for Autocorrect Model

This document explains how to use multiple GPUs for training the autocorrect model using `run_reu.py`.

## Options for Multi-GPU Training

There are two main approaches for multi-GPU training:

1. **DataParallel** - Simpler but less efficient
2. **DistributedDataParallel (DDP)** - More efficient but requires more setup

## Using DataParallel (Simplest)

DataParallel splits batches across multiple GPUs on a single machine:

```bash
python run_reu.py --train --multi_gpu --word2vec=path/to/word2vec.bin --data=path/to/data.json
```

This will automatically use all available GPUs on your system.

## Using DistributedDataParallel (Recommended)

For better performance, use DistributedDataParallel:

### Option 1: Using the helper script

The simplest way to launch distributed training is with the provided helper script:

```bash
./run_distributed.py --word2vec=path/to/word2vec.bin --data=path/to/data.json --epochs=3 --batch_size=32
```

This will automatically launch one process per GPU.

### Option 2: Manual launch with torch.distributed.launch

You can also manually launch distributed training:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS run_reu.py --train --distributed --word2vec=path/to/word2vec.bin --data=path/to/data.json
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Performance Tips

1. **Batch Size**: When using multiple GPUs, the effective batch size is multiplied by the number of GPUs. You may want to adjust the learning rate accordingly.

2. **Memory Usage**: Monitor GPU memory usage with `nvidia-smi`. If you're running out of memory, reduce the batch size.

3. **CPU Bottlenecks**: If data loading becomes a bottleneck, increase the `num_workers` parameter in the DataLoader.

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or model size
- **Process Hanging**: Check if all GPUs are properly initialized
- **Different Results Between Runs**: Set a fixed random seed for reproducibility

## Example Commands

Train on all available GPUs using DataParallel:
```bash
python run_reu.py --train --multi_gpu --word2vec=GoogleNews-vectors-negative300.bin --data=autogen_char_data.json --epochs=5 --batch_size=64
```

Train on 2 GPUs using DistributedDataParallel:
```bash
./run_distributed.py --nproc_per_node=2 --word2vec=GoogleNews-vectors-negative300.bin --data=autogen_char_data.json --epochs=5 --batch_size=64
```
