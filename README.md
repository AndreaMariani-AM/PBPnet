# PBPnet

[![PyPI Downloads](https://static.pepy.tech/badge/bpnet-lite)](https://pepy.tech/projects/bpnet-lite)

> IMPORTANT: PBPNet is a personal implementation of the original [bpnet-lite](https://github.com/jmschrei/bpnet-lite) by Jacob Schreiber, which in turn is a lightweigth implementation in PyTorch of the original [BPNet](https://github.com/kundajelab/bpnet). Where i've adapted code from others it's properly cited at the top of the corresponding file. PBPNet stands for Polycomb BPNet.

Polycomb BPNet (PBPNet) is a lightweight modification of [BPNet](https://github.com/kundajelab/bpnet). It builds on top of the original [bpnet-lite](https://github.com/jmschrei/bpnet-lite) adding features like a test and validation PyTorch DataLoader (adatped from [Adam He](https://github.com/adamyhe/PersonalBPNet)), concurrent model training and single task metrics. It also extends genomic context to 5kb to be able to model better Polycomb signal.

## Different strategy for data creation

Due to memory constrains on my current GPU server, i had to adapt a different strategy to create datasets. Those cannot be created as we go like the original implementation.  
I've created the script `create_datasets.py` that takes in a set of *peaks*, *signal tracks* and optionally *control tracks* and it extracts from the reference genome the underlying sequences and the corresponding base pair signal.  
These info are then stored into three different slots in a `hdf5` file. I've wrote the `DataLoader` expecting this format.