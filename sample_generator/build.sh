#!/bin/bash

# Build in release mode
cargo build --release

# Run the sample generator
./target/release/sample_generator \
    --input ../sentences_wikipedia.txt \
    --output ../data/samples.txt \
    --selection-prob 0.2
