#!/usr/bin/env python3
"""
Analyze ONNX file structure to understand parameter naming and shapes
"""

import sys
import onnx
import numpy as np

def analyze_onnx_file(onnx_path):
    """Thoroughly analyze an ONNX file structure"""
    print(f"Analyzing ONNX file: {onnx_path}")
    print("=" * 80)
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    
    # Basic model info
    print(f"ONNX version: {model.opset_import[0].version}")
    print(f"Producer name: {model.producer_name}")
    print(f"Producer version: {model.producer_version}")
    print()
    
    # Graph info
    graph = model.graph
    print(f"Graph name: {graph.name}")
    print(f"Graph inputs: {len(graph.input)}")
    print(f"Graph outputs: {len(graph.output)}")
    print(f"Graph nodes: {len(graph.node)}")
    print(f"Graph initializers: {len(graph.initializer)}")
    print()
    
    # Inputs
    print("INPUTS:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else f"dynamic_{d.dim_param}" for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[inp.type.tensor_type.elem_type]} {shape}")
    print()
    
    # Outputs  
    print("OUTPUTS:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else f"dynamic_{d.dim_param}" for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[out.type.tensor_type.elem_type]} {shape}")
    print()
    
    # Initializers (weights/parameters)
    print("INITIALIZERS (Parameters):")
    print(f"Total: {len(graph.initializer)}")
    
    # Group by naming patterns
    by_pattern = {}
    for init in graph.initializer:
        name = init.name
        shape = tuple(init.dims)
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
        
        # Extract pattern
        if name.startswith('onnx::'):
            pattern = 'onnx_generated'
        elif '.' in name:
            parts = name.split('.')
            if 'lora_A' in name:
                pattern = 'lora_A'
            elif 'lora_B' in name:
                pattern = 'lora_B'
            elif 'weight' in name:
                pattern = 'standard_weight'
            elif 'bias' in name:
                pattern = 'standard_bias'
            else:
                pattern = 'other'
        else:
            pattern = 'root_level'
            
        if pattern not in by_pattern:
            by_pattern[pattern] = []
        by_pattern[pattern].append((name, shape, dtype))
    
    # Print by pattern
    for pattern, params in by_pattern.items():
        print(f"\n{pattern.upper()} ({len(params)} parameters):")
        for name, shape, dtype in sorted(params):
            print(f"  {name}: {dtype} {shape}")
    
    print()
    
    # Analyze nodes to understand computation graph
    print("COMPUTATION NODES:")
    node_types = {}
    for node in graph.node:
        op_type = node.op_type
        if op_type not in node_types:
            node_types[op_type] = 0
        node_types[op_type] += 1
    
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")
    print()
    
    # Look for specific patterns that might indicate LoRA vs full-rank
    lora_a_count = len([p for p in by_pattern.get('lora_A', [])])
    lora_b_count = len([p for p in by_pattern.get('lora_B', [])])
    onnx_generated_count = len([p for p in by_pattern.get('onnx_generated', [])])
    
    print("ANALYSIS:")
    print(f"  LoRA A parameters: {lora_a_count}")
    print(f"  LoRA B parameters: {lora_b_count}")
    print(f"  ONNX generated parameters: {onnx_generated_count}")
    
    if lora_a_count > 0 and lora_b_count > 0:
        print("  -> This appears to be a LoRA model")
    elif onnx_generated_count > 0:
        print("  -> This has ONNX-generated parameter names (possibly optimized)")
    
    print()
    
    # Show some example onnx:: parameters and their shapes
    if onnx_generated_count > 0:
        print("ONNX GENERATED PARAMETERS (first 10):")
        onnx_params = [p for p in by_pattern.get('onnx_generated', [])][:10]
        for name, shape, dtype in onnx_params:
            print(f"  {name}: {dtype} {shape}")
        print()
    
    # Try to find patterns in shapes
    print("SHAPE ANALYSIS:")
    shape_counts = {}
    for init in graph.initializer:
        shape = tuple(init.dims)
        if shape not in shape_counts:
            shape_counts[shape] = []
        shape_counts[shape].append(init.name)
    
    print("Most common shapes:")
    for shape, names in sorted(shape_counts.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {shape}: {len(names)} parameters")
        if len(names) <= 5:
            print(f"    {', '.join(names)}")
        else:
            print(f"    {', '.join(names[:3])}, ... (+{len(names)-3} more)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_onnx.py <onnx_file>")
        sys.exit(1)
    
    analyze_onnx_file(sys.argv[1])