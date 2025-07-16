class BatchSamplerByChunks(torch.utils.data.Sampler):
    """Samples batches in chunks to limit memory usage"""
    
    def __init__(self, dataset_size, batch_size, chunk_size=1000000, shuffle=True):
        """
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Size of each batch
            chunk_size: Number of samples to load into memory at once
            shuffle: Whether to shuffle samples
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.chunk_size = min(chunk_size, dataset_size)  # Don't exceed dataset size
        self.shuffle = shuffle
        
        # Calculate number of batches
        self.num_batches = dataset_size // batch_size
        if dataset_size % batch_size > 0:
            self.num_batches += 1
    
    def __iter__(self):
        # Process the dataset in chunks
        chunks = [(i, min(i + self.chunk_size, self.dataset_size)) 
                 for i in range(0, self.dataset_size, self.chunk_size)]
        
        if self.shuffle:
            # Shuffle the order of chunks
            random.shuffle(chunks)
        
        # Process each chunk
        for chunk_start, chunk_end in chunks:
            # Create indices for this chunk
            indices = list(range(chunk_start, chunk_end))
            
            if self.shuffle:
                # Shuffle indices within the chunk
                random.shuffle(indices)
            
            # Yield batches from this chunk
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield batch_indices
    
    def __len__(self):
        return self.num_batches
