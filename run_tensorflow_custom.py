#!/usr/bin/env python3
"""
TensorFlow implementation using custom word2vec format (no gensim dependency)
Compatible with CUDA 12.8 environments
"""
import tensorflow as tf
import numpy as np
import json
import argparse
import os
import time
import string
from tqdm import tqdm
from custom_word2vec import load_custom_word2vec

# Enable mixed precision for better performance
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Enable memory growth for GPUs to avoid OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

# Custom birely activation function: birely(x) = relu(x) - 0.3 * relu(-x)
@tf.function
def birely(x):
    return tf.nn.relu(x) - 0.3 * tf.nn.relu(-x)

# ---- Char Map ----
def create_charmap():
    printable = string.printable.replace('"', '')  # Avoid JSON quote issues
    char_list = list(printable) + ["<eow>"]
    char_to_id = {c: i for i, c in enumerate(char_list)}
    id_to_char = {i: c for i, c in enumerate(char_list)}
    return char_to_id, id_to_char

# ---- Input Preparation ----
def one_hot_chars(seq, char_to_id, max_len):
    arr = np.zeros((max_len, len(char_to_id)), dtype=np.float32)
    seq = seq[-max_len:]
    for i, c in enumerate(seq[::-1]):
        idx = char_to_id.get(c, 0)
        arr[max_len - 1 - i, idx] = 1.0
    return arr.flatten()

def pad_context(words, ctx_len=10):
    return [""] * max(0, ctx_len - len(words)) + words[-ctx_len:]

def vectorize_context(context_words, w2v_model, ctx_len=10, embed_dim=300):
    vecs = []
    for word in context_words[-ctx_len:]:
        if word in w2v_model:
            vecs.append(w2v_model[word])
        else:
            vecs.append(np.zeros(embed_dim))
    while len(vecs) < ctx_len:
        vecs.insert(0, np.zeros(embed_dim))
    return np.concatenate(vecs, axis=0)

# ---- Data Pipeline ----
def parse_sample(line, w2v_model, char_to_id, ctx_len, max_word_len, max_gen_len):
    """Parse a single JSON line and convert to model inputs"""
    try:
        sample = json.loads(line.strip())
        context = pad_context(sample["context"], ctx_len)
        misspelled = sample["misspelled"]
        prefix = sample["generated_prefix"]
        next_char = sample["next_char"]
        
        context_vec = vectorize_context(context, w2v_model, ctx_len)
        misspelled_oh = one_hot_chars(misspelled, char_to_id, max_word_len)
        prefix_oh = one_hot_chars(prefix, char_to_id, max_gen_len)
        
        # "<eow>" is used as end-of-word
        if next_char == "<eow>":
            next_id = char_to_id["<eow>"]
        else:
            next_id = char_to_id.get(next_char, 0)
            
        return context_vec, misspelled_oh, prefix_oh, next_id
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def create_dataset(json_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, 
                  max_gen_len=50, batch_size=32, num_parallel_calls=tf.data.AUTOTUNE):
    """Create TensorFlow dataset from JSON file"""
    
    # Count total lines for progress tracking
    with open(json_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total samples in dataset: {total_lines}")
    
    def generator():
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = parse_sample(line, w2v_model, char_to_id, ctx_len, max_word_len, max_gen_len)
                if result is not None:
                    yield result
    
    # Define output signature
    context_dim = ctx_len * w2v_model.vector_size
    word_onehot_dim = max_word_len * len(char_to_id)
    gen_onehot_dim = max_gen_len * len(char_to_id)
    
    output_signature = (
        tf.TensorSpec(shape=(context_dim,), dtype=tf.float32),
        tf.TensorSpec(shape=(word_onehot_dim,), dtype=tf.float32),
        tf.TensorSpec(shape=(gen_onehot_dim,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Optimize pipeline
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, total_lines

# ---- ResNet FFN Model ----
class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dense = tf.keras.layers.Dense(hidden_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.layer_norm(x)
        x = birely(x)  # Custom activation
        return inputs + x  # Residual connection
    
    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

class ResNetFFN(tf.keras.Model):
    def __init__(self, context_dim, word_onehot_dim, gen_onehot_dim, 
                 char_vocab_size, hidden_dim=600, num_layers=20, **kwargs):
        super().__init__(**kwargs)
        self.context_dim = context_dim
        self.word_onehot_dim = word_onehot_dim
        self.gen_onehot_dim = gen_onehot_dim
        self.char_vocab_size = char_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        input_dim = context_dim + word_onehot_dim + gen_onehot_dim
        
        self.input_proj = tf.keras.layers.Dense(hidden_dim, name="input_proj")
        self.resnet_blocks = [
            ResNetBlock(hidden_dim, name=f"resnet_block_{i}")
            for i in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(char_vocab_size, name="output_layer")
        
    def call(self, inputs):
        context_vec, misspelled_oh, prefix_oh = inputs
        
        # Concatenate all inputs
        x = tf.concat([context_vec, misspelled_oh, prefix_oh], axis=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)
            
        # Output layer
        logits = self.output_layer(x)
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "context_dim": self.context_dim,
            "word_onehot_dim": self.word_onehot_dim,
            "gen_onehot_dim": self.gen_onehot_dim,
            "char_vocab_size": self.char_vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers
        })
        return config

# ---- Training Function ----
def train_model(model, dataset, total_samples, batch_size, epochs=3, save_path="char_autocorrect_tf"):
    """Train the model with distributed strategy"""
    
    # Setup optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Calculate steps per epoch
    steps_per_epoch = total_samples // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{save_path}_epoch_{{epoch:02d}}.h5",
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{save_path}_best.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor='loss',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=1,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train the model
    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save_weights(f"{save_path}_final.h5")
    print(f"Training completed! Final model saved to {save_path}_final.h5")
    
    return history

# ---- Prediction Function ----
def predict_correction(model, w2v_model, char_to_id, sentence, ctx_len=10, 
                      max_word_len=50, max_gen_len=50):
    """Predict character correction for a sentence"""
    id_to_char = {v: k for k, v in char_to_id.items()}
    
    words = sentence.strip().split()
    context = words[:-1]
    last_word = words[-1]
    
    # Prepare inputs
    context_vec = vectorize_context(context, w2v_model, ctx_len)
    misspelled_oh = one_hot_chars(last_word, char_to_id, max_word_len)
    
    # Generate correction character by character
    corrected = ""
    prefix = ""
    
    for _ in range(max_word_len):
        prefix_oh = one_hot_chars(prefix, char_to_id, max_gen_len)
        
        # Prepare batch inputs
        inputs = [
            tf.expand_dims(context_vec, 0),
            tf.expand_dims(misspelled_oh, 0),
            tf.expand_dims(prefix_oh, 0)
        ]
        
        # Predict next character
        logits = model(inputs)
        predicted_id = tf.argmax(logits[0]).numpy()
        predicted_char = id_to_char.get(predicted_id, "")
        
        if predicted_char == "<eow>" or predicted_char == "":
            break
            
        corrected += predicted_char
        prefix = corrected
    
    return corrected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="autogen_char_data.json")
    parser.add_argument("--word2vec", type=str, required=True, help="Custom word2vec file (.npz, .pkl, .json)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="char_autocorrect_tf")
    parser.add_argument("--predict", type=str, help="Sentence to predict correction for")
    parser.add_argument("--load_weights", type=str, help="Path to saved weights")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()
    
    # Force CPU execution if requested
    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Forcing CPU execution")
    
    # Setup distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # Adjust batch size for multiple GPUs
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    
    # Load custom word2vec model
    print("Loading custom Word2Vec model...")
    w2v_model = load_custom_word2vec(args.word2vec)
    char_to_id, id_to_char = create_charmap()
    char_vocab_size = len(char_to_id)
    
    # Calculate dimensions
    context_dim = args.ctx_len * w2v_model.vector_size
    word_onehot_dim = args.max_word_len * char_vocab_size
    gen_onehot_dim = args.max_gen_len * char_vocab_size
    
    with strategy.scope():
        # Create model
        model = ResNetFFN(
            context_dim=context_dim,
            word_onehot_dim=word_onehot_dim,
            gen_onehot_dim=gen_onehot_dim,
            char_vocab_size=char_vocab_size,
            hidden_dim=600,
            num_layers=30
        )
        
        # Build model by calling it once
        dummy_context = tf.zeros((1, context_dim))
        dummy_word = tf.zeros((1, word_onehot_dim))
        dummy_gen = tf.zeros((1, gen_onehot_dim))
        _ = model([dummy_context, dummy_word, dummy_gen])
        
        # Print model info
        total_params = model.count_params()
        print(f"Model created with {total_params:,} parameters")
        
        # Load weights if specified
        if args.load_weights:
            print(f"Loading weights from {args.load_weights}")
            model.load_weights(args.load_weights)
    
    # Prediction mode
    if args.predict:
        if not args.load_weights:
            print("Error: --load_weights required for prediction")
            return
        
        correction = predict_correction(
            model, w2v_model, char_to_id, args.predict,
            ctx_len=args.ctx_len,
            max_word_len=args.max_word_len,
            max_gen_len=args.max_gen_len
        )
        print(f"Input: {args.predict}")
        print(f"Predicted correction: {correction}")
        return
    
    # Training mode
    print("Creating dataset...")
    dataset, total_samples = create_dataset(
        args.data, w2v_model, char_to_id,
        ctx_len=args.ctx_len,
        max_word_len=args.max_word_len,
        max_gen_len=args.max_gen_len,
        batch_size=global_batch_size
    )
    
    # Distribute dataset
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    print(f"Dataset created with {total_samples} samples")
    print(f"Global batch size: {global_batch_size}")
    
    # Train model
    history = train_model(
        model, dataset, total_samples, global_batch_size,
        epochs=args.epochs, save_path=args.model
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()