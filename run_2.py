import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import string
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

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

# ---- Lazy Dataset ----

class CharGenLazyDataset(Dataset):
    def __init__(self, json_path, w2v_model, char_to_id, ctx_len=10, max_word_len=50, max_gen_len=50):
        self.json_path = json_path
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.ctx_len = ctx_len
        self.max_word_len = max_word_len
        self.max_gen_len = max_gen_len

        # Build byte offsets for all lines
        self.offsets = []
        with open(json_path, encoding="utf-8") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.json_path, encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line.strip())
        context = pad_context(sample["context"], self.ctx_len)
        misspelled = sample["misspelled"]
        prefix = sample["generated_prefix"]
        next_char = sample["next_char"]
        context_vec = vectorize_context(context, self.w2v_model, self.ctx_len)
        misspelled_oh = one_hot_chars(misspelled, self.char_to_id, self.max_word_len)
        prefix_oh = one_hot_chars(prefix, self.char_to_id, self.max_gen_len)
        # "<eow>" is used as end-of-word
        if next_char == "<eow>":
            next_id = self.char_to_id["<eow>"]
        else:
            next_id = self.char_to_id.get(next_char, 0)
        return (
            torch.tensor(context_vec, dtype=torch.float32),
            torch.tensor(misspelled_oh, dtype=torch.float32),
            torch.tensor(prefix_oh, dtype=torch.float32),
            torch.tensor(next_id, dtype=torch.long)
        )

# ---- Model ----

class ResNetFFN(nn.Module):
    def __init__(self, context_dim, word_onehot_dim, gen_onehot_dim, char_vocab_size, hidden_dim=600, num_layers=20):
        super().__init__()
        input_dim = context_dim + word_onehot_dim + gen_onehot_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, context_vec, misspelled_oh, prefix_oh):
        x = torch.cat([context_vec, misspelled_oh, prefix_oh], dim=1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        logits = self.output_layer(x)
        return logits

# ---- Training & Prediction ----

def train_model(model, dataloader, vocab_size, epochs=3, save_path="char_autocorrect.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        for context_vec, misspelled_oh, prefix_oh, next_id in loop:
            context_vec = context_vec.to(device)
            misspelled_oh = misspelled_oh.to(device)
            prefix_oh = prefix_oh.to(device)
            next_id = next_id.to(device)
            logits = model(context_vec, misspelled_oh, prefix_oh)
            loss = F.cross_entropy(logits, next_id)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}: Avg Loss = {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), save_path)

def predict_word(model, w2v_model, char_to_id, id_to_char, context_words, misspelled_word,
                 max_word_len=30, max_gen_len=50, ctx_len=10, max_output_len=30, beam_width=10):
    model.eval()
    with torch.no_grad():
        # Initialize beam with empty sequence
        beams = [(0.0, "")]  # (log_prob, sequence)
        
        for i in range(max_output_len):
            candidates = []
            
            for log_prob, sequence in beams:
                # If sequence is complete, keep it as is
                if sequence.endswith("<eow>") or len(sequence) >= max_output_len:
                    candidates.append((log_prob, sequence))
                    continue
                
                # Get predictions for this sequence
                context_vec = torch.tensor([vectorize_context(context_words, w2v_model, ctx_len)], dtype=torch.float32).to(device)
                misspelled_oh = torch.tensor([one_hot_chars(misspelled_word, char_to_id, max_word_len)], dtype=torch.float32).to(device)
                prefix_oh = torch.tensor([one_hot_chars(sequence, char_to_id, max_gen_len)], dtype=torch.float32).to(device)
                
                logits = model(context_vec, misspelled_oh, prefix_oh)
                log_probs = F.log_softmax(logits, dim=1).squeeze()
                
                # Get top beam_width candidates
                top_k = torch.topk(log_probs, beam_width)
                
                for j in range(beam_width):
                    char_id = top_k.indices[j].item()
                    char_log_prob = top_k.values[j].item()
                    pred_char = id_to_char[char_id]
                    
                    new_sequence = sequence + pred_char
                    new_log_prob = log_prob + char_log_prob
                    
                    candidates.append((new_log_prob, new_sequence))
            
            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Check if all beams are complete
            if all(seq.endswith("<eow>") for _, seq in beams):
                break
        
        # Return top 10 predictions, removing <eow> marker
        results = []
        for log_prob, sequence in beams:
            word = sequence.replace("<eow>", "")
            results.append((word, log_prob))
        
        return results[:10]

def model_matches(model, state_dict):
    """Check if the loaded state_dict fits the model structure. Print all mismatches."""
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    loaded_keys = set(state_dict.keys())

    mismatch = False

    missing = model_keys - loaded_keys
    extra = loaded_keys - model_keys
    if missing:
        print(f"Parameters missing from checkpoint: {missing}")
        mismatch = True
    if extra:
        print(f"Extra parameters in checkpoint: {extra}")
        mismatch = True

    # Check shapes for matching keys
    for k in model_keys & loaded_keys:
        if model_state[k].shape != state_dict[k].shape:
            print(f"Shape mismatch for parameter '{k}': expected {model_state[k].shape}, found {state_dict[k].shape}")
            mismatch = True

    return not mismatch

# ---- CLI ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data", type=str, default="autogen_char_data.json")
    parser.add_argument("--word2vec", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="char_autocorrect.pt")
    args = parser.parse_args()

    from gensim.models import KeyedVectors
    w2v_model = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    char_to_id, id_to_char = create_charmap()
    char_vocab_size = len(char_to_id)

    context_dim = args.ctx_len * w2v_model.vector_size
    word_onehot_dim = args.max_word_len * char_vocab_size
    gen_onehot_dim = args.max_gen_len * char_vocab_size

    model = ResNetFFN(
        context_dim=context_dim,
        word_onehot_dim=word_onehot_dim,
        gen_onehot_dim=gen_onehot_dim,
        char_vocab_size=char_vocab_size,
        hidden_dim=600,
        num_layers=30
    ).to(device)

    if args.train:
        from pathlib import Path
        model_path = args.model
        model_exists = Path(model_path).exists()
        should_resume = False

        if model_exists:
            print(f"Model file '{model_path}' found, checking compatibility...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # If saved as state_dict only
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                if model_matches(model, checkpoint):
                    print("Model structure matches. Resuming training from saved model.")
                    model.load_state_dict(checkpoint)
                    should_resume = True
                else:
                    print("Saved model structure does not match the current code. Exiting for safety.")
                    exit(1)
            except Exception as e:
                print(f"Error loading model: {e}\nExiting.")
                exit(1)
        else:
            print("No existing model found. Training new model from scratch.")

        # Now proceed to dataset and training as before
        dataset = CharGenLazyDataset(args.data, w2v_model, char_to_id,
                                ctx_len=args.ctx_len, max_word_len=args.max_word_len, max_gen_len=args.max_gen_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        train_model(model, dataloader, vocab_size=char_vocab_size, epochs=args.epochs, save_path=args.model)   
    elif args.predict:
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print("Interactive prediction mode. Press Ctrl+C to exit.")
        try:
            while True:
                print("\nEnter context words (space-separated):")
                context = input("> ").strip().split()
                print("Enter misspelled word:")
                misspelled = input("> ").strip()
                predictions = predict_word(model, w2v_model, char_to_id, id_to_char, context, misspelled,
                                           max_word_len=args.max_word_len, max_gen_len=args.max_gen_len, ctx_len=args.ctx_len)
                print("Top 10 predictions:")
                for i, (word, log_prob) in enumerate(predictions, 1):
                    print(f"{i:2d}. {word} (log_prob: {log_prob:.3f})")
        except KeyboardInterrupt:
            print("\nExiting...")

