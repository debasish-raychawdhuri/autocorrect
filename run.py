
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
import json
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
import string

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

class LowRankLinearRelu(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Linear(in_dim + 1, rank, bias=False)   # +1 for constant feature
        self.B = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x):
        bias_feature = torch.ones(x.size(0), 1, device=x.device)
        x = torch.cat([x, bias_feature], dim=1)
        return torch.relu(self.B(self.A(x)))

class AutocorrectFFN(nn.Module):
    def __init__(self, context_dim=3000, word_len=50, char_vocab_size=100, char_embed_dim=16, ffn_rank=300, num_layers=20):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim).to(device)
        self.context_dim = context_dim
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.word_len = word_len
        self.input_dim = context_dim + word_len * char_embed_dim
        self.layers = nn.ModuleList([
            LowRankLinearRelu(self.input_dim, self.input_dim, ffn_rank).to(device)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(self.input_dim, word_len * char_vocab_size).to(device)

    def forward(self, context_vec, word_char_ids):
        context_vec = context_vec.to(device)
        word_char_ids = word_char_ids.to(device)

        char_vec = self.char_embed(word_char_ids)  # [batch, word_len, char_embed_dim]
        char_vec = char_vec.view(char_vec.size(0), -1)  # flatten to [batch, word_len * char_embed_dim]

        x = torch.cat([context_vec, char_vec], dim=1)  # [batch, total_input_dim]
        for layer in self.layers:
            x = x + layer(x)  # ResNet-style skip connection
        output = self.output_layer(x)
        return output.view(-1, self.word_len, self.char_vocab_size)  # [batch, word_len, char_vocab_size]

# ---------------------- Input Preparation ----------------------

def load_word2vec(path):
    return KeyedVectors.load_word2vec_format(path, binary=True)

def vectorize_context(context_words, w2v_model, max_context_len=10, embed_dim=300):
    vecs = []
    for word in context_words[-max_context_len:]:
        if word in w2v_model:
            vecs.append(w2v_model[word])
        else:
            vecs.append(np.zeros(embed_dim))
    while len(vecs) < max_context_len:
        vecs.insert(0, np.zeros(embed_dim))  # pad on the left
    return np.concatenate(vecs, axis=0)

def encode_chars(word, char_to_id, max_len=50):
    ids = [char_to_id.get(c, 0) for c in word[-max_len:]]
    while len(ids) < max_len:
        ids.insert(0, 0)
    return ids[:max_len]

def create_charmap():
    printable = string.printable
    return {c: i + 1 for i, c in enumerate(printable)}  # reserve 0 for padding

def decode_chars(char_ids, id_to_char):
    return ''.join(id_to_char.get(c, '?') for c in char_ids if c != 0)

# ---------------------- Dataset Loader ----------------------

class AutocorrectDataset(Dataset):
    def __init__(self, json_path, w2v_model, char_to_id, max_context_len=10, max_word_len=50):
        self.samples = []
        self.w2v_model = w2v_model
        self.char_to_id = char_to_id
        self.max_context_len = max_context_len
        self.max_word_len = max_word_len
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_words = sample["input"].split()
        context = input_words[:-1]
        last_word = input_words[-1]
        target_word = sample["target"]

        context_vec = vectorize_context(context, self.w2v_model, self.max_context_len)
        input_char_ids = encode_chars(last_word, self.char_to_id, self.max_word_len)
        target_char_ids = encode_chars(target_word, self.char_to_id, self.max_word_len)

        return (
            torch.tensor(context_vec, dtype=torch.float32),
            torch.tensor(input_char_ids, dtype=torch.long),
            torch.tensor(target_char_ids, dtype=torch.long)
        )

# ---------------------- CLI ----------------------

def train_model(model, dataloader, char_vocab_size, epochs=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        for context_vec, char_ids, target_ids in loop:
            context_vec, char_ids, target_ids = context_vec.to(device), char_ids.to(device), target_ids.to(device)
            logits = model(context_vec, char_ids)  # [batch, word_len, char_vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, char_vocab_size),
                target_ids.view(-1),
                ignore_index=0
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(dataloader)
        torch.save(model.state_dict(), f"autocorrect_model_{epoch+1}.pt")
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    torch.save(model.state_dict(), "autocorrect_model.pt")

def predict(model, w2v_model, char_to_id, sentence):
    id_to_char = {v: k for k, v in char_to_id.items()}
    words = sentence.strip().split()
    context = words[:-1]
    last_word = words[-1]
    context_vec = torch.tensor([vectorize_context(context, w2v_model)], dtype=torch.float32).to(device)
    char_ids = torch.tensor([encode_chars(last_word, char_to_id)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(context_vec, char_ids)
        pred_ids = logits.argmax(dim=2)[0].tolist()
    print("Predicted word:", decode_chars(pred_ids, id_to_char))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    parser.add_argument("--data", type=str, default="data.json")
    parser.add_argument("--word2vec", type=str, required=True)
    args = parser.parse_args()

    w2v_model = load_word2vec(args.word2vec)
    char_to_id = create_charmap()

    model = AutocorrectFFN(char_vocab_size=len(char_to_id) + 1).to(device)

    if args.train:
        dataset = AutocorrectDataset(args.data, w2v_model, char_to_id)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        train_model(model, dataloader, char_vocab_size=len(char_to_id) + 1)
    elif args.predict:
        model.load_state_dict(torch.load("autocorrect_model.pt", map_location=device))
        model.eval()
        predict(model, w2v_model, char_to_id, args.predict)
