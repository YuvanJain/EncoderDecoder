import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datasets import load_dataset
from model_utils import Encoder, Attention, Decoder, Seq2Seq, Vocab
import random
import numpy as np
import time
import os
from tqdm import tqdm

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize(text):
    return [tok.lower() for tok in text.split()]

def build_vocabs():
    print("Loading SAMSum dataset...")
    try:
        dataset = load_dataset('samsum', split='train[:50]') 
    except Exception as e:
        print(f"Error loading dataset: {e}. Using dummy data.")
        dataset = [{'dialogue': 'A: hi B: hello', 'summary': 'A and B greet each other'}] * 100
    
    print("Building Dialogue Vocabulary...")
    diag_counter = Counter()
    for item in dataset:
        diag_counter.update(tokenize(item['dialogue']))
    diag_vocab = Vocab(diag_counter)
    
    print("Building Summary Vocabulary...")
    summ_counter = Counter()
    for item in dataset:
        summ_counter.update(tokenize(item['summary']))
    summ_vocab = Vocab(summ_counter)
    
    return diag_vocab, summ_vocab, dataset

def preprocess_data(dataset, diag_vocab, summ_vocab):
    processed_data = []
    for item in dataset:
        src = [diag_vocab['<sos>']] + [diag_vocab[token] for token in tokenize(item['dialogue'])] + [diag_vocab['<eos>']]
        trg = [summ_vocab['<sos>']] + [summ_vocab[token] for token in tokenize(item['summary'])] + [summ_vocab['<eos>']]
        processed_data.append((torch.tensor(src), torch.tensor(trg)))
    return processed_data

def collate_fn(batch):
    src_list, trg_list = [], []
    for src, trg in batch:
        src_list.append(src)
        trg_list.append(trg)
    src_list = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=diag_vocab.pad_index)
    trg_list = torch.nn.utils.rnn.pad_sequence(trg_list, padding_value=summ_vocab.pad_index)
    return src_list, trg_list

if __name__ == "__main__":
    diag_vocab, summ_vocab, raw_dataset = build_vocabs()
    
    print(f"Unique tokens in source (dialogue) vocabulary: {len(diag_vocab)}")
    print(f"Unique tokens in target (summary) vocabulary: {len(summ_vocab)}")
    
    train_data = preprocess_data(raw_dataset, diag_vocab, summ_vocab)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    INPUT_DIM = len(diag_vocab)
    OUTPUT_DIM = len(summ_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=summ_vocab.pad_index)
    
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(tqdm(iterator)):
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    N_EPOCHS = 1
    CLIP = 1
    
    print("Starting training for Summarization...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        end_time = time.time()
        print(f'Epoch: {epoch+1:02} | Time: {end_time - start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'diag_vocab': diag_vocab,
        'summ_vocab': summ_vocab,
        'params': {
            'input_dim': INPUT_DIM,
            'output_dim': OUTPUT_DIM,
            'enc_emb_dim': ENC_EMB_DIM,
            'dec_emb_dim': DEC_EMB_DIM,
            'enc_hid_dim': ENC_HID_DIM,
            'dec_hid_dim': DEC_HID_DIM,
            'enc_dropout': ENC_DROPOUT,
            'dec_dropout': DEC_DROPOUT
        }
    }, 'summarization_model.pt')
    print("Model saved as summarization_model.pt")
