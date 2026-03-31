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

def tokenize_en(text):
    return [tok.lower() for tok in text.split()]

def tokenize_hi(text):
    return [tok for tok in text.split()]

# Common conversational sentences to ensure everyday words are in vocabulary
COMMON_PAIRS = [
    {'translation': {'en': 'how are you', 'hi': 'आप कैसे हैं'}},
    {'translation': {'en': 'what is your name', 'hi': 'आपका नाम क्या है'}},
    {'translation': {'en': 'where are you going', 'hi': 'आप कहाँ जा रहे हैं'}},
    {'translation': {'en': 'i am fine', 'hi': 'मैं ठीक हूँ'}},
    {'translation': {'en': 'thank you very much', 'hi': 'आपका बहुत बहुत धन्यवाद'}},
    {'translation': {'en': 'good morning', 'hi': 'शुभ प्रभात'}},
    {'translation': {'en': 'good night', 'hi': 'शुभ रात्रि'}},
    {'translation': {'en': 'please sit down', 'hi': 'कृपया बैठ जाइए'}},
    {'translation': {'en': 'what time is it', 'hi': 'कितने बजे हैं'}},
    {'translation': {'en': 'where do you live', 'hi': 'आप कहाँ रहते हैं'}},
    {'translation': {'en': 'i love you', 'hi': 'मैं तुमसे प्यार करता हूँ'}},
    {'translation': {'en': 'what is this', 'hi': 'यह क्या है'}},
    {'translation': {'en': 'who are you', 'hi': 'आप कौन हैं'}},
    {'translation': {'en': 'how much does this cost', 'hi': 'इसकी कीमत कितनी है'}},
    {'translation': {'en': 'i do not understand', 'hi': 'मुझे समझ नहीं आया'}},
    {'translation': {'en': 'please help me', 'hi': 'कृपया मेरी मदद करें'}},
    {'translation': {'en': 'where is the market', 'hi': 'बाज़ार कहाँ है'}},
    {'translation': {'en': 'i am going home', 'hi': 'मैं घर जा रहा हूँ'}},
    {'translation': {'en': 'what do you want', 'hi': 'आप क्या चाहते हैं'}},
    {'translation': {'en': 'he is my friend', 'hi': 'वह मेरा दोस्त है'}},
    {'translation': {'en': 'she is a teacher', 'hi': 'वह एक शिक्षिका है'}},
    {'translation': {'en': 'we are students', 'hi': 'हम छात्र हैं'}},
    {'translation': {'en': 'they are coming tomorrow', 'hi': 'वे कल आ रहे हैं'}},
    {'translation': {'en': 'i like this book', 'hi': 'मुझे यह किताब पसंद है'}},
    {'translation': {'en': 'the weather is good today', 'hi': 'आज मौसम अच्छा है'}},
    {'translation': {'en': 'can you speak hindi', 'hi': 'क्या आप हिंदी बोल सकते हैं'}},
    {'translation': {'en': 'i am learning hindi', 'hi': 'मैं हिंदी सीख रहा हूँ'}},
    {'translation': {'en': 'the food is very tasty', 'hi': 'खाना बहुत स्वादिष्ट है'}},
    {'translation': {'en': 'i need water', 'hi': 'मुझे पानी चाहिए'}},
    {'translation': {'en': 'let us go', 'hi': 'चलो चलते हैं'}},
]

def build_vocabs():
    print("Loading dataset...")
    # Loading only a subset for faster training
    try:
        dataset = load_dataset('cfilt/iitb-english-hindi', split='train[:15000]') 
        dataset = list(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}. Using dummy data for demonstration.")
        dataset = []
    
    # Inject common conversational pairs (repeated to boost their weight)
    for pair in COMMON_PAIRS:
        for _ in range(10):  # repeat each 10 times for emphasis
            dataset.append(pair)
    
    print("Building English Vocabulary...")
    en_counter = Counter()
    for item in dataset:
        en_counter.update(tokenize_en(item['translation']['en']))
    en_vocab = Vocab(en_counter)
    
    print("Building Hindi Vocabulary...")
    hi_counter = Counter()
    for item in dataset:
        hi_counter.update(tokenize_hi(item['translation']['hi']))
    hi_vocab = Vocab(hi_counter)
    
    return en_vocab, hi_vocab, dataset

def preprocess_data(dataset, en_vocab, hi_vocab):
    processed_data = []
    for item in dataset:
        src = [en_vocab['<sos>']] + [en_vocab[token] for token in tokenize_en(item['translation']['en'])] + [en_vocab['<eos>']]
        trg = [hi_vocab['<sos>']] + [hi_vocab[token] for token in tokenize_hi(item['translation']['hi'])] + [hi_vocab['<eos>']]
        processed_data.append((torch.tensor(src), torch.tensor(trg)))
    return processed_data

def collate_fn(batch):
    src_list, trg_list = [], []
    for src, trg in batch:
        src_list.append(src)
        trg_list.append(trg)
    src_list = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=en_vocab['<pad>'])
    trg_list = torch.nn.utils.rnn.pad_sequence(trg_list, padding_value=hi_vocab['<pad>'])
    return src_list, trg_list

if __name__ == "__main__":
    en_vocab, hi_vocab, raw_dataset = build_vocabs()
    
    print(f"Unique tokens in source (en) vocabulary: {len(en_vocab)}")
    print(f"Unique tokens in target (hi) vocabulary: {len(hi_vocab)}")
    
    train_data = preprocess_data(raw_dataset, en_vocab, hi_vocab)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(hi_vocab)
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
    TRG_PAD_IDX = hi_vocab['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(tqdm(iterator)):
            src = src.to(device)
            trg = trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            # output = [trg len, batch size, output dim]
            # trg = [trg len, batch size]
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    N_EPOCHS = 5 # More epochs for better convergence
    CLIP = 1
    
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        end_time = time.time()
        
        print(f'Epoch: {epoch+1:02} | Time: {end_time - start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
    
    # Save model and vocabs
    torch.save({
        'model_state_dict': model.state_dict(),
        'en_vocab': en_vocab,
        'hi_vocab': hi_vocab,
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
    }, 'en_hi_model.pt')
    print("Model saved as en_hi_model.pt")
