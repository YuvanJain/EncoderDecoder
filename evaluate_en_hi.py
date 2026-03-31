import torch
import torch.nn as nn
from model_utils import Encoder, Attention, Decoder, Seq2Seq, Vocab
from datasets import load_dataset
import random
from sacrebleu.metrics import BLEU
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_en(text):
    return [tok.lower() for tok in text.split()]

def tokenize_hi(text):
    return [tok for tok in text.split()]

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = [token.lower() for token in sentence]
    
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    trg_indexes = [trg_vocab['<sos>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break
            
    trg_tokens = [trg_vocab.lookup_token(i) for i in trg_indexes]
    return trg_tokens[1:]

if __name__ == "__main__":
    checkpoint = torch.load('en_hi_model.pt', map_location=device, weights_only=False)
    en_vocab = checkpoint['en_vocab']
    hi_vocab = checkpoint['hi_vocab']
    params = checkpoint['params']
    
    attn = Attention(params['enc_hid_dim'], params['dec_hid_dim'])
    enc = Encoder(params['input_dim'], params['enc_emb_dim'], params['enc_hid_dim'], params['dec_hid_dim'], params['enc_dropout'])
    dec = Decoder(params['output_dim'], params['dec_emb_dim'], params['enc_hid_dim'], params['dec_hid_dim'], params['dec_dropout'], attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading test dataset subset...")
    test_dataset = load_dataset('cfilt/iitb-english-hindi', split='train[5000:5100]')
    
    bleu = BLEU()
    references = []
    candidates = []
    
    print("Evaluating on 100 samples...")
    for item in tqdm(test_dataset):
        src = item['translation']['en']
        trg = item['translation']['hi']
        
        translation = translate_sentence(src, en_vocab, hi_vocab, model, device)
        translation_str = " ".join(translation[:-1]) # remove <eos>
        
        candidates.append(translation_str)
        references.append([trg])
        
    result = bleu.corpus_score(candidates, references)
    print(f"BLEU score: {result.score:.2f}")
    
    print("\nSample Translations:")
    for i in range(5):
        item = test_dataset[i]
        src = item['translation']['en']
        trg = item['translation']['hi']
        translation = translate_sentence(src, en_vocab, hi_vocab, model, device)
        print(f"SRC: {src}")
        print(f"TRG: {trg}")
        print(f"PREDD: {' '.join(translation[:-1])}")
        print("-" * 20)
