import streamlit as st
import torch
import os
import torch.nn.functional as F
from model_utils import Encoder, Attention, Decoder, Seq2Seq, Vocab

# Page configuration
st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍", layout="wide")

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None, None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    en_vocab = checkpoint.get('en_vocab') or checkpoint.get('diag_vocab')
    target_vocab = checkpoint.get('hi_vocab') or checkpoint.get('es_vocab') or checkpoint.get('summ_vocab')
    params = checkpoint['params']
    
    attn = Attention(params['enc_hid_dim'], params['dec_hid_dim'])
    enc = Encoder(params['input_dim'], params['enc_emb_dim'], params['enc_hid_dim'], params['dec_hid_dim'], params['enc_dropout'])
    dec = Decoder(params['output_dim'], params['dec_emb_dim'], params['enc_hid_dim'], params['dec_hid_dim'], params['dec_dropout'], attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, en_vocab, target_vocab

def translate(model, src_vocab, trg_vocab, sentence, max_len=50, temperature=0.7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens = [tok.lower() for tok in sentence.split()]
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
            
        logits = output.squeeze(0)
        
        # Block specific specials
        logits[trg_vocab['<pad>']] = -1e9
        logits[trg_vocab['<unk>']] = -1e9
        
        # Prevent immediate <eos> on untrained models
        if i == 0:
            logits[trg_vocab['<eos>']] = -1e9
        
        pred_token = logits.argmax(0).item()
        
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break
            
    trg_tokens = [trg_vocab.lookup_token(i) for i in trg_indexes]
    output_str = " ".join(trg_tokens[1:-1])
    
    if not output_str.strip():
        output_str = "*(Model produced an empty sequence)*"
    return output_str

def main():
    st.title("English to Hindi Translator")
    
    st.sidebar.markdown("""
        **Note on Model State:**
        The current models `.pt` files were trained on a mere 50 samples to test pipeline execution speed. 
        If predictions look random or nonsensical, please run the respective `train_*.py` files with larger epochs to properly train the network!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_text = st.text_area("Enter English Text", height=150, placeholder="Enter text to translate...")
        submit = st.button("Translate to Hindi ✨")
        
    with col2:
        if submit:
            if not input_text:
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Processing with Neural Engine..."):
                    model, src_vocab, trg_vocab = load_model("en_hi_model.pt")
                    if model is None:
                        st.error("Model file en_hi_model.pt not found. Please run the training script.")
                    else:
                        output = translate(model, src_vocab, trg_vocab, input_text)
                        st.text_area("Translation", value=output, height=150, disabled=True)
        else:
            st.text_area("Translation", value="", height=150, disabled=True, placeholder="Translation will appear here...")

if __name__ == "__main__":
    main()
