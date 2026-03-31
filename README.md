# Neural Machine Translation & Summarization (Seq2Seq with Attention)

This repository implements a versatile Sequence-to-Sequence (Seq2Seq) framework for **Neural Machine Translation (NMT)** and **Text Summarization** using PyTorch. The architecture utilizes a Gated Recurrent Unit (GRU) based Encoder-Decoder with **Bahdanau Attention**.

## 🚀 Key Features

- **English to Hindi Translation**: Trained on the IITB English-Hindi dataset.
- **Dialogue Summarization**: Capability to summarize dialogues using the SAMSum dataset.
- **Bahdanau Attention**: Implements a powerful attention mechanism to focus on relevant parts of the input sequence during decoding.
- **Interactive Web App**: A streamlined [Streamlit](https://streamlit.io/) interface for real-time inference.
- **Evaluation Pipeline**: Automated BLEU score calculation using `SacreBLEU`.
- **Hybrid Vocabulary**: Combines large-scale dataset tokens with common conversational phrases for improved real-world performance.

---

## 🏗️ Model Architecture

The model follows a classic Encoder-Decoder structure:

1.  **Encoder**: A bidirectional GRU that processes the input sequence and produces context-aware hidden states.
2.  **Attention**: Bahdanau Attention calculates alignment scores between the current decoder hidden state and all encoder hidden states.
3.  **Decoder**: A GRU that uses the attention-weighted context and the previous prediction to generate the target sequence.

| Component | Specification |
| :--- | :--- |
| **RNN Cell** | GRU (Gated Recurrent Unit) |
| **Hidden Dim** | 512 |
| **Embedding Dim** | 256 |
| **Attention Type** | Bahdanau (Additive) |
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy (with padding ignored) |

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd ATML_Encoder
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### 1. Running the Web Application
Launch the interactive Streamlit app to translate English text to Hindi:
```bash
streamlit run app.py
```

### 2. Training the Models
To train the translation or summarization models from scratch:

**English-Hindi Translation**:
```bash
python train_en_hi.py
```

**Dialogue Summarization**:
```bash
python train_summarization.py
```

### 3. Evaluation
Evaluate the translation model's performance on a test subset using BLEU scores:
```bash
python evaluate_en_hi.py
```

---

## 📊 Datasets

- **Translation**: `cfilt/iitb-english-hindi` (via Hugging Face Datasets).
- **Summarization**: `samsum` (via Hugging Face Datasets).

> [!NOTE]
> The included `.pt` files may be trained on smaller subsets for demonstration. For production-grade results, update the training scripts to use the full dataset and increase the number of epochs.

## 📂 Project Structure

- `app.py`: Streamlit frontend.
- `model_utils.py`: Core architecture (Encoder, Decoder, Attention, Seq2Seq).
- `train_*.py`: Training logic and data preprocessing.
- `evaluate_en_hi.py`: BLEU score evaluation.
- `requirements.txt`: List of Python dependencies.

---

## 📜 License
This project is for educational and research purposes.
