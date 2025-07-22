# FINE-TUNING

import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from transformers import pipeline
from core import fine_tune_tl, combined_augment, w2v_augment
from tensorflow.keras import mixed_precision

# Rimuove eventuali file con quei nomi e crea le cartelle
for sub in ("train", "validation"):
    path = os.path.join("logs", sub)
    if os.path.isfile(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

# Uso GPU se presente
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # abilita l’allocazione dinamica di memoria
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Utilizzo GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print("Errore nella configurazione GPU:", e)
else:
    print("Nessuna GPU rilevata, userò la CPU.")

# Mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Abilita XLA JIT compiler
tf.config.optimizer.set_jit(True)

# Parametri per import
TOKENIZER_PATH       = "tokenizer.pkl"
EMBED_MATRIX_PATH    = "embedding_matrix.npy"
W2V_MODEL_PATH       = "Word2Vec.bin"
BASE_MODEL_PATH      = "best_model.keras"
MAX_LEN              = 200

# nuova batch di esempi per il fine-tuning
new_records = [
    "This film had a stunning visual style and a captivating storyline.",
    "I found the plot quite predictable and the acting mediocre.",
    "The cinematography was breathtaking, but the story lacked depth.",
    "An amazing performance by the lead actor makes this film a must-watch.",
    "Not worth the hype; the movie was too slow and confusing."
]
new_labels = [3, 1, 2, 4, 0]

if __name__ == "__main__":
    # Carica gli artefatti già creati da main1.py
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    embedding_matrix = np.load(EMBED_MATRIX_PATH)
    w2v_model = Word2Vec.load(W2V_MODEL_PATH)

    # Prepara le pipelines di traduzione
    en2fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=-1)
    fr2en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en", device=-1)
    en2de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", device=-1)
    de2en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en", device=-1)

    expanded_records = []
    expanded_labels  = []

    # Data augmentation per fine tuning
    for rec, lab in zip(new_records, new_labels):
    # 1) Combino i due EDA classici
        for aug in combined_augment(rec):
            expanded_records.append(aug)
            expanded_labels.append(lab)
            # 2) una W2V-augment
            expanded_records.append(w2v_augment(rec, w2v_model, p=0.15, topn=8))
            expanded_labels.append(lab)
            # 3) aggiungo anche l’originale
            expanded_records.append(rec)
            expanded_labels.append(lab)

    print("Starting fine-tuning…")
    history1, history2, _ = fine_tune_tl(
        records=expanded_records,
        labels=expanded_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        base_model_path=BASE_MODEL_PATH,
        en2fr=en2fr, fr2en=fr2en, en2de=en2de, de2en=de2en,
        w2v_model=w2v_model,
        random_state=42,
        n_epochs=8,
        val_split=0.2,
        pct_bt=1.0
    )
    print("Fine-tuning completato. Modello salvato in 'best_model_finetuned.keras'.")

    # Plot delle curve principali del modello
    # solo head
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history1.history['loss'], label='Train Loss')
    plt.plot(history1.history['val_loss'], label='Val Loss')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(history1.history['accuracy'], label='Train Acc')
    plt.plot(history1.history['val_accuracy'], label='Val Acc')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Accuracy Curve')
    plt.tight_layout(), plt.show()

    # rifinitura completa transfer learning
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history2.history['loss'], label='Train Loss')
    plt.plot(history2.history['val_loss'], label='Val Loss')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(history2.history['accuracy'], label='Train Acc')
    plt.plot(history2.history['val_accuracy'], label='Val Acc')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Accuracy Curve')
    plt.tight_layout(), plt.show()
