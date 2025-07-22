############### Progetto di RNN per classificazione di recensioni ###################

# Importazioni librerie
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import pickle
import random
import nltk
import string
from math import ceil
import os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, GaussianNoise, LayerNormalization, Input, MultiHeadAttention, SpatialDropout1D, GlobalAveragePooling1D, Dropout, BatchNormalization, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, Conv1D, MaxPooling1D, Attention
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, CosineDecay, LearningRateSchedule
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.saving import register_keras_serializable
from keras.regularizers import l2
from sklearn.utils import resample
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline

# Rimuove eventuali file con quei nomi e crea le cartelle
for sub in ("train", "validation"):
    path = os.path.join("logs", sub)
    if os.path.isfile(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

# Uso GPU se presente
print("Built with CUDA:", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print("GPU disponibile:", gpus)
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

# Impostazione del seed per la riproducibilità
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Download delle risorse necessarie per embedding
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Classe per i layer adattivi per transfer learning
class Adapter(Layer):
    def __init__(self, bottleneck_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.down = Dense(self.bottleneck_dim, activation='relu', name=self.name+'_down')
        self.up   = Dense(dim, activation='linear', name=self.name+'_up')
        super().build(input_shape)

    def call(self, x):
        h = self.down(x)
        return self.up(h) + x

# FUNZIONI DI DATA AUGMENTATION
# Aggiunta di sinonimi della parola in modo randomico 
def synonym_rep(sentence):
    words = nltk.word_tokenize(sentence)
    new_words = []
    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            synonym = random.choice(synsets[0].lemmas()).name()
            new_words.append(synonym if synonym != word else word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Cancellazione di alcune parole in modo randomico
def random_deletion(sentence, p=0.2):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0,1) > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

# Scambio di alcune frasi tra loro in maniera randomica
def random_swap(sentence, n_swaps=1):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

# Inserimento di nuovi token coerenti dentro una frase in maniera randomica
def random_insertion(sentence, n_insertions=1):
    words = sentence.split()
    new_words = words.copy()
    for _ in range(n_insertions):
        random_word = random.choice(words)
        synsets = wordnet.synsets(random_word)
        if synsets:
            synonym = random.choice(synsets[0].lemmas()).name()
            insert_idx = random.randint(0, len(new_words))
            new_words.insert(insert_idx, synonym)
    return ' '.join(new_words)

# Scelta randomica di quale metodo di data augmentation attuare
def eda_augment(sentence):
    methods = [synonym_rep, random_insertion, random_swap, random_deletion]
    return random.choice(methods)(sentence)

def combined_augment(sentence):
    return [eda_augment(sentence), eda_augment(sentence)]

# Per ogni parola la sostituisce in maniera randomica con un suo vicino tra le parole più gettonate nel modello word2vec addestrato sul mio dataset
def w2v_augment(sentence, w2v_model, p=0.1, topn=5):
    words = sentence.split()
    new_words = []
    for w in words:
        if w in w2v_model.wv and random.random() < p:
            neighbors = [x for x,_ in w2v_model.wv.most_similar(w, topn=topn)]
            new_words.append(random.choice(neighbors))
        else:
            new_words.append(w)
    return " ".join(new_words)

# classe per gestire learning rate warmup iniziale e adattarlo progressivamente
@register_keras_serializable(package="Custom", name="WarmUpCosineDecay")
class WarmUpCosineDecay(LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps):
        super().__init__()
        self.initial_lr   = initial_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps
        # cosine decay da total_steps - warmup_steps
        self.cosine_sched = CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps - warmup_steps
        )

    def __call__(self, step):
        # cast a float
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        # lr durante il warmup
        lr_warmup = self.initial_lr * (step / warmup)
        # lr col decay coseno (passando step - warmup)
        lr_cosine = self.cosine_sched(step - warmup)
        # scegli quale usare con tf.where
        return tf.where(step < warmup, lr_warmup, lr_cosine)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

# PRE-PROCESSING DEL TESTO
def preprocess(sentence):
    keep_points = ["'", "-", "'s", "'d", "'t"]
    remove_points = string.punctuation.translate(str.maketrans('', '', ''.join(keep_points)))
    sentence = sentence.lower().translate(str.maketrans('', '', remove_points))
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# FACTORING DEL MODELLO per k-fold
def build_model(vocab_size, embedding_dim, embedding_matrix, max_len, embed_trainable):
    inp = Input(shape=(max_len,), name="input_seq")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=embed_trainable,
        input_length=max_len,
        name="embedding"
    )(inp)
    x = GaussianNoise(0.2)(x)
    x = SpatialDropout1D(0.3)(x)
    #x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
    #x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
    x = MultiHeadAttention(num_heads=3, key_dim=32)(x, x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Adapter(bottleneck_dim=32, name="adapter")(x)
    out = Dense(5, activation='softmax', kernel_regularizer=l2(0.005), name="out")(x)
    return Model(inputs=inp, outputs=out)

# OVESAMPLING SOLO SULLE CLASSI MINORITARIE
def oversample_minority(X, Y, target_ratio=1.0, random_state=42):
    # 1) array con NumPy
    X = np.array(X)
    Y = np.array(Y)

    counts = Counter(Y)
    max_count = max(counts.values())
    X_blocks, Y_blocks = [], []

    for cls, cnt in counts.items():
        # 2) prendo gli indici della classe minoritaria
        idx_cls = np.where(Y == cls)[0]
        X_cls, Y_cls = X[idx_cls], Y[idx_cls]

        # 3) tramite target_ratio si decide quanti campioni vogliamo
        n_desired = int(target_ratio * max_count)
        if cnt < n_desired:
            # oversample con replacement
            X_up, Y_up = resample(
                X_cls, Y_cls,
                replace=True,
                n_samples=n_desired,
                random_state=random_state
            )
            X_blocks.append(X_up)
            Y_blocks.append(Y_up)
        else:
            X_blocks.append(X_cls)
            Y_blocks.append(Y_cls)

    # 4) concatenazione e shuffling
    X_res = np.vstack(X_blocks)
    Y_res = np.concatenate(Y_blocks)
    p = np.random.permutation(len(Y_res))
    return X_res[p], Y_res[p]

# OVERSAMPLING DUPLICATIVO GLOBALE <- ho usato questa versione
def oversample_duplicative(X, Y):
    unique_classes, counts = np.unique(Y, return_counts=True)
    max_count = max(counts)
    X_list = []
    Y_list = []
    for cls in unique_classes:
        indices = [i for i, y in enumerate(Y) if y == cls]
        X_cls = X[indices]
        Y_cls = [cls] * len(indices)
        replicates_needed = max_count - len(indices)
        if replicates_needed > 0:
            replicated_indices = np.random.choice(len(indices), size=replicates_needed, replace=True)
            X_rep = X_cls[replicated_indices]
            Y_rep = [cls] * replicates_needed
            X_cls = np.concatenate([X_cls, X_rep], axis=0)
            Y_cls = Y_cls + Y_rep
        X_list.append(X_cls)
        Y_list.extend(Y_cls)
    X_res = np.concatenate(X_list, axis=0)
    Y_res = np.array(Y_list)
    indices = np.arange(len(Y_res))
    np.random.shuffle(indices)
    return X_res[indices], Y_res[indices]

# FINE-TUNING e TRANSFER LEARNING
def fine_tune_tl(records,
                 labels,
                 tokenizer,
                 max_len,
                 base_model_path,
                 en2fr, fr2en, en2de, de2en,
                 w2v_model,
                 random_state=42,
                 n_epochs=5,
                 val_split=0.2,
                 pct_bt=1.0): # modificare valore per decidere quanta porzione di back-translation fare

    # 1) Preparazione random_seed
    random.seed(random_state)
    np.random.seed(random_state)

    # 2) Back-translation su pct_bt * len(records)
    n = len(records)
    k = max(1, int(n * pct_bt))
    idx_bt = random.sample(range(n), k)

    bt_texts, bt_labels = [], []
    for i in idx_bt:
        text, lab = records[i], labels[i]
        fr     = en2fr([text],   max_length=250)[0]['translation_text']
        re_fr  = fr2en([fr],     max_length=250)[0]['translation_text']
        de     = en2de([text],   max_length=250)[0]['translation_text']
        re_de  = de2en([de],     max_length=250)[0]['translation_text']
        bt_texts.extend([re_fr, re_de])
        bt_labels.extend([lab, lab])

    # 3) EDA + W2V augment
    aug_recs, aug_labs = [], []
    for text, lab in zip(records + bt_texts, labels + bt_labels):
        for aug in combined_augment(text):
            aug_recs.append(aug); aug_labs.append(lab)
        aug2 = w2v_augment(text, w2v_model, p=0.15, topn=8)
        aug_recs.append(aug2); aug_labs.append(lab)

    # 4) Oversample duplicativo
    X = np.array(records + bt_texts + aug_recs)
    Y = np.array(labels  + bt_labels + aug_labs)
    unique, counts = np.unique(Y, return_counts=True)
    max_c = counts.max()
    X_blocks, Y_blocks = [], []
    for cls in unique:
        idxs = np.where(Y == cls)[0]
        X_cls, Y_cls = X[idxs], Y[idxs]
        reps = max_c - len(idxs)
        if reps > 0:
            sel = np.random.choice(len(idxs), size=reps, replace=True)
            X_blocks.append(np.concatenate([X_cls, X_cls[sel]]))
            Y_blocks.append(np.concatenate([Y_cls, Y_cls[sel]]))
        else:
            X_blocks.append(X_cls); Y_blocks.append(Y_cls)
    X_res = np.concatenate(X_blocks)
    Y_res = np.concatenate(Y_blocks)
    perm = np.random.permutation(len(Y_res))
    X_res, Y_res = X_res[perm], Y_res[perm]

    # 5) Tokenizzo e split stratificato
    prots = [preprocess(r) for r in X_res]
    seqs  = tokenizer.texts_to_sequences(prots)
    X_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    y_cat = to_categorical(Y_res, num_classes=5)

    X_ft, X_val_ft, y_ft, y_val_ft = train_test_split(
        X_pad, y_cat,
        test_size=val_split,
        random_state=random_state,
        stratify=Y_res
    )

    # 6) Carico il modello base
    model = load_model(base_model_path,
                       custom_objects={'Adapter': Adapter,
                                       'WarmUpCosineDecay': WarmUpCosineDecay})

    # 7) Congelo solo embedding + primo Bi-LSTM
    for layer in model.layers:
        if layer.name in ("embedding", "bidirectional"):
            layer.trainable = False
        else:
            layer.trainable = True

    # 8) Primo fit solo sulla head
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )
    model.summary()
    history1 = model.fit(
        X_ft, y_ft,
        epochs=n_epochs//2,
        validation_data=(X_val_ft, y_val_ft),
        callbacks=[
          EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
          #ModelCheckpoint('bt_stage1.keras', save_best_only=True)
        ],
        verbose=1
    )

    # 9) Sblocco tutto il modello e rifinisco
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3),
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=['accuracy']
    )
    history2 = model.fit(
        X_ft, y_ft,
        epochs=n_epochs - n_epochs // 2,
        validation_data=(X_val_ft, y_val_ft),
        callbacks=[
          EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
          ModelCheckpoint('best_model_finetuned.keras', save_best_only=True)
        ],
        verbose=1
    )

    return history1, history2, model


# MAIN del core

if __name__ == "__main__":

    # Lettura del dataset
    with open('dataset_eng.txt','r',encoding='utf-8') as f:
        lines = f.readlines()[1:]

    raw = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts)>=2 and parts[1].isdigit():
            raw.append(int(parts[1]))
    print("Tutte le etichette raw nel dataset:", sorted(set(raw)))
    print("Frequenze:", Counter(raw))

    recensioni, valori = [], []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts)>=2 and parts[1].isdigit():
            lbl = int(parts[1])
            # intervallo 0-4
            if 0 <= lbl <= 4:
                recensioni.append(parts[0])
                valori.append(lbl)
    print("Esempi di label caricati:", sorted(set(valori)))
    print("Dataset letto correttamente! ✓")

    # Back-translation -> fondamentale per aver ottenuto val_accuracy > 90%
    print("Inizializzo back-translation pipelines…")
    en2fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=-1)
    fr2en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en", device=-1)
    en2de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", device=-1)
    de2en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en", device=-1)

    pct_bt = 0.5 # valore per andare a gestire la quantità di dataset su cui applicare back-translation (per ora con 0.5 ottengo val_accuracy > 90%)
    n, k = len(recensioni), max(1, int(len(recensioni) * pct_bt))
    idx_bt = random.sample(range(n), k)
    batch_size, num_batches = 32, ceil(k / 32)
    bt_reviews, bt_labels = [], []

    for b in range(num_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, k)
        batch_idx = idx_bt[start:end]
        orig_texts = [recensioni[i] for i in batch_idx]
        orig_labels = [valori[i] for i in batch_idx]

        # EN → FR → EN
        fr_texts = [tr['translation_text'] for tr in en2fr(orig_texts, max_length=200)]
        back_fr = [tr['translation_text'] for tr in fr2en(fr_texts, max_length=200)]

        # EN → DE → EN
        de_texts = [tr['translation_text'] for tr in en2de(orig_texts, max_length=200)]
        back_de = [tr['translation_text'] for tr in de2en(de_texts, max_length=200)]

        bt_reviews.extend(back_fr + back_de)
        bt_labels.extend(orig_labels * 2)
        #print(f" Batch {b+1}/{num_batches} completato")

    recensioni += bt_reviews
    valori    += bt_labels
    print("Back-translation finita, nuova dimensione dataset:", len(recensioni))

    # Pre-processing e Word2Vec
    print("Pre-processing del dataset in corso…")
    processed = [preprocess(r) for r in recensioni]
    print("Pre-processing completato! ✓")

    print("Generazione e caricamento modello Word2Vec...")
    if os.path.exists("Word2Vec.bin"):
        w2v_model = Word2Vec.load("Word2Vec.bin")
    else:
        w2v_model = Word2Vec(processed, vector_size=200, window=10, min_count=2, workers=4, epochs=10)
        w2v_model.save("Word2Vec.bin")
    print("Modello Word2Vec caricato correttamente! ✓")

    # Tokenizer e embedding matrix
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(recensioni)
    sequences = tokenizer.texts_to_sequences(recensioni)
    max_len = 200
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    vocab_size = len(tokenizer.word_index) + 1
    emb_dim    = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    for w, i in tokenizer.word_index.items():
        if w in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[w]

    # Salva tokenizer e embedding
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    np.save("embedding_matrix.npy", embedding_matrix)
    print("Matrice di embedding creata! ✓")

    # Split train/val/test e oversampling
    X_tv, X_test, y_tv, y_test = train_test_split(padded, valori, test_size=0.15, random_state=seed_value)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765, random_state=seed_value)

    X_train_res, y_train_res = oversample_duplicative(X_train, y_train)
    y_train_cat = to_categorical(y_train_res, num_classes=5)
    y_val_cat   = to_categorical(y_val, num_classes=5)
    y_test_cat  = to_categorical(y_test, num_classes=5)     

    # Costruzione e training modello
    steps_per_epoch = len(X_train_res) // 64
    total_steps     = 50 * steps_per_epoch
    warmup_steps    = int(0.1 * total_steps)
    lr_schedule     = WarmUpCosineDecay(1e-3, total_steps, warmup_steps)

    model = build_model(vocab_size, emb_dim, embedding_matrix, max_len, embed_trainable=True)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5, clipnorm=1.0),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir="logs", histogram_freq=1)
    ]

    # gestisco i pesi delle classi
    weights = compute_class_weight('balanced',
                               classes=np.unique(y_train_res),
                               y=y_train_res)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

    # genera sample_weight per il train con un vettore con etichette codificate in one-hot encoding
    sample_weight = np.array([
        class_weight_dict.get(int(lbl), 1.0)
        for lbl in y_train_res
    ])
    model.summary()
    history = model.fit(
        X_train_res, y_train_cat,
        epochs=50, batch_size=64,
        sample_weight=sample_weight,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=2
    )

    # Salvataggio pesi e valutazione
    model.save("best_model.keras")
    print("Training iniziale completato e modello salvato come 'best_model.keras'.")

    # Valutazione sul test set
    print("Valutazione modello sul test set:")
    score = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Test Accuracy:", score[1])

    # Confusion matrix e report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)
   

    print("\nEsempi di classificazione:")
    test_texts = tokenizer.sequences_to_texts(X_test.tolist())
    for i in range(5):
        print("Review:", test_texts[i])
        print("True Label:", y_test[i])
        print("Predicted Label:", y_pred[i])
        print("------")

    # 3) 3-Fold Cross-Validation per robustezza -> momentaneamente commentata per brevità
    '''
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    all_histories = []
    all_conf_mats = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(padded_sequences, all_values), 1):
        print(f"\n--- Fold {fold} ---")
        # split dei dati
        X_tr, X_val_cv = padded_sequences[train_idx], padded_sequences[val_idx]
        y_tr, y_val_cv = np.array(all_values)[train_idx], np.array(all_values)[val_idx]
        y_tr_cat = to_categorical(y_tr, num_classes=5)
        y_val_cat = to_categorical(y_val_cv, num_classes=5)

        # build del modello
        def build_model_cv():
            inp = Input(shape=(max_len,))
            x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=True)(inp)
            x = GaussianNoise(0.2)(x)
            x = SpatialDropout1D(0.3)(x)
            x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
            x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(x)
            x = MultiHeadAttention(num_heads=3, key_dim=32)(x, x)
            x = Dropout(0.5)(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.4)(x)
            out = Dense(5, activation='softmax')(x)
            m = Model(inp, out)

            # learning-rate schedule
            steps = (len(X_tr) // 64) * 20   # ipotizziamo max 20 epoche per CV
            warmup = int(0.1 * steps)
            lr_sched = WarmUpCosineDecay(initial_lr=1e-3, total_steps=steps, warmup_steps=warmup)

            m.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3),
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched),
                metrics=['accuracy']
            )
            return m

        # creazione dei modelli per cross validation
        model_cv = build_model_cv()

        # callback di EarlyStopping sul validation loss
        es_cv = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        # allenamento
        history = model_cv.fit(
            X_tr, y_tr_cat,
            validation_data=(X_val_cv, y_val_cat),
            epochs=20,
            batch_size=64,
            verbose=2,
            callbacks=[es_cv]
        )
        all_histories.append(history)

    # confusion matrix di questo fold
    y_pred = np.argmax(model_cv.predict(X_val_cv), axis=1)
    cm_fold = confusion_matrix(y_val_cv, y_pred)
    all_conf_mats.append(cm_fold)

    # Riepilogo performance cross-validation
    cv_scores = [h.history['val_accuracy'][np.argmin(h.history['val_loss'])] for h in all_histories]
    print(f"CV Val Accuracy: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")'''

    # Plot delle curve principali del modello
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Accuracy Curve')
    plt.tight_layout(), plt.show()
