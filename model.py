import os
import re
import unicodedata

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

EMBEDDING_DIM = 256
UNITS = 1024
BATCH_SIZE = 64


def gru():
    # If you have a GPU,
    # we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(UNITS,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    return tf.keras.layers.GRU(UNITS,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]
                  for l in lines[:num_examples]]

    return word_pairs


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset():
    # Download the file
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip',
        origin='http://download.tensorflow.org/data/spa-eng.zip',
        extract=True)

    path = os.path.join(os.path.dirname(path_to_zip), 'spa-eng', 'spa.txt')

    num_examples = 30000

    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)

    # Vectorize the input and target languages

    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')]
                    for en, sp in pairs]

    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')]
                     for en, sp in pairs]

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp = max_length(input_tensor)
    max_length_tar = max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = sequence.pad_sequences(input_tensor,
                                          maxlen=max_length_inp,
                                          padding='post')

    target_tensor = sequence.pad_sequences(target_tensor,
                                           maxlen=max_length_tar,
                                           padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang, \
        max_length_inp, max_length_tar


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = gru()

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((BATCH_SIZE, UNITS))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = gru()
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(UNITS)
        self.W2 = tf.keras.layers.Dense(UNITS)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through
        # embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after
        # concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((BATCH_SIZE, UNITS))


def create_encoder(inp_lang):
    vocab_inp_size = len(inp_lang.word2idx)
    return Encoder(vocab_inp_size)


def create_decoder(targ_lang):
    vocab_tar_size = len(targ_lang.word2idx)
    return Decoder(vocab_tar_size)


def create_optimizer():
    return tf.train.AdamOptimizer()


def create_checkpoint(optimizer, encoder, decoder):
    return './training_checkpoints', tf.train.Checkpoint(optimizer=optimizer,
                                                         encoder=encoder,
                                                         decoder=decoder)
