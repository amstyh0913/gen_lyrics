'''Restore a character-level sequence to sequence model from disk and use it
to generate predictions.

This script loads the s2s.h5 model saved by lstm_seq2seq.py and generates
sequences from it.  It assumes that no changes have been made (for example:
latent_dim is unchanged, and the input data and model architecture are unchanged).

See lstm_seq2seq.py for more details on the model architecture and how
it is trained.
'''
from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.backend import print_tensor, get_value
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import gc
from collections import defaultdict
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--token', choices=['word', 'char'],type=str, default='word')
parser.add_argument('--bs',type=int, default=64)
parser.add_argument('--epoch',type=int, default=50)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--num',type=int, default=100000)
parser.add_argument('--min',type=int, default=10)
args = parser.parse_args()

batch_size = args.bs  # Batch size for training.
epochs = args.epoch  # Number of epochs to train for.
latent_dim = args.dim  # Latent dimensionality of the encoding space.
num_samples = args.num  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'train_data_nn_vb_prp_jj.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_vocab = set()
target_vocab = set()
input2count = defaultdict(int)
output2count = defaultdict(int)
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    # print(line.split('\t'))
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    # input_texts.append(input_text)
    # target_texts.append(target_text)
    if args.token == 'char':
        for char in input_text:
            if char not in input_vocab:
                input2count[char] += 1
                # input_vocab.add(char)
        for char in target_text:
            if char not in target_vocab:
                output2count[char] += 1
                # target_vocab.add(char)
        input_texts.append(input_text)
        target_texts.append(target_text)
    else:
        input_words = input_text.split()
        target_words = target_text.split()
        # start sequence and end sequence
        target_words.insert(0, '\t')
        target_words.append('\n')
        # print(target_words)
        if len(input_words) <= 5 and len(target_words) <= 15:
            for word in input_words:
                if word not in input_vocab:
                    input2count[word] += 1
                    # input_vocab.add(word)
            for word in target_words:
                if word not in target_vocab:
                    output2count[word] += 1
                    # target_vocab.add(word)
            input_texts.append(input_text)
            target_texts.append(target_text)
            if len(input_texts) == 40000:
                break

del lines
gc.collect()

# vocabulary
input_vocab  = [ word for word, count in input2count.items()  if count > args.min]
target_vocab = [ word for word, count in output2count.items() if count > args.min]
input_vocab.append('unk')
target_vocab.append('unk')
input_vocab = sorted(list(input_vocab))
target_vocab = sorted(list(target_vocab))
num_encoder_tokens = len(input_vocab)
num_decoder_tokens = len(target_vocab)

f = open('test_input_vocab.json','w')
json.dump(input_vocab, f, indent = 2)
f.close()

# max input sequence length
if args.token == 'char':
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
else:
    max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
    max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# char to id
input_token_index = dict(
    [(token, i) for i, token in enumerate(input_vocab)])
target_token_index = dict(
    [(token, i) for i, token in enumerate(target_vocab)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(input_text):
    
    if args.token == 'char':
        input_tokens = list(input_text)
    else:
        input_tokens = input_text.split()

    for t, token in enumerate(input_tokens):
        if token in input_token_index:
            encoder_input_data[i, t, input_token_index[token]] = 1.
        else:
            encoder_input_data[i, t, input_token_index['unk']] = 1.

# Restore the model and construct the encoder and decoder.
model = load_model('nn_vb_prp_jj_models/seq2seq_17-2.09.hdf5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
