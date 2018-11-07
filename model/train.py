'''Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
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

f = open('train_input_vocab.json','w')
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
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    
    if args.token == 'char':
        input_tokens = list(input_text)
        target_tokens = list(target_text)
    else:
        input_tokens = input_text.split()
        target_tokens = target_text.split()

    for t, token in enumerate(input_tokens):
        if token in input_token_index:
            encoder_input_data[i, t, input_token_index[token]] = 1.
        else:
            encoder_input_data[i, t, input_token_index['unk']] = 1.

    for t, token in enumerate(target_tokens):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if token in target_token_index:
            decoder_input_data[i, t, target_token_index[token]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[token]] = 1.
        else:
            decoder_input_data[i, t, target_token_index['unk']] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index['unk']] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

callbacks = []
fpath = 'nn_vb_prp_jj_models/seq2seq_{epoch:02d}-{val_loss:.2f}.hdf5'
callbacks.append(ModelCheckpoint(filepath = fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=callbacks)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, token) for token, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, token) for token, i in target_token_index.items())

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
