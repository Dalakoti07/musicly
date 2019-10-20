import os
import json
import argparse

import numpy as np

from model import build_model, save_weights
from keras.utils import plot_model

DATA_DIR = './data'
LOG_DIR = './logs'

BATCH_SIZE = 16 # batch size of 16 means that each batch would have 16 rows/datsets/records/datapoints ,like in image batches
SEQ_LENGTH = 64 # seq_length means that each row in batch would contain 64 characters
# one batch X would be of size 16*64 and corresponding Y batch would be 16*64*noOfUniqueChars ,where last dims is vocab size, characters are represented as one-hot-encoded vectors

# python class that would log the training details
class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

# this function make batches
def read_batches(T, vocab_size):
    length = T.shape[0]; #129,665, total number of characters in the text corpus
    batch_chars = int(length / BATCH_SIZE); # 8,104 , number of characters in one rows of all batches, characters in 1st row of batch 1 + batch 2 + B3 + B4 +.... Bk would be 8104, similarly characters of row 2 of all batches sum to 8104
    # nuber of batches would be formed is 8104/64 is 126
    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # (0, 8040, 64), this line essentially is counting over number of batches
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 16X64
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) # 16X64X86
        for batch_idx in range(0, BATCH_SIZE): # (0,16)
            for i in range(0, SEQ_LENGTH): #(0,64)
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # setting value of X[batch_idx][i] from text corpus
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1 # setting the value of next character as 1
        yield X, Y
# ci is the character i at text T
# batch 1 row 1 is c0,c1,c2,........c63  
# batch 1 row 2 is c8104,c8105,.............c8167
# batch 1 row 3 is c16208,c16209,...............,c16271
# ..... we have 16 rows for all batches, and last row is 129664 ........ somewhere 
# corresponding y would be same it would have extra dim telling the next character after it and thus Y would be tensor, whereas X is matrix

# batch 2 ,start=64 ,then row 1 is c64,c65,.............,c127
# batch 2 row 2 is c8168,c8169,.....................,c8123
# ........we have 16 rows batch 2 and last row is ...........

# creting these types of batches is beneficial.

def train(text, epochs=100, save_freq=10):

    # character to index and vice-versa mappings
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of unique characters: " + str(len(char_to_idx))) #86

    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    #model_architecture
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    # vizualizing the model in form of png
    plot_model(model, to_file='model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train data generation
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) #convert complete text into numerical indices

    print("Length of text:" + str(T.size)) #129,665

    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  # 12, didn't used anywhere

    log = TrainLogger('training_log.csv') # making a csv file to the log the training

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], [] # they contain loss and accuracy for each batch of all epochs , length of losses would be 128*100 
        # reading the batches and training on each batch sequentially
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)
        # entries in .csv would be no. of epoches
        log.add_entry(np.average(losses), np.average(accs))

        # saving the weights at frequency
        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs, args.freq)
