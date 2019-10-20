import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding


MODEL_DIR = './model'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def build_model(batch_size, seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    for i in range(3):
        # return_sequence true, means that at each and every timestamp we want output at each timestamp
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        # stateful=true means output of one batch would be treated as input to next batch
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    # the model consist of three LSTM hidden layers ,where each layer has 256 LSTM blocks or neurons
    # time distributed means output of each layer would be collected and a dense vector is made out of it
    model.add(TimeDistributed(Dense(vocab_size))) 
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = build_model(16, 64, 50)
    model.summary()
