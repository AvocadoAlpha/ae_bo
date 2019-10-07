import sys
from utils import set_gpu
import sys
set_gpu(sys.argv)
from keras.models import Model
from keras.layers import Dense, Input
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import mean_squared_error
import tensorflow as tf
from keras import regularizers
from keras import backend as K
import os
import utils
import numpy as np

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_val, x_test = utils.generate_data_small()

space = {
    'units1': hp.choice('units1', [80]),
    'batch_size': hp.quniform('batch_size', 0, 1000, 25)  #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    }

space_str = """
space = {
    'units1': hp.choice('units1', [180]),
    'batch_size': hp.quniform('batch_size', 0, 1000, 25)  
    }"""


def objective(params):


    for x in params.keys():
        if params[x] == 0:
            params[x] = 1

    K.clear_session()

    print('Params testing: ', params)
    print('\n ')

    layer1 = int(params['units1'])



    input = Input(shape=(784,))

    enc = Dense(layer1, activation='relu')(input)
    dec = Dense(784, activation='sigmoid')(enc)
    model = Model(input, dec)

    encoder = Model(input, enc)

    model.compile(loss='mean_squared_error', optimizer='adadelta')
    model.fit(x_train, x_train,
                    epochs=100,
                    batch_size=int(params['batch_size']),
                    shuffle=True,
                    validation_data = (x_val, x_val),
                    callbacks =utils.callback(script_name))

    preds = model.predict(x_test)
    loss = tf.keras.backend.sum(mean_squared_error(tf.convert_to_tensor(x_test), tf.convert_to_tensor(preds)))
    sess = tf.Session()
    loss = sess.run(loss)
    score = round(loss / (len(x_test)), 4)

    print('Params tested: ', params)
    print('\n ')
    print(model.summary())
    return {'loss': score, 'status': STATUS_OK}





# loop indefinitely and stop whenever you like
if __name__ == "__main__":
    while True:
        utils.run_trials_grid_2(script_name, space, objective)