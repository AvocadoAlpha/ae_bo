import sys
from utils import set_gpu
import sys
set_gpu(sys.argv)
from keras.models import Model
from keras.layers import Dense, Input
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K
from keras import regularizers
import os
import utils
import numpy as np

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_test, x_val = utils.generate_data_micro()
print("train: ",len(x_train))
print("val: ",len(x_val))
print("test: ",len(x_test))

# space for hyperopt to search quniform is discrete uniform (lower bound, upper bound, step) lower bound always zero whatever you set it
# batch size is fixed could be changed anytime to quniform for example
space = {
    'units1': hp.quniform('units1', 0, 25000, 1500), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'batch_size': hp.choice('batch_size', [30])
    }

space_str = """
space = {
    'units1': hp.quniform('units1', 0, 100000, 1500), 
    'batch_size': hp.choice('batch_size', [30])
    }"""



def objective(params):
    # params is dict with parameters chosen by BO params = {"batch_size": 30, "units1": 49}

    for x in params.keys(): # if "units1":0 add one -> units1:1
        if params[x] == 0:
            params[x] = 1

    K.clear_session()

    print('Params testing: ', params)
    print('\n ')

    layer1 = int(params['units1'])

    #one hidden layer keras functional API

    input = Input(shape=(784,))
    enc = Dense(layer1, activation='relu')(input)
    dec = Dense(784, activation='sigmoid')(enc)
    model = Model(input, dec)

    model.compile(loss='mean_squared_error', optimizer='adadelta')
    model.fit(x_train, x_train,
                    epochs=100,
                    batch_size=int(params['batch_size']),
                    shuffle=True,
                    validation_data = (x_val, x_val),
                    callbacks =utils.callback(script_name))

    preds = model.predict(x_test)
    #loss_ = model.evaluate(x=x_test, y=preds)
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