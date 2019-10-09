import sys
from utils import set_gpu
from utils import set_gpu
import sys
set_gpu(sys.argv)
from keras.models import Model
from keras.layers import Dense, Input
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K
import os
import utils
import inspect
import numpy as np
from keras import regularizers

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_val, x_test = utils.generate_data_medium_2()

space = {
    'units1': hp.quniform('units1', 0, 100, 20), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'units2': hp.quniform('units2', 0, 100, 20), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'batch_size': hp.choice('batch_size', [128])
    }

space_str = """
space = {
    'units1': hp.uniform('units1', 0, 1), 
    'units2': hp.uniform('units2', 0, 1), 
    'batch_size': hp.choice('batch_size', [128])
    }"""


def objective(params):

    for x in params.keys():
        if params[x] == 0:
            params[x] = 1

    K.clear_session()

    print('Params tested: ', params)
    print('\n ')


    layer1 = int(np.ceil(params['units1']/100 * 784))
    layer2 = int(np.ceil(params['units2']/100 * layer1))

    input = Input(shape=(784,))
    enc = Dense(layer1, activation='relu',kernel_regularizer=regularizers.l2(0.001))(input)
    enc2 = Dense(layer2, activation='relu',kernel_regularizer=regu.l2(0.001))(enc)
    dec1 = Dense(layer1, activation='relu',kernel_regularizer=regularizers.l2(0.001))(enc2)
    dec2 = Dense(784, activation='sigmoid')(dec1)
    model = Model(input, dec2)

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
    score =round(sess.run(loss)/len(x_test), 4)

    print('Params testing: ', params)
    print('\n ')
    print(model.summary())
    return {'loss': score, 'status': STATUS_OK}





# loop indefinitely and stop whenever you like
if __name__ == "__main__":
    while True:
        utils.run_trials_grid_2(script_name, space, objective)