import os
from utils import set_gpu
import sys
set_gpu(sys.argv)
from keras.models import Model
from keras.layers import Dense, Input
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K
import numpy as np
import os
import utils
import numpy as np

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_val, x_test = utils.generate_data_medium2()

def node_params(n_layers):
    # define the parameters that are conditional on the number of layers here
    # in this case the number of nodes in each layer
    params = {}
    for n in range(n_layers):
        params['n_nodes_layer_{}'.format(n)] = hp.quniform('n_nodes_{}_{}'.format(n_layers, n), 0, 100, 4)
    return params

# list of the number of layers you want to consider
layer_options = [1,2]

space = {'choice': hp.choice('layers', [node_params(n) for n in layer_options]),
        'batch_size': hp.choice('batch_size', [128])
         }

space_str = """
space = {'choice': hp.choice('layers', [node_params(n) for n in layer_options]),
        'batch_size': hp.choice('batch_size', [128])
         }"""


def objective(params):


    for x in params.keys():
        if params[x] == 0:
            params[x] = 1
        elif x =='choice':
            for y in params[x].keys():
                if params[x][y] == 0:
                    params[x][y] = 1

    K.clear_session()

    print('Params testing: ', params)
    print('\n ')

    layersAndNodes = [x/100 for x in list(params['choice'].values())] # len is number of layers


    input = Input(shape=(784,))
    next_layer = 784
    prev_layer = input


    output_layers = []

    for x in layersAndNodes:
        next_layer = int(np.ceil(x * next_layer))
        output_layers.append(next_layer)
        enc = Dense(next_layer, activation='relu')(prev_layer)
        prev_layer = enc


    output_layers.reverse()
    for x in output_layers[1:]:
        dec = Dense(x, activation='relu')(prev_layer)
        prev_layer = dec

    dec2 = Dense(784, activation='sigmoid')(prev_layer)
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

    print('Params tested: ', params)
    print('\n ')
    print(model.summary())
    return {'loss': score, 'status': STATUS_OK}





# loop indefinitely and stop whenever you like
if __name__ == "__main__":
    while True:
        utils.run_trials_grid_2(script_name, space, objective)