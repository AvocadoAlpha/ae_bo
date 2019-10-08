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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np

script_name = os.path.basename(__file__).split('.')[0]

x_train, x_val, x_test = utils.generate_data_cnn()
print("train: ",len(x_train))
print("val: ",len(x_val))
print("test: ",len(x_test))

# space for hyperopt to search quniform is discrete uniform (lower bound, upper bound, step) lower bound always zero whatever you set it
# batch size is fixed could be changed anytime to quniform for example
space = {
    'units1': hp.quniform('units1', 0, 10000, 50), #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    'batch_size': hp.choice('batch_size', [128])
    }

space_str = """
space = {
    'units1': hp.quniform('units1', 0, 10000, 10), 
    'batch_size': hp.choice('batch_size', [128])
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

    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, decoded)
    model.compile(loss='mean_squared_error', optimizer='adadelta')
    model.fit(x_train, x_train,
                    epochs=100,
                    batch_size=int(params['batch_size']),
                    shuffle=True,
                    validation_data = (x_val, x_val),
                    callbacks =utils.callback(script_name))

    preds = model.predict(x_test)
    loss_ = model.evaluate(x=x_test, y=preds)
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