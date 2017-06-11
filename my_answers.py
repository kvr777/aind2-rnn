import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # define the position when we should stop
    last_pos = len(series) - window_size

    # create X and y dataset, beginning with the first position of serie
    for i in range(last_pos):
        # define interval we have to extract from series to create X dataset and extract data
        start_int = i
        end_int = start_int + window_size
        X_data = series[start_int:end_int]

        # y data is the next point in serie data
        y_data = series[end_int]

        # add extracted data to list
        X.append(X_data)
        y.append(y_data)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.summary()

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    # import necessary library
    import string

    # build list of good symbols
    good_signs = set([' ', '!', ',', '.', ':', ';', '?'])
    good_symbols = good_signs.union(string.ascii_lowercase)

    # find symbols to remove. Print them for control
    bad_symb_list = list(good_symbols.symmetric_difference(set(text)))
    print(bad_symb_list)

    # remove as many non-english characters and character sequences as you can
    ## replace the char from list to spaces
    for char in bad_symb_list:
        text = text.replace(char, ' ')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # define the position when we should stop
    last_pos = len(text) - window_size

    # create X and y dataset, beginning with the first position of serie
    for i in range(0, last_pos, step_size):
        # define interval we have to extract from series to create X dataset and extract data
        start_int = i
        end_int = start_int + window_size
        inputs_data = text[start_int:end_int]

        # y data is the next point in serie data
        outputs_data = text[end_int]

        # add extracted data to list
        inputs.append(inputs_data)
        outputs.append(outputs_data)

    
    return inputs,outputs
