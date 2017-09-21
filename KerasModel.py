# python KerasModel class

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import linkage_tools # for the Linker superclass

# Wrapper for keras model class definition
class KerasModel(linkage_tools.Linker):

    # initialise the neural network
    def __init__(self, features, hiddennodes, outputnodes, **keywords):
        
        # Set a classifier descriptors
        #self.classifier_type = 'keras model wrapper'
        self.features = features
        if ('learningrate' in keywords):
            self.lr = keywords['learningrate'] # Learning rate
        else:
            self.lr = 0.1

        # set number of nodes in each input, hidden, output layer
        self.model = models.Sequential()
        self.model.add(layers.Dense(hiddennodes, activation='relu', input_shape=(len(features),)))
        self.model.add(layers.Dense(hiddennodes, activation='relu'))
        self.model.add(layers.Dense(outputnodes, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=self.lr),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        # Initialize performance lists
        self = linkage_tools.Linker.init_lists(self)

    # train the neural network
    def train(self, inputs_list, truth_list, guess_list):
                
        inputs = np.array(inputs_list, ndmin=2, dtype=np.float)
        truth = np.squeeze(truth_list)
        self.model.train_on_batch(inputs, truth)
        # self.model.fit(x_train, y_train, epochs=4, batch_size=512)
        
    # query the neural network
    def query(self, inputs_list, input_ids):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2, dtype=np.float)

        results = self.model.predict(inputs)        
        
        # Convert to linkage score to linkage 'cost'
        score_cost = abs(results - results.max())

        # Maximize pair-wise linkage scores (minimize cost with Munkres)
        link_list = input_ids.assign(linkage_cost = score_cost)
        winner_ind = self.maximize(link_list)

        # Return boolean array
        winners = np.zeros(len(link_list), np.bool)
        winners[winner_ind] = 1
        
        return winners, results
