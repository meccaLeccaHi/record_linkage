# python KerasModel class

import numpy as np
#import scipy.special # for the sigmoid function expit()
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
        
        self.train_precision_list = []
        self.train_recall_list = []
        self.train_fscore_list = []
        self.train_accuracy_list = []

        self.val_precision_list = []
        self.val_recall_list = []
        self.val_fscore_list = []
        self.val_accuracy_list = []

    # train the neural network
    def train(self, inputs_list, truth_list, guess_list):
        
        # self.model.fit(x_train, y_train, epochs=4, batch_size=512)
        
        record_list = inputs_list*.98+.01 # scale inputs to between .01 and .99
        truth_list = truth_list*.98+.01
        for record,truth in zip(record_list.values.tolist(),truth_list):

            # convert inputs list to 2d array
            inputs = np.array(record, ndmin=2, dtype=np.float).T
            truth = np.array(truth, ndmin=2, dtype=np.float)

            # calculate signals into hidden layer
            hidden_inputs = np.dot(self.wih, inputs)
            # calculate the signals emerging from hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)

            # calculate signals into final output layer
            final_inputs = np.dot(self.who, hidden_outputs)
            # calculate the signals emerging from final output layer
            final_outputs = self.activation_function(final_inputs)

            # output layer error is the (truth - guesses)
            output_errors = truth - final_outputs
            # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            hidden_errors = np.dot(self.who.T, output_errors) 

            # update the weights for the links between the hidden and output layers
            self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

            # update the weights for the links between the input and hidden layers
            self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

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
