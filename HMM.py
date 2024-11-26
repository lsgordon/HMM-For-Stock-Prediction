"""
Hidden Markov Model class
This file contains the source code for the hidden markov model for this assignment
"""
import numpy as np
import pandas as pd
np.random.seed(42)
class HiddenMarkovModel:
    def __init__(self,observation_sequence,number_states) -> None:

        # first step of the model is to initialize all of the parameters to random states
        self.transition_matrix = np.random.rand(number_states,number_states)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=0, keepdims=True)

        # must do everything with log probabilities to prevent underflow
        self.transition_matrix = np.log(self.transition_matrix)

        # conditional probabilities of seeing some value given state
        self.emission_matrix = np.random.rand(number_states,len(np.unique(observation_sequence)))
        self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1, keepdims=True)

        # must do everything with log probabilities to prevent underflow
        self.emission_matrix = np.log(self.emission_matrix)

        # pi distribution, the starting probabilities
        pi = np.random.rand(number_states,1)
        pi = pi / pi.sum()

        pi = np.log(pi)
        print(self.emission_matrix,self.transition_matrix)
        # random initial state sequence

        return
    def forward_algorithm(self,observations,end_st): 
        """
        Forward-backward Algorithm
        """
        fwd = []
        for i, observation_i in enumerate(observations):
            f_curr = {}
            for st in range(self.emission_matrix.shape(0)):
                if i == 0:
                  prev_f_sum = start_prob[st]  
    

if __name__ == "__main__":
    df = pd.read_csv('Data/input.csv')
    model = HiddenMarkovModel(df['decile'],4)