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
        transition_matrix = np.random.rand(number_states,number_states)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)

        # conditional probabilities of seeing some value given state
        emission_matrix = np.random.rand(number_states,len(np.unique(observation_sequence)))
        emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

        # pi distribution
        pi = np.random.rand(number_states,1)
        pi = pi / pi.sum()

        print(emission_matrix,transition_matrix)
        # random initial state sequence

        pass
    

if __name__ == "__main__":
    df = pd.read_csv('Data/input.csv')
    model = HiddenMarkovModel(df['decile'],8)