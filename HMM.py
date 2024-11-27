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
        self.pi = np.random.rand(number_states,1)
        self.pi = self.pi / self.pi.sum()

        self.pi = np.log(self.pi)
        print(self.emission_matrix,self.transition_matrix)
        # random initial state sequence

        return
    def forward_algorithm(self,observations,end_st): 
        """
        Forward-backward Algorithm, ported from:
        https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Pseudocode
        """
        fwd = []
        for i, observation_i in enumerate(observations):
            f_curr = {}
            for st in range(self.emission_matrix.shape[0]):

                # base case for the algorithm
                if i == 0:
                    prev_f_sum = self.pi[st]
                
                # else use the inductive definition here
                else:
                    probs = [f_prev[k] + self.transition_matrix[k][st] for k in range(self.emission_matrix.shape[0])]
                    prev_f_sum = np.log(sum(np.exp(probs))) 

                f_curr[st] = self.emission_matrix[st][observation_i-1] + prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr
        probs = [f_curr[k] + self.transition_matrix[k][end_st] for k in range(self.emission_matrix.shape[0])]
        p_fwd = np.log(sum(np.exp(probs)))
        print(fwd)
        print(p_fwd)
        bwd = []
        for i in range(len(observations) - 1, -1, -1):
            b_curr = {}
            for st in range(self.emission_matrix.shape[0]):
                # base case for the algorithm
                if i == len(observations) - 1:
                    prev_b_sum = self.transition_matrix[st][end_st] 
                else:
                    probs = [b_prev[l] + self.emission_matrix[l][observations[i+1]-1] + self.transition_matrix[st][l] 
                             for l in range(self.emission_matrix.shape[0])]
                    prev_b_sum = np.log(sum(np.exp(probs)))

                b_curr[st] = prev_b_sum

            bwd.insert(0,b_curr)
            b_prev = b_curr
        probs = [self.pi[l] + self.emission_matrix[l][observations[0]-1] + b_curr[l] for l in range(self.emission_matrix.shape[0])]
        p_bwd = np.log(sum(np.exp(probs)))
        print(bwd)
        print(p_bwd)
        assert(p_fwd == p_bwd)
        

    

if __name__ == "__main__":
    df = pd.read_csv('Data/input.csv')
    model = HiddenMarkovModel(df['decile'],4)
    model.forward_algorithm(df['decile'][:10],3)