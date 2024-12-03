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
    def forward_backward_algorithm(self,observations,end_st): 
        """
        Forward-backward Algorithm, ported from:
        https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Pseudocode

        This code find the most likely hidden state at all times t. It puts these into dictionaries,
        and returns them. This algorithm is essential for the viterbi and baum-welch algorithms 
        later on.
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

        # now compute the backward proceedure
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

        # Compute posterior, which is 
        posterior = []
        for i in range(len(observations)):
            posterior.append({st: fwd[i][st] * bwd[i][st] / p_fwd for st in range(self.emission_matrix.shape[0])})

        assert p_fwd ==  p_bwd
        self.fwd = fwd
        self.bwd = bwd
        self.posterior = posterior
        return fwd, bwd, posterior
    
    def viterbi(self,observation_sequence: list):
        '''
        The viterbi algorithm finds the most likely sequence of states to use, which will be used
        for our baum-welch algorithm. 
        '''
        num_states = len(self.pi)
        num_obs = len(observation_sequence)

        # this is to hold the probabilities for the maximization step
        trellis = np.zeros((num_states,num_obs))

        # same but for all of the traversed paths
        backpointers = np.zeros((num_states,num_obs))

        for s in range(num_states):
            trellis[s][0] = self.pi[s] + self.emission_matrix[s][observation_sequence[0]]
        print(trellis)

        # compute argmax
        for t in range(1,num_obs):
            for s in range(num_states):
                max_prob = -(2 ** 512)
                max_state = -1

                # this is why we did the init probabilities seperately
                for prev_s in range(num_states):
                    prob = trellis[prev_s][t-1] + self.transition_matrix[prev_s][s]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_s

                # write the max trellis and states into storage
                trellis[s][t] = max_prob + self.emission_matrix[s][observation_sequence[t]-1]
                backpointers[s][t] = max_state
        
        # find the probability of the best path by going through all of the final
        # states probabilities
        best_path_prob = -(2 ** 512)
        best_path_end = -1
        for s in range(num_states):
            if trellis[s][num_obs-1] > best_path_prob:
                best_path_prob = trellis[s][num_obs-1]
                best_path_end = s

        # then find the path that gave those probabilities
        best_path = np.zeros(num_obs)
        best_path[num_obs-1] = best_path_end
        # Iterate backwards from the second-to-last element to the first
        for t in range(num_obs - 2, -1, -1):
            # print(best_path[t + 1],t+1)
            print(backpointers[int(best_path[t + 1]),t + 1])
            best_path[t] = backpointers[int(best_path[t + 1])][t + 1]
        
        print(best_path)
        return best_path, best_path_prob

    

if __name__ == "__main__":
    df = pd.read_csv('Data/input.csv')
    model = HiddenMarkovModel(df['decile'],4)
    model.forward_backward_algorithm(df['decile'][:10],3)
    print(model.posterior)
    model.viterbi(df['decile'][:100])