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
        # print(fwd)
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
        # print(bwd)
        print(p_bwd)
        # assert(p_fwd == p_bwd)

        # Compute posterior, which is 
        posterior = []
        for i in range(len(observations)):
            posterior.append({st: fwd[i][st] * bwd[i][st] / p_fwd for st in range(self.emission_matrix.shape[0])})

        # assert p_fwd ==  p_bwd
        self.fwd = fwd
        self.bwd = bwd
        self.posterior = posterior
        return fwd, bwd, posterior
    
    def viterbi(self,observation_sequence: list):
        '''
        The viterbi algorithm finds the most likely sequence of states given a specific set of
        observations
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
    
    def baum_welch(self, observation_sequence, iterations=25):
        """
        Baum-Welch algorithm (Expectation-Maximization) to estimate HMM parameters.
        """
        N = self.emission_matrix.shape[0]  # Number of states
        T = len(observation_sequence)  # Length of observation sequence
        M = self.emission_matrix.shape[1]  # Number of observation symbols

        for _ in range(iterations):
            # E-step: calculate forward and backward probabilities
            fwd, bwd, posterior = self.forward_backward_algorithm(observation_sequence, N - 1)

            # Calculate xi
            xi = np.zeros((T - 1, N, N))

            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = fwd[t][i] + self.transition_matrix[i][j] + self.emission_matrix[j][observation_sequence[t + 1] - 1] + bwd[t + 1][j]
                
                # Normalize xi
                norm_factor = self.logsumexp(xi[t].flatten())
                xi[t] -= norm_factor

            # now calculate gamma
            gamma = np.zeros((T, N))

            for t in range(T):
                log_probs = [fwd[t][i] + bwd[t][i] for i in range(N)]  # Unnormalized gamma
                norm_factor = self.logsumexp(log_probs)             # Normalization factor
                for i in range(N):
                    gamma[t, i] = fwd[t][i] + bwd[t][i] - norm_factor 

            # M-step: re-estimate parameters
            # Update pi
            self.pi = gamma[0]

            # Update transition matrix
            for i in range(N):
                for j in range(N):
                    print(self.logsumexp(xi[:, i, j]))
                    print(self.logsumexp(gamma[:-1, i]))
                    self.transition_matrix[i, j] = self.logsumexp(xi[:, i, j]) - self.logsumexp(gamma[:-1, i])

            # Update emission matrix
            for j in range(N):
                for vk in range(M):
                    log_probs = []
                    for t in range(T):
                        if observation_sequence[t] == vk + 1:
                            log_probs.append(gamma[t, j])
                    self.emission_matrix[j, vk] = self.logsumexp(log_probs) - self.logsumexp(gamma[:, j])

    def logsumexp(self, log_probs):
        """
        Log-sum-exp trick to avoid underflow.
        """
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

if __name__ == "__main__":
    df = pd.read_csv('Data/input.csv')
    model = HiddenMarkovModel(df['decile'],4)
    model.forward_backward_algorithm(df['decile'][:10],3)
    print(model.posterior)
    # model.viterbi(df['decile'][:100])
    pass
    model.baum_welch(df['decile'][:100])
    pass