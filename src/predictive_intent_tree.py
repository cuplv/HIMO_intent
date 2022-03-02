from collections import deque 
import numpy as np 

class PredictiveTree:
    def __init__(self, g_AP, NUM_REGIONS):
        self.g_AP = g_AP
        self.NUM_REGIONS = NUM_REGIONS
        
    def compute_probab_of_transitions(self, beta = 1):
        # ----------------------------------------------------------------------------------------
        # output:
        #     AP_prob_dict[(new vertex ID, AP)] = probability
        # ----------------------------------------------------------------------------------------
        
        AP_prob_dict = dict()
        for state in deque(self.g_AP.g.vertex_index):
            neighbors_idx = self.g_AP.g.get_out_neighbors(state)
            neighbors = self.g_AP.vertex_with_AP[neighbors_idx]
            num_out_edges = neighbors[:,0].shape[0]

            AP_prob = deque()
            for AP in range(self.NUM_REGIONS):
                one_AP_prob = [self.g_AP.new_transition_pairs_with_prob[(state, n)] for idx, n in enumerate(neighbors_idx) if neighbors[idx][0] == AP]
                AP_prob.append(sum(one_AP_prob))

            sum_AP_prob = sum(AP_prob)
            AP_prob = [self.g_AP.normalize(p, sum_AP_prob) for p in AP_prob]

            for idx, AP in enumerate(range(self.NUM_REGIONS)):    
                AP_prob_dict[(state, AP)] = AP_prob[idx]  
        self.AP_prob_dict = AP_prob_dict
        
    def find_next_state_ID_with_AP(self, AP, current_state, vertex_with_AP_dict_to_list):
        # ----------------------------------------------------------------------------------------
        # Receives an atomic proposition and currente state(vertex) ID
        # returns the next state IDs
        # This function is mainly used by the 'build_tree' function.
        # ----------------------------------------------------------------------------------------
        neighbors_idx = (self.g_AP.g.get_out_edges(current_state))[:,1]
        neighbors_with_AP = [vertex_with_AP_dict_to_list[idx] for idx in neighbors_idx]        
        neighbors_with_AP_filtered = [x for x in neighbors_with_AP if x[0] == AP]
        neighbors_with_AP_filtered_idx = [self.g_AP.vertex_with_AP_dict[(x[0], x[1])] for x in neighbors_with_AP_filtered]
        return neighbors_with_AP_filtered_idx
    
    def build_tree(self, depth=4, n_intents = 1):
        # ----------------------------------------------------------------------------------------
        # tree node ID is a tuple starting from 0 that means the root.
        #     As tree grows, tree node ID also grows (0, 1st AP, 2nd AP, 3rd AP, ...)
        # current_state dict includes tree node IDs and corresponding state IDs.
        #     current_state[(tree node ID)] = [resulting state IDs]
        # probability dict includes tree node IDs and corresponding probability (Markov property).
        #     probability[(tree node ID)] = [probability given a parent and the last AP]
        #
        # pred_tree dict includes all information about a tree.
        #     pred_tree[tree-depth] = [current_state, probability]
        # ----------------------------------------------------------------------------------------
                
        pred_tree = dict()
        current_state = {(0,):[0]}
        # probability = {(0,):1.0}
        probability = {(0,):1/n_intents}        
        pred_tree[0] = [current_state, probability]
        
        neighbors_with_AP_filtered_idx_dict = dict()
        vertex_with_AP_dict_to_list = deque(self.g_AP.vertex_with_AP_dict)
        for AP in range(self.NUM_REGIONS):                    
            for cs in self.g_AP.g.vertex_index:
                neighbors_with_AP_filtered_idx_dict[(AP, cs)] = self.find_next_state_ID_with_AP(AP, cs, vertex_with_AP_dict_to_list)
            
        # when p0, p1, p2 are available,
        # tree node ID in depth 0: 0 (root)
        # tree node ID in depth 1: 00 01 02
        # tree node ID in depth 2: 000 001 002 010 011 012 ...
        for d in range(depth):
            current_state = pred_tree[d][0]
            probability = pred_tree[d][1]    
            next_cs_key = deque()
            next_cs = deque()
            next_p = deque()
            
            for idx, cs_key in enumerate(current_state):

                for AP in range(self.NUM_REGIONS):                    
                    next_cs_key.append((cs_key + (AP,)))

                    cs_list = current_state[cs_key]
                    next_cs_tmp = []
                    next_p_tmp = []
                    for cs in cs_list:
                        neighbors_with_AP_filtered_idx = neighbors_with_AP_filtered_idx_dict[(AP, cs)]
                        # neighbors_with_AP_filtered_idx = self.find_next_state_ID_with_AP(AP, cs)
                        if len(neighbors_with_AP_filtered_idx) > 0:
                            next_cs_tmp.append(neighbors_with_AP_filtered_idx) 
                            # next_p_tmp.append(self.AP_prob_dict[cs, AP])
                            next_p_tmp.append(probability[cs_key] * self.AP_prob_dict[cs, AP])                            

                    # next_cs_tmp = sum(next_cs_tmp, [])
                    next_cs_tmp = list(np.unique(sum(next_cs_tmp, [])))
                    if len(cs_list) == 0:
                        next_p_tmp = 0
                    else:
                        next_p_tmp = sum(next_p_tmp) / len(cs_list)
                        
                    if next_p_tmp == 0:
                        next_cs_tmp = []
                        
                    next_cs.append(next_cs_tmp)
                    next_p.append(next_p_tmp)
                
                next_current_state = dict([(key, cs) for key, cs in zip(next_cs_key,next_cs) ])
                next_probability = dict([(key, p) for key, p in zip(next_cs_key,next_p) ])

            pred_tree[d+1] = [next_current_state, next_probability]
            # print(pred_tree[d+1][0])
            # print(pred_tree[d+1][1])
        self.tree = pred_tree
                    
    def build_tree_on_the_fly(self, nodeID, current_state, root_prob, depth=4, n_intents = 1):
        # ----------------------------------------------------------------------------------------
        # tree node ID is a tuple starting from 0 that means the root.
        #     As tree grows, tree node ID also grows (0, 1st AP, 2nd AP, 3rd AP, ...)
        # current_state dict includes tree node IDs and corresponding state IDs.
        #     current_state[(tree node ID)] = [resulting state IDs]
        # probability dict includes tree node IDs and corresponding probability (Markov property).
        #     probability[(tree node ID)] = [probability given a parent and the last AP]
        #
        # pred_tree dict includes all information about a tree.
        #     pred_tree[tree-depth] = [current_state, probability]
        # ----------------------------------------------------------------------------------------
                
        pred_tree = dict()
        current_state = {nodeID: current_state}
        probability = {nodeID: root_prob}
        pred_tree[0] = [current_state, probability]
        
        neighbors_with_AP_filtered_idx_dict = dict()
        vertex_with_AP_dict_to_list = deque(self.g_AP.vertex_with_AP_dict)
        for AP in range(self.NUM_REGIONS):                    
            for cs in self.g_AP.g.vertex_index:
                neighbors_with_AP_filtered_idx_dict[(AP, cs)] = self.find_next_state_ID_with_AP(AP, cs, vertex_with_AP_dict_to_list)
            
        # when p0, p1, p2 are available,
        # tree node ID in depth 0: 0 (root)
        # tree node ID in depth 1: 00 01 02
        # tree node ID in depth 2: 000 001 002 010 011 012 ...
        for d in range(depth):
            current_state = pred_tree[d][0]
            probability = pred_tree[d][1]    
            next_cs_key = deque()
            next_cs = deque()
            next_p = deque()
            
            for idx, cs_key in enumerate(current_state):

                for AP in range(self.NUM_REGIONS):                    
                    next_cs_key.append((cs_key + (AP,)))

                    cs_list = current_state[cs_key]
                    next_cs_tmp = []
                    next_p_tmp = []
                    for cs in cs_list:
                        neighbors_with_AP_filtered_idx = neighbors_with_AP_filtered_idx_dict[(AP, cs)]
                        # neighbors_with_AP_filtered_idx = self.find_next_state_ID_with_AP(AP, cs)
                        if len(neighbors_with_AP_filtered_idx) > 0:
                            next_cs_tmp.append(neighbors_with_AP_filtered_idx) 
                            # next_p_tmp.append(self.AP_prob_dict[cs, AP])
                            next_p_tmp.append(probability[cs_key] * self.AP_prob_dict[cs, AP])                            
                    # next_cs_tmp = sum(next_cs_tmp, [])
                    next_cs_tmp = list(np.unique(sum(next_cs_tmp, [])))
                    if len(cs_list) == 0:
                        next_p_tmp = 0
                    else:
                        next_p_tmp = sum(next_p_tmp) / len(cs_list)
                        
                    if next_p_tmp == 0:
                        next_cs_tmp = []
                                                
                    next_cs.append(next_cs_tmp)
                    next_p.append(next_p_tmp)
             
                next_current_state = dict([(key, cs) for key, cs in zip(next_cs_key,next_cs) ])
                next_probability = dict([(key, p) for key, p in zip(next_cs_key,next_p) ])

            pred_tree[d+1] = [next_current_state, next_probability]
            # print(pred_tree[d+1][0])
            # print(pred_tree[d+1][1])
        self.tree = pred_tree
                                