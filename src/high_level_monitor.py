from collections import deque 
from src.predictive_intent_tree import *
from src.buchi import *
from src.buchi_with_AP import *

import numpy as np
import copy

class Intent:
    def __init__(self, LTL, patrol_mission_checker, kitchen_mission_checker, ws, alpha=10, beta=1, tree_depth=4, n_intents = 1):
        NUM_REGIONS = ws.NUM_REGIONS
        self.tree_depth = tree_depth
        self.n_intents = n_intents
        self.LTL = LTL
        # print("---------------------------------------------------")
        # print("LTL formula: ", self.LTL)
        self.buchi = generate_Buchi(self.LTL, NUM_REGIONS)
        # ------------------------------------------------------------------
        # Useful vars
        #     buchi['graph']: buchi automaton in graph-tool format
        #     buchi['vertex']: vertex info
        #     buchi['edges']: edge info
        #     buchi['v_accept']: accepting states        
        #     buchi['v_init']: initial state    
        #     buchi['e_gate']: transition condition                 
        # ------------------------------------------------------------------        
        # print("Original Buchi info: \n", self.buchi['graph'], "\n")
        
        # If LTL is a patrolling mission, change edges
        if patrol_mission_checker == 1:
            acc_state = deque([idx for idx, v in enumerate(list(self.buchi['v_accept'])) if v==[0]])[0]
            first_neighbors = self.buchi['graph'].get_out_neighbors(0)
            
            self.buchi['graph'].remove_edge(self.buchi['graph'].edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][acc_state]))
            for neighbor in first_neighbors:
                e_gate_idx = list(self.buchi['edges'].values()).index(self.buchi['edges'][(self.buchi['vertex'][0], neighbor)])
                neighbor_e_gate = list(self.buchi['e_gate'])[e_gate_idx]

                self.buchi['graph'].add_edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][neighbor])
                self.buchi['e_gate'][self.buchi['graph'].edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][neighbor])] = neighbor_e_gate

        if kitchen_mission_checker == 1:
            acc_state = deque([idx for idx, v in enumerate(list(self.buchi['v_accept'])) if v==[0]])[0]
            first_neighbors = self.buchi['graph'].get_out_neighbors(0)
            self.buchi['graph'].remove_edge(self.buchi['graph'].edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][acc_state]))
            for neighbor in first_neighbors:
                if neighbor != 0:
                    e_gate_idx = list(self.buchi['edges'].values()).index(self.buchi['edges'][(self.buchi['vertex'][0], neighbor)])
                    neighbor_e_gate = list(self.buchi['e_gate'])[e_gate_idx]

                    self.buchi['graph'].add_edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][neighbor])
                    self.buchi['e_gate'][self.buchi['graph'].edge(self.buchi['vertex'][acc_state], self.buchi['vertex'][neighbor])] = neighbor_e_gate
        # ==================================================================
        self.bh = Buchi_handler(self.buchi, NUM_REGIONS, self.LTL)
        self.bh.get_transition_pairs_and_dist(ws)
        # ------------------------------------------------------------------
        # Useful vars
        #     bh.transition_pairs_with_dist : [previous AP, source node, next AP, target node, dist]
        # ------------------------------------------------------------------        
        # ==================================================================
        
        # ==================================================================
        self.g_AP = Buchi_AP(self.bh)
        # print("New Buchi(state w/ incoming AP) info: \n", self.g_AP.g, "\n")
        # ------------------------------------------------------------------
        # Useful vars
        #     g_AP.vertex_with_AP = [incoming AP, resulting state]
        #     g_AP.vertex_with_AP_dict[(incoming AP, resulting state)] = new vertex ID
        #     g_AP.new_transition_pairs_with_dist[(new starting_node, new end_node)] = distance
        #     g_AP.g: new graph
        # ------------------------------------------------------------------
        
        self.g_AP.compute_repulsive_potential(gamma = 0.8)
        # print("Repulsive potential value computation...done")
        # ------------------------------------------------------------------
        # Useful vars
        #     g_AP.repulsive_potential[new vertex ID] = repulsive potential value
        # -----------------------------------------------------------------
        
        self.g_AP.compute_probab_of_transitions(beta = beta, alpha = alpha)
        # print(f"Probability of transition computation...done (beta: {b}, alpha: {a})")        
        # ------------------------------------------------------------------
        # Useful vars
        #     g_AP.new_transition_pairs_with_prob[(new vertex ID, neighbor ID)] = probability of the transition
        # -----------------------------------------------------------------
        # ==================================================================        
        
        # ==================================================================                
        self.pt = PredictiveTree(self.g_AP, NUM_REGIONS)
        self.pt.compute_probab_of_transitions(beta = beta)
        self.pt.build_tree(depth = tree_depth, n_intents = self.n_intents)
        # print("Predictive intent tree generation...done")                
        # ------------------------------------------------------------------
        # Useful vars
        #     pt.tree[depth][0][tree node ID] = [states]
        #     pt.tree[depth][1][tree node ID] = [probability]
        # -----------------------------------------------------------------
        # ==================================================================                
        
        
class HighLvMonitor:
    def __init__(self, intent_set, ws, lb_prob=0.01, tree_depth=4):
        self.intent_set = intent_set
        self.lb_prob = lb_prob
        self.tree_depth = tree_depth
        self.ws = ws
        self.NUM_REGIONS = ws.NUM_REGIONS
        
        self.initialize()
            
    def initialize(self):
        self.init_probability_tree()
        self.probab_intent = [1/len(self.intent_set)] * len(self.intent_set)
        self.observation = [self.ws.start_position]
        self.belief_events = dict((intent, [0]) for intent in self.intent_set)
        self.observed_events = [0]
        # for intent in self.intent_set: 
        #     intent.pt.build_tree(depth = self.tree_depth)

    def update_short_term_goal(self, updated_prior):
        # print('before update: ', self.probability_tree, sum(self.probability_tree[2].values()))
        # print('-------------------------------')        
        for idx, (key, v) in enumerate(self.saved_tree[1].items()):
            self.saved_tree[1][key] = updated_prior[idx] 
        self.probability_tree = self.product_probability(copy.deepcopy(self.saved_tree))
        # print('update the tree: ', self.probability_tree, sum(self.probability_tree[2].values()))
        # print('-------------------------------')         
        # print('saved tree: ', self.saved_tree, sum(self.saved_tree[2].values()))
        # print('-------------------------------')                 
    
    def product_probability(self, tree):    
        for d in range(1, self.tree_depth):
            children = tree[d]
            parents = tree[d-1]    
            for key, val in children.items():
                # print(key, children[key], children[key] * parents[key[:-1]])
                children[key] = children[key] * parents[key[:-1]]
            tree[d] = children
        return tree

    def init_probability_tree(self):
        n = len(self.intent_set)
        probability_tree = dict()
        
        for d in range(self.tree_depth):
            combined_probab = deque()

            for intent in self.intent_set:
                combined_probab.append(list(intent.pt.tree[d][1].values()))

            combined_probab = np.array(combined_probab)
            if combined_probab.shape[1] > self.NUM_REGIONS:
                combined_probab_list = np.sum(combined_probab, axis=0)  
                combined_probab_list = np.hsplit(combined_probab_list, self.NUM_REGIONS**(d-1))
                combined_probab_list_sum = np.sum(combined_probab_list, axis=1)
            else:
                combined_probab_list = combined_probab
                combined_probab_list = [np.sum(combined_probab_list, axis=0)]
                combined_probab_list_sum = [np.sum(combined_probab_list, axis=1)]

            combined_probab = np.array([p/sum_p if sum_p > 0 else [0]*len(p) for p, sum_p in zip(combined_probab_list, combined_probab_list_sum)])
            combined_probab = list(combined_probab.flatten())            
            # combined_probab= np.sum(combined_probab, axis=0) / n         

            probability_tree[d] = dict()    
            for idx, key in enumerate(intent.pt.tree[d][1]):
                probability_tree[d][key] = combined_probab[idx]

        self.saved_tree = copy.deepcopy(probability_tree)

        probability_tree_product = self.product_probability(probability_tree)
        self.probability_tree = probability_tree_product

    
    def update_probability_of_intents(self):        
        num_dead_intent = self.probab_intent.count(0)
        num_alive_intent = len(self.intent_set) - num_dead_intent
        sum_probab_intent = sum(self.probab_intent)
        
        for idx, _ in enumerate(self.probab_intent):
            self.probab_intent[idx] = self.probab_intent[idx] / sum_probab_intent            
            
            # if self.probab_intent[idx] == 0:
            #     self.probab_intent[idx] = self.lb_prob / num_dead_intent
            # elif num_dead_intent == 0:
            #     # self.probab_intent[idx] = 1 / num_alive_intent
            #     self.probab_intent[idx] = self.probab_intent[idx] / sum_probab_intent
            # else:
            #     # self.probab_intent[idx] = (1 - self.lb_prob) / num_alive_intent
            #     self.probab_intent[idx] = (1 - self.lb_prob) * (self.probab_intent[idx] / sum(self.probab_intent))

    def update_probability_tree(self):
   
        for idx, intent in enumerate(self.intent_set): 
            if len(self.belief_events[intent]) > 1:
                root = tuple(self.belief_events[intent])
                depth = 1
                root_state = intent.pt.tree[depth][0][root]
                # root_prob = intent.pt.tree[depth][1][root]
                root_prob = self.probab_intent[idx]
                intent.pt.build_tree_on_the_fly(nodeID = root, current_state = root_state, root_prob = root_prob)
            else:
                root_prob = self.probab_intent[idx]                
                intent.pt.build_tree_on_the_fly(nodeID = (0,), current_state = [0], root_prob = root_prob)
                # intent.pt.build_tree_on_the_fly(nodeID = (0,), current_state = [0], root_prob = 1/len(self.intent_set))  

        n = len(self.intent_set)
        updated_probability_tree = dict()        
        for d in range(self.tree_depth):
            combined_probab = deque()
            updated_probability_tree[d] = dict.fromkeys(self.probability_tree[d], 0)

            for idx, intent in enumerate(self.intent_set): 
                # print(intent.pt.tree[d][1].values(), updated_probability_tree[d])

                combined_probab.append(list(intent.pt.tree[d][1].values()))
                # combined_probab.append([x * self.probab_intent[idx] for x in intent.pt.tree[d][1].values()])
            # print(combined_probab)
            combined_probab = np.array(combined_probab)
            if combined_probab.shape[1] > self.NUM_REGIONS:
                combined_probab_list = np.sum(combined_probab, axis=0)  
                combined_probab_list = np.hsplit(combined_probab_list, self.NUM_REGIONS**(d-1))
                combined_probab_list_sum = np.sum(combined_probab_list, axis=1)
            else:
                combined_probab_list = combined_probab
                combined_probab_list = [np.sum(combined_probab_list, axis=0)]
                combined_probab_list_sum = [np.sum(combined_probab_list, axis=1)]

            # combined_probab = np.array([p/sum_p for p, sum_p in zip(combined_probab_list, combined_probab_list_sum)])
            combined_probab = np.array([p/sum_p if sum_p > 0 else [0]*len(p) for p, sum_p in zip(combined_probab_list, combined_probab_list_sum)])
            
            combined_probab = list(combined_probab.flatten())            
            # combined_probab= np.sum(combined_probab, axis=0) / n            

            for idx, key in enumerate(self.probability_tree[d]):
                updated_probability_tree[d][key] = combined_probab[idx]
        self.saved_tree = copy.deepcopy(updated_probability_tree)

        updated_probability_tree_product = self.product_probability(updated_probability_tree)
        self.probability_tree = updated_probability_tree_product  

        
    def update(self):
        self.update_probability_of_intents()
        self.update_probability_tree()        
        
    def add_observation(self, obs_list):
        for obs in obs_list:
            self.observation.append(obs)
            
            # If the observed state is one of regions
            if obs in self.ws.region:
                AP = self.ws.region.index(obs)
                self.observed_events.append(AP)
                
                # check probability 
                for idx, intent in enumerate(self.intent_set):
                    self.belief_events[intent].append(AP) # update high level event list
                    belief_highlv_events = self.belief_events[intent]

                    nodeID = tuple(belief_highlv_events)
                    depth = 1

                    p = intent.pt.tree[depth][1][nodeID]
                    self.probab_intent[idx] = p                
                    if p ==0:
                        self.belief_events[intent] = [0]
                        self.probab_intent[idx] = 0.01

                self.update()        
                
    def add_observation_thor(self, obs_list, region_accept_dist):
        
        for obs in obs_list:
            self.observation.append(obs)
            
            # < dist
            # no repeat
            for i, r in enumerate(self.ws.region):
                if np.sqrt((obs[0]-r[0])**2+(obs[1]-r[1])**2) <= region_accept_dist[i]:
                    obs = r
                    break
            # print("Obs found: ", obs_list[0], obs)
            if obs in self.ws.region:
                AP = self.ws.region.index(obs)
                
                if (AP == self.observed_events[-1]) and len(self.observed_events) > 1:
                    continue

                self.observed_events.append(AP)
                # print("AP: ", obs_list[0], obs, AP)
                # check probability 
                for idx, intent in enumerate(self.intent_set):
                    self.belief_events[intent].append(AP) # update high level event list
                    belief_highlv_events = self.belief_events[intent]

                    nodeID = tuple(belief_highlv_events)
                    depth = 1

                    p = intent.pt.tree[depth][1][nodeID]
                    self.probab_intent[idx] = p
                    if p ==0:
                        self.belief_events[intent] = [0]
                        self.probab_intent[idx] = 0.01

                self.update()                        