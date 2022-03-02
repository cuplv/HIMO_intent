import spot
from src.readlbt import *
from collections import deque 
import numpy as np

def generate_LTL(NUM_REGIONS):
    # ------------------------------------------------------------------
    # Randomly generates LTL using specification patterns.
    # Note: currently, it generates 'visit all in any order' only.
    # TODO: use specification patterns
    # ------------------------------------------------------------------
    ltl_list = ''
    for i in range(NUM_REGIONS):
        if i == NUM_REGIONS-1:
            ltl_list = (ltl_list + 'F p' + str(i))
        else:
            ltl_list = (ltl_list + 'F p' + str(i) + ' & ')

    combinedLTL1 = ''
    for i in range(NUM_REGIONS):
        for j in range(NUM_REGIONS):
            if j > i:
                combinedLTL1 = (combinedLTL1 + 'G ! (p'+ str(i) + ' & p' + str(j) + ') & ')

    combinedLTL = (combinedLTL1 + ltl_list)
    return combinedLTL

def convert_LTL_to_LBTT(LTL_formula):
    # ------------------------------------------------------------------
    # Convert a LTL formula into an automaton written in the LBTT format.
    # 'readlbt.py' can generate a graph(graph-tool format) from LBTT only.
    # ------------------------------------------------------------------    
    LTL_formula = spot.formula(LTL_formula)
    LTL_formula = spot.translate(LTL_formula, 'Buchi', 'state-based', 'unambig') #'unambig', 'complete'
    LTL_formula_LBTT = LTL_formula.to_str('lbtt')
    return LTL_formula_LBTT

def generate_Buchi(LTL_formula, NUM_REGIONS):
    # ------------------------------------------------------------------
    # output:
    #     Buchi automaton expressed in graph-tool
    # ------------------------------------------------------------------    
    LTL_formula_LBTT = convert_LTL_to_LBTT(LTL_formula)
    (g, vertex, edges, v_accept, v_init, e_gate) = convert_LBTT_to_graph(LTL_formula_LBTT)    
    
    G = dict()
    G['graph'] = g
    G['vertex'] = vertex
    G['edges'] = edges
    G['v_accept'] = v_accept
    G['v_init'] = v_init
    G['e_gate'] = e_gate
    return G

def is_gate_open(gate, NUM_REGIONS, status):
    # ------------------------------------------------------------------
    # input:
    #     gate: transition condition
    #     NUM_REGIONS: number of atomic propositions, |AP|
    #     status: one-hot vector of atomic propositions.
    #         ex) p0, !p1, !p2 -> [1, 0, 0]
    # output:
    #     1: pass the transition, 0: nope
    # ------------------------------------------------------------------    
    for num in range(NUM_REGIONS):
        p = 'p' + str(num)
        gate = gate.replace(p, str(status[num]))        
    gate = gate.replace(' ', '')

    stack = deque()
    for exp in reversed(gate):
        stack.append(exp)
        if exp == '&':
            trash = stack.pop()
            num1, num2 = int(stack.pop()), int(stack.pop())
            stack.append(int(num1 and num2))
        elif exp == '|':
            trash = stack.pop()
            num1, num2 = int(stack.pop()), int(stack.pop())
            stack.append(int(num1 or num2))
        elif exp == '!':
            trash = stack.pop()
            num1 = int(stack.pop())
            stack.append(int(not num1))
        elif exp == 't':
            trash = stack.pop()
            stack.append(int(1))
    stack = [int(i) for i in stack]            

    return stack    

class Buchi_handler:
    def __init__(self, g, NUM_REGIONS, LTL):    
        self.g = g
        self.LTL = LTL
        self.init_state = deque(g['v_init']).index(1)
        self.acc_state = deque([idx for idx, v in enumerate(list(g['v_accept'])) if v==[0]])
        self.NUM_REGIONS = NUM_REGIONS
        
        self.find_transition_pairs()
    
    def find_passing_AP(self):
        # ----------------------------------------------------------------------------------------
        # For each transition, find APs that can pass the transition.
        # output:
        #     transitions: [source node, target node, possible AP]
        # transition between the source node and the target node is allowed with possible AP
        # ----------------------------------------------------------------------------------------

        e_gate_list = deque(self.g['e_gate'])
        e_gate_list_uniq = deque(np.unique(e_gate_list))

        opening_AP = dict()
        for gate in e_gate_list_uniq:
            tmp = deque()
            for n in range(self.NUM_REGIONS):
                region_one_hot = deque([0])*self.NUM_REGIONS
                region_one_hot[n] = 1 # check one AP at a time

                gate_open = is_gate_open(gate, self.NUM_REGIONS, region_one_hot)[0]

                if gate_open:
                    tmp.append(n)
            opening_AP[gate] = tmp

        transitions = deque()
        # for edge_idx, edge in enumerate(self.g['edges'].values()):
        for edge_idx, edge in enumerate(list(self.g['graph'].edges())):
            # edge_idx = list(g['graph'].edges()).index(edge)
            gate = e_gate_list[edge_idx]

            current_state = int(edge.source())
            next_state = int(edge.target())

            # One transition may allow multiple APs to pass.
            for n in opening_AP[gate]:
                next_p = n
                transitions.append([current_state, next_state, next_p])
            
        return np.array(transitions)

    def find_transition_pairs(self):
        # -----------------------------------------------------------------------
        # Find all pairs of (previous AP, source node, next AP, target node)
        # source node is the resulting state of previous AP
        # target node is the resulting state of next PA from the source node
        # output:
        #     transition_pairs: [previous AP, source node, next AP, target node]
        # -----------------------------------------------------------------------
        transitions = self.find_passing_AP()
        
        t_from = dict()
        for vertex in self.g['vertex'].keys():
            t_from[vertex] = transitions[transitions[:,0]==vertex,:]

        transition_pairs = deque()
        for t in transitions:
            current_state = t[1]
            prev_p = t[2]

            next_transition = t_from[current_state]
            for n in next_transition:
                next_state = n[1]
                next_p = n[2]
                transition_pairs.append([prev_p, current_state, next_p, next_state])

        # for the initial state, incoming AP is set to -1
        prev_p = -1
        next_transition = t_from[self.init_state]

        for n in next_transition:
            next_state = n[1]
            next_p = n[2]
            transition_pairs.append([prev_p, self.init_state, next_p, next_state])

        transition_pairs = np.unique(transition_pairs, axis=0)
        self.transition_pairs = transition_pairs
    
    def get_transition_pairs_and_dist(self, ws):    
        # ----------------------------------------------------------------------------------------
        # Add distance info, between previous AP and next AP, to transition pairs.
        # Distance info is provided by workspace.
        # output:
        #     transition_pairs_with_dist: [previous AP, source node, next AP, target node, dist]
        # ----------------------------------------------------------------------------------------

        tmp = deque()
        for pair in self.transition_pairs:
            if pair[0] != -1:
                prev_AP = ws.region_idx[pair[0]]
            else:
                prev_AP = -1
            next_AP = ws.region_idx[pair[2]]

            if (prev_AP, next_AP) in ws.dist_between_regions_idx.keys():
                if prev_AP == next_AP and prev_AP in self.acc_state:
                    dist = 0
                else:
                    dist = ws.dist_between_regions_idx[(prev_AP, next_AP)]
                tmp.append(dist)
            else:
                tmp.append(np.inf)
                
        transition_pairs_with_dist = np.hstack([self.transition_pairs, np.array([tmp]).T])
        self.transition_pairs_with_dist = transition_pairs_with_dist
        
        # For default automaton
        if self.LTL.endswith('& 1'):
            # # No self-loop is allowed in the default automaton
            # non_inf_val = self.transition_pairs_with_dist[:,4][self.transition_pairs_with_dist[:,4]!=np.inf]
            # num_non_inf_val = len(non_inf_val)       
            # self.transition_pairs_with_dist[:,4][self.transition_pairs_with_dist[:,4]!=np.inf] = 1 / num_non_inf_val
            
            # Self-loop is allowed in the default automaton            
            non_inf_val = self.transition_pairs_with_dist[:,4]            
            num_non_inf_val = len(non_inf_val)        
            self.transition_pairs_with_dist[:,4] = 1 / num_non_inf_val


    def plot_buchi(self):
        gt.graph_draw(self.g['graph'], vertex_text=self.g['graph'].vertex_index)