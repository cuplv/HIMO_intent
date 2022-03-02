from collections import deque 
import numpy as np
import graph_tool.all as gt
from cvxopt import matrix, solvers

class Buchi_AP:
    def __init__(self, h):
        self.h = h
        self.generate_new_vertices()
        self.generate_new_edges()
        self.generate_graph()
        
    def generate_new_vertices(self):
        # ----------------------------------------------------------------------------------------
        # New vertex consists of vertex ID of Buchi automaton and an incoming AP.
        # Therefore, this graph has more vertices than Buchi.
        # output:
        #     vertex_with_AP = [incoming AP, resulting state]
        #     vertex_with_AP_dict[(incoming AP, resulting state)] = new vertex ID
        # ----------------------------------------------------------------------------------------
        vertex_with_AP = deque()
        for t in self.h.transition_pairs:
            vertex_with_AP.append((t[0], t[1]))
            vertex_with_AP.append((t[2], t[3]))
        vertex_with_AP = np.unique(vertex_with_AP, axis=0)
        self.vertex_with_AP = vertex_with_AP
        
        vertex_with_AP_dict=dict()
        for idx, v in enumerate(vertex_with_AP):
            vertex_with_AP_dict[(v[0], v[1])] = idx
        self.vertex_with_AP_dict = vertex_with_AP_dict
        
        self.new_init_state = vertex_with_AP[(vertex_with_AP[:,1]==self.h.init_state) & (vertex_with_AP[:,0]==-1),:][0]
        self.new_init_state_idx = vertex_with_AP_dict[(self.new_init_state[0], self.new_init_state[1])]

        # self.new_acc_state = vertex_with_AP[vertex_with_AP[:,1]==self.h.acc_state,:]
        self.new_acc_state = [s for s in vertex_with_AP if s[1] in self.h.acc_state]
        self.new_acc_state_idx = deque()
        for nas in self.new_acc_state:
            self.new_acc_state_idx.append(vertex_with_AP_dict[(nas[0], nas[1])])


    def generate_new_edges(self):        
        # ----------------------------------------------------------------------------------------
        # New edges consist of two new vertex IDs 
        # output:
        #     new_transition_pairs_with_dist[(new starting_node, new end_node)] = distance
        # ----------------------------------------------------------------------------------------
        new_transition_pairs_with_dist = dict()
        for t in self.h.transition_pairs_with_dist:
            starting_node = self.vertex_with_AP_dict[(t[0], t[1])]
            end_node = self.vertex_with_AP_dict[(t[2], t[3])]
            new_transition_pairs_with_dist[(starting_node, end_node)] = t[4]
        self.new_transition_pairs_with_dist = new_transition_pairs_with_dist
        
    def generate_graph(self):
        # ------------------------------------
        # output:
        #     graph with new verties and edges
        # weight is distance between two APs
        # ------------------------------------
        g = gt.Graph()

        edges = deque()
        [edges.append((x[0], x[1])) for x in self.new_transition_pairs_with_dist.keys()]
        g.add_edge_list(edges)

        g.edge_properties['weight'] = g.new_edge_property("double")
        g.edge_properties['weight'].fa = self.h.transition_pairs_with_dist[:,4]
        self.g = g

    def compute_epoch_cost(self):
        # ----------------------------------------------------------------------------------------
        # Epoch cost is the min of cumulative weights between a state and one of acc states.
        # Used 'shortest_distance' to compute the minimum cost and used inversed transition graph to minimize computation.
        #    Instead of computing distance from each state to acc states(|all vertices|), 
        #    it's computing distance from acc states to all cells (|accepting states|)
        # output:
        #     epoch_cost[new vertex ID] = (closest new acc state idx, cost)
        # ----------------------------------------------------------------------------------------
        
        # inverted direction
        g_inverse= gt.Graph()

        edges = deque()
        [edges.append((x[1], x[0])) for x in self.new_transition_pairs_with_dist.keys()]
        g_inverse.add_edge_list(edges)

        g_inverse.edge_properties['weight'] = g_inverse.new_edge_property("double")
        g_inverse.edge_properties['weight'].fa = self.h.transition_pairs_with_dist[:,4]

        # Compute epoch cost from each state
        dist_inverse = dict()
        for s in self.new_acc_state_idx:
            dist_inverse[s] = deque(gt.shortest_distance(g_inverse, source=s, weights=g_inverse.ep.weight).a)
        dist_inverse = np.array([*dist_inverse.values()])

        epoch_cost=dict()
        for s in deque(self.g.vertex_index):
            epoch_cost[s] = (self.new_acc_state_idx[np.argmin(dist_inverse[:,s])], min(dist_inverse[:,s]))
        self.epoch_cost = epoch_cost
        
    def compute_repulsive_potential(self, gamma = 0.8):  
        # ----------------------------------------------------------------------------------------
        # 'epochs' list grows until new epoch is not availble.
        # Once epochs list is ready, run lp solver to find repulsive potential value.
        # cvxopt is the fastest lp solver.
        # Used 'shortest_distance' to compute the minimum cost and used inversed transition graph to minimize computation.
        #    Instead of computing distance from each state to acc states(|all vertices|), 
        #    it's computing distance from acc states to all cells (|accepting states|)
        # output:
        #     repulsive_potential[new vertex ID] = repulsive potential value
        # ----------------------------------------------------------------------------------------
        
        self.compute_epoch_cost() 
        
        # make solver quite
        solvers.options['show_progress'] = False

        repulsive_potential = dict()
        for state in deque(self.g.vertex_index):
            s = state
            epochs = [state]
            epochs_cost = []
            while True:
                if len(epochs) > 1 and (epochs[-1] == epochs[-2]):
                    break
                closest_acc_idx = self.epoch_cost[state][0]
                closest_acc_cost = self.epoch_cost[state][1]
                epochs.append(closest_acc_idx)
                epochs_cost.append(closest_acc_cost)
                state = epochs[-1]

            epoch_len = len(epochs)-1
            
            # if the first cost is 'inf', then none of acc states are reachable.
            # Then the repulsive potential is an infinite.
            if epochs_cost[0] == np.inf:
                repulsive_potential[s] = np.inf
                continue
            
            # LP solver------------------------------
            c = np.ones(epoch_len)*-1
            b = np.array(epochs_cost)*gamma
            a = np.zeros([epoch_len,epoch_len])

            for idx1 in range(epoch_len):
                if idx1 == epoch_len-1:
                    a[idx1][idx1] = 1 - gamma**2
                else:
                    a[idx1][idx1] = 1
                    a[idx1][idx1+1] = -gamma**2

            a, b, c = matrix(a), matrix(b), matrix(c)

            result = solvers.lp(c, a, b)
            repulsive_potential[s] = result['x'][0]
            # ----------------------------------------
            
        self.repulsive_potential = repulsive_potential

    def compute_probab_of_transitions(self, beta=1, alpha=10):  
        # ----------------------------------------------------------------------------------------
        # From each state, find out edges and compute the probability.
        # output:
        #     new_transition_pairs_with_prob[(new vertex ID, neighbor ID)] = probability of the transition
        # ----------------------------------------------------------------------------------------

        new_transition_pairs_with_prob = dict()
        for state in deque(self.g.vertex_index):
            neighbors_idx = self.g.get_out_neighbors(state)

            boltzmann = deque()
            self.beta = beta
            for n in neighbors_idx:
                cost = self.new_transition_pairs_with_dist[(state, n)]
                boltzmann.append(self.compute_Boltzmann(cost, self.repulsive_potential[n], alpha))

            sum_boltzmann = sum(boltzmann)
            if sum_boltzmann == 0:
                boltzmann = [1/len(boltzmann) for p in boltzmann]    
            else:
                boltzmann = [self.normalize(p, sum_boltzmann) for p in boltzmann]

            for idx, n in enumerate(neighbors_idx):
                new_transition_pairs_with_prob[(state, n)] = boltzmann[idx]
        self.new_transition_pairs_with_prob = new_transition_pairs_with_prob
        
    def compute_Boltzmann(self, cost, U, alpha):
        p = np.e**(-self.beta * (cost + alpha*U))
        return p

    def normalize(self, p, sum):
        normalized_p = p / sum
        return normalized_p 

    def plot_graph(self):
        gt.graph_draw(self.g, vertex_text=self.g.vertex_index)