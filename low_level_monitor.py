import numpy as np
from collections import deque 
import matplotlib.pyplot as plt
import random

class LowLvMonitor():
    def __init__(self, hm, ws, start_pos, beta=1, epsilon=0.1):
        self.hm = hm   
        self.ws = ws
        self.MAP_SIZE = ws.MAP_SIZE
        self.observation = np.array([(start_pos[0],start_pos[1])])
        self.beta = beta
        self.epsilon = epsilon
        
        self.initialize_prior()
        self.initialize_grid()

    def initialize_prior(self):
        # self.prior = np.array([1/len(self.ws.region)]*len(self.ws.region))
        self.prior = deque(self.hm.probability_tree[1].values())
        
    def add_observation(self, new_state):
        prob_next = self.inference()
        self.observation = np.vstack([self.observation, new_state])
        self.update_posterior_intent(prob_next)
        
        if new_state in self.ws.region:
            self.initialize_prior()
            
    def add_observation_thor(self, new_state, region_accept_dist):
        prob_next = self.inference()
        self.observation = np.vstack([self.observation, new_state])
        self.update_posterior_intent(prob_next)
        
        for i, r in enumerate(self.ws.region):
            if np.sqrt((new_state[0]-r[0])**2+(new_state[1]-r[1])**2) <= region_accept_dist[i]:
                self.initialize_prior()            
                break
    
    def inference(self):
        current_node = self.observation[-1]
        current_node_idx = self.ws.get_node(current_node)
        
        # Find neighbors
        neighbors = np.array([(node, weight) for _, node, weight in self.ws.g.get_out_edges(current_node_idx, [self.ws.g.ep.weight])])

        prob_next = dict()
        for r in self.ws.region_idx:
            prob_next[r] = self.inference_one_intent(r, current_node_idx, neighbors)
        return prob_next
    
    def inference_one_intent(self, region_idx, current_node_idx, neighbors):
        dist_shortest_to_region = self.ws.dist_to_cells_from[region_idx][current_node_idx]

        if current_node_idx in self.ws.region_idx and current_node_idx!= region_idx:
            dist_shortest_to_region = dist_shortest_to_region - 9999999999
        
        cost = np.array([[n[0], self.ws.dist_to_cells_from[region_idx][n[0]] + n[1] - dist_shortest_to_region] for n in neighbors if n[0] not in self.ws.region_idx])
        

        if region_idx in [x[0] for x in neighbors]:
            cost = np.vstack([cost, [region_idx, 0]])

        prob_next = self.compute_Boltzmann(cost[:,1])

        sum_prob_next = sum(prob_next)
        prob_next = self.normalize(prob_next, sum_prob_next)
        
        self.normalized_p = dict()
        for n, p in zip(cost, prob_next):
            self.normalized_p[n[0]] = p
        prob_next = self.normalized_p
        
        return prob_next        
    
    def update_posterior_intent(self, prob_next):
        observation = self.ws.get_node(self.observation[-1])        
        posterior = []
        for idx, r in enumerate(self.ws.region_idx):
            if observation in prob_next[r].keys():
                posterior.append(prob_next[r][observation] * self.prior[idx])
            else:
                posterior.append(0)
        sum_posterior = sum(posterior)
        posterior = [self.normalize(p, sum_posterior) for p in posterior]
        
        self.prior = self.epsilon_transition(posterior)

    def epsilon_transition(self, posterior):
        uniform = 1/len(self.ws.region)
        return [(1 - self.epsilon) * x + self.epsilon * uniform for x in posterior]
    
    def compute_Boltzmann(self, dist):
        p = np.e**(-self.beta * dist)
        return p
    
    def normalize(self, p, sum_p):
        # normalized_p = p / sum_p
        # return normalized_p 
    
        if sum_p == 0:
            return 0
        else:
            return p / sum_p 
        
    def initialize_grid(self):
        # Set all cells zeros
        self.grid = np.zeros((self.MAP_SIZE['x'], self.MAP_SIZE['y']))               
    
    def predict(self, n_simulation=300, steps=1):
        self.n_simulation = n_simulation
        self.counting_grid = np.zeros((self.MAP_SIZE['x'], self.MAP_SIZE['y']))

        sample_n_regions = random.choices(self.ws.region, weights=self.prior, k=n_simulation)
        
        for n in range(n_simulation):
            
            # Sample intent
            sample_r = sample_n_regions[n]
            sample_r_idx = self.ws.get_node(sample_r)

            current_node = self.observation[-1]
            current_node_idx = self.ws.get_node(current_node)
            
            hit_region = (0,)

            for s, _ in enumerate(range(steps)):
                # Sample intent 
                if current_node_idx in self.ws.region_idx and s > 0:
                    p_idx = int(np.where(self.ws.region_idx==current_node_idx)[0])
                    hit_region = hit_region + (p_idx,)
                    d = len(hit_region)
                    new_weights = [self.hm.probability_tree[d][hit_region + (i,)] for i in range(self.ws.NUM_REGIONS)]
                    sample_next_region = random.choices(self.ws.region, weights=new_weights, k=1)[0]
                    sample_r_idx = self.ws.get_node(sample_next_region)
                    # print(hit_region, new_weights, sample_next_region)

                # Find neighbors
                neighbors = np.array([(node, weight) for _, node, weight in self.ws.g.get_out_edges(current_node_idx, [self.ws.g.ep.weight])])
                prob_next = self.inference_one_intent(sample_r_idx, current_node_idx, neighbors)
                sample_next_idx = random.choices(list(prob_next.keys()), weights=list(prob_next.values()), k=1)[0]
                current_node_idx = sample_next_idx
            
            last_position = self.ws.get_position(int(current_node_idx))
            
            # Update counting grid (y-x order for plotting):
            self.counting_grid[last_position[1]][last_position[0]] += 1 

        # Remove previous results
        self.initialize_grid()
        self.grid = self.counting_grid / self.n_simulation
            
#     def plot(self):
#         fig = plt.figure(figsize=(8, 8))
#         plt.cla()
#         ax = plt.gca()
#         ax.set_xticks(np.arange(0, self.MAP_SIZE['x'], 1))
#         ax.set_yticks(np.arange(0, self.MAP_SIZE['y'], 1))
#         ax.set_aspect('equal')
#         ax.grid(color='w', linestyle='-', linewidth=0.1)

#         for o in self.observation:
#             plt.plot(o[0], o[1], 'or', ms = 20)
#         plt.plot(self.observation[0][0], self.observation[0][1], '*w', ms = 15)
        
#         for idx, r in enumerate(self.ws.region):
#             plt.plot(r[0], r[1], 'sy', ms = 20)            
#             plt.text(r[0], r[1], idx, fontsize=12)
            
#         plt.imshow(self.grid, cmap='binary')
#         ax.invert_yaxis()                
        
#         plt.clim(0, 1)
#         plt.colorbar()

#         plt.show()
    