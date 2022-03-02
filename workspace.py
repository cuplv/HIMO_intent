import graph_tool.all as gt
import numpy as np
import random
from itertools import permutations

class Workspace:
    def __init__(self, MAP_SIZE, NUM_REGIONS, start_position, region=0):
        self.MAP_SIZE = MAP_SIZE
        self.NUM_VERTICES = MAP_SIZE['x'] * MAP_SIZE['y']
        self.NUM_REGIONS = NUM_REGIONS
        assert self.NUM_REGIONS < self.NUM_VERTICES, "Number of regions is bigger than number of verticies!"
        
        self.start_position = start_position
        self.g = gt.Graph()
    
        # get xy position of nodes
        self.coordinates= np.array([(x, y) for y in range(self.MAP_SIZE['y']) for x in range(self.MAP_SIZE['x'])])

        # add edges
        self.add_edges()
        
        # add regions and update corresponding weights
        if region == 0:
            self.region = self.generate_random_regions_by_coordinates()
        else:
            self.region = region
        self.region_idx = np.array([self.get_node(r) for r in self.region])
        self.region_dict = dict([(r, idx) for r, idx in zip(self.region, self.region_idx)])
        self.add_regions()

    # edges --------------------------------------------------------------------------------
    # Edges: Up<->down, left<->right, upright<->leftdown, upleft<->rightdown
    def add_edges(self):
        self.g.edge_properties['weight'] = self.g.new_edge_property("double")
        
        # Add edges with weights to dict
        edges = []
        diag_edges = []

        # edge: left-right, up-down, upright, upleft
        [edges.append((idx, idx + 1)) for idx in range(self.NUM_VERTICES) if (idx + 1) % self.MAP_SIZE['y'] != 0]
        [edges.append((idx, idx + self.MAP_SIZE['y'])) for idx in range(self.NUM_VERTICES - self.MAP_SIZE['y'])]
        [diag_edges.append((idx, idx + self.MAP_SIZE['y'] + 1)) for idx in range(self.NUM_VERTICES - self.MAP_SIZE['y']) if
         (idx + 1) % self.MAP_SIZE['y'] != 0]
        [diag_edges.append((idx, idx - self.MAP_SIZE['y'] + 1)) for idx in range(self.MAP_SIZE['y'], self.NUM_VERTICES) if
         (idx + 1) % self.MAP_SIZE['y'] != 0]

        reverse_edges = [(y, x) for x, y in edges]
        reverse_diag_edges = [(y, x) for x, y in diag_edges]
        edges = edges + reverse_edges + diag_edges + reverse_diag_edges

        self.g.add_edge_list(edges)

        # Init weight values
        self.g.edge_properties['weight'].fa = 1
        self.g.edge_properties['weight'].fa[len(edges) - len(diag_edges) - len(reverse_diag_edges):] = 1.4
    # edges --------------------------------------------------------------------------------
    
    # regions --------------------------------------------------------------------------------
    def add_regions(self):
        for r in self.region_idx:
            self.update_edge_weight_to(r)  
        
        self.dist_to_cells_from = dict()
        for r in self.region_idx:    
            self.dist_to_cells_from[r] = self.run_shortest_path_length_from(r)
        
        self.distance_between_regions()
        
    def distance_between_regions(self):
        perm = permutations(self.region_idx, 2)

        dist_between_regions_idx = dict()
        for t in list(perm):
            start_region_idx, goal_region_idx = t[0], t[1]
            dist = np.array(self.dist_to_cells_from[start_region_idx].fa, dtype=float)[goal_region_idx] - 9999999999
            dist_between_regions_idx[(start_region_idx, goal_region_idx)] = dist
            # dist_between_regions_idx[(start_region_idx, start_region_idx)] = 0 # self loop

        # from current to regions
        current_pos_idx = self.get_node(self.start_position)
        for r_idx in self.region_idx:
            dist = np.array(self.dist_to_cells_from[r_idx].fa, dtype=float)[current_pos_idx]
            dist_between_regions_idx[(-1, r_idx)] = dist    
        self.dist_between_regions_idx = dist_between_regions_idx

    def set_pre_defined_obstacle(self, obs_list):
        for o in obs_list:
            node_idx = self.get_node(o)
            self.update_edge_weight_to(node_idx)
        
    def update_edge_weight_to(self, node_idx):
        e_in = self.g.get_in_edges(node_idx, [self.g.edge_index])
        for e in e_in:
            e_idx = e[2]
            self.g.ep.weight.fa[e_idx]+=9999999999 # Edges going to regions have large weights.
            
    def run_shortest_path_length_from(self, starting_node):
        return gt.shortest_distance(self.g, source=starting_node, weights=self.g.ep.weight)    
    # regions --------------------------------------------------------------------------------
    
    # helpers --------------------------------------------------------------------------------
    def get_position(self, node):
        # return x-, y-cooridnate of node
        return self.coordinates[node]
    
    def get_node(self, position):
        # return node index
        # return self.coordinates.index(position)  
        assert position[0] < self.MAP_SIZE['x']
        assert position[1] < self.MAP_SIZE['y']
        return self.MAP_SIZE['x'] * position[1] + position[0] # faster
    
    def plot_graph(self):
        # assert self.NUM_VERTICES <= 100, "Too many vertices to draw!"
        # add vertex property - xy coordinates
        self.g.vertex_properties['pos'] = self.g.new_vertex_property("vector<double>", self.coordinates)
        gt.graph_draw(self.g, pos=self.g.vertex_properties['pos'])
        
    def generate_random_regions_by_index(self):
        vertex = np.arange(self.NUM_VERTICES).reshape(self.MAP_SIZE['x'], -1)
        start_vertex = vertex[self.start_position[0], self.start_position[1]]
        region = random.sample(range(0, self.NUM_VERTICES), self.NUM_REGIONS + 1)
        region = list(filter(lambda x: x != start_vertex, region))[:self.NUM_REGIONS]    
        return region

    def generate_random_regions_by_coordinates(self):
        region = []
        while len(region)<self.NUM_REGIONS:
            x = random.sample(range(0, self.MAP_SIZE['x']), 1)[0]
            y = random.sample(range(0, self.MAP_SIZE['y']), 1)[0]

            if (x, y) != self.start_position and (x, y) not in region:
                region.append((x,y))
        return region            
