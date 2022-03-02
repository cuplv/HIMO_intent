import toytree
import toyplot.svg
import networkx as nx
import matplotlib.image as mpimg
from collections import deque 
from high_level_monitor import *
from low_level_monitor import *
from itertools import combinations
from common import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class HIMO():
    def __init__(self, LTL_formula, ws, alpha=10, beta_high=1, beta_low=1, epsilon=0.1, tree_depth=4):    
        self.LTL_formula = LTL_formula
        
        self.ws = ws
        self.MAP_SIZE = ws.MAP_SIZE
        
        self.alpha = alpha        
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.epsilon = epsilon
        self.tree_depth = tree_depth
        
        self.generate_additional_ltl()
        self.generate_intent_set()
        self.initialize_monitors()
 
    def generate_additional_ltl(self):
        # Default LTL
        if '1' not in self.LTL_formula:
            self.LTL_formula = ['1'] + self.LTL_formula
        
        # Patrol, avoid, patrol-while-avoid
        patrol_mission_checker = np.zeros(len(self.LTL_formula))
        kitchen_mission_checker = np.zeros(len(self.LTL_formula))        
        for idx, ltl in enumerate(self.LTL_formula):
            if ltl[0] == 'patrol':
                patrol_regions = ltl[1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"                
                patrol_ltl = [f"F p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                # patrol_ltl = "G (" + "".join(patrol_ltl) + ")"
                patrol_ltl = "(" + "".join(patrol_ltl) + ")"                
                self.LTL_formula[idx] = patrol_ltl
                patrol_mission_checker[idx] = 1
                
            elif ltl[0] == 'kitchen':
                self.LTL_formula[idx] = ltl[1]
                kitchen_mission_checker[idx] = 1
                
            elif ltl[0] == 'avoid':
                avoid_regions = ltl[1:]
                assert (max(avoid_regions) < self.ws.NUM_REGIONS) & (min(avoid_regions) >= 0), "Invalid region index"                                
                avoid_ltl = [f"G ! p{r} & " if idx < len(avoid_regions)-1 else f"G ! p{r}" for idx, r in enumerate(avoid_regions)]
                avoid_ltl = "".join(avoid_ltl)
                self.LTL_formula[idx] = avoid_ltl        

            elif ltl[0] == 'seq_patrol':
                patrol_regions = ltl[1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"          

                seq_patrol_ltl = [f"F ( p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                # seq_patrol_ltl = "G (" + "".join(seq_patrol_ltl) + "".join([")"] *len(patrol_regions))
                seq_patrol_ltl = "(" + "".join(seq_patrol_ltl) + "".join([")"] *len(patrol_regions))
                self.LTL_formula[idx] = seq_patrol_ltl
                patrol_mission_checker[idx] = 1                
                                
            elif ltl[0] == 'ord_patrol':
                patrol_regions = ltl[1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"          

                ord_patrol_ltl = [f"F ( p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                ord_patrol_ltl = "G (" + "".join(ord_patrol_ltl) + "".join([")"] *len(patrol_regions))
                not_until = [f"( ! p{patrol_regions[idx+1]} U p{patrol_regions[idx]} ) & " for idx, _ in enumerate(patrol_regions[:-1])]
                no_repeat_until = [f"G ( p{patrol_regions[idx+1]} -> X ( ! p{patrol_regions[idx+1]} U p{patrol_regions[idx]})) & " if idx < len(patrol_regions)-1 else f"G ( p{patrol_regions[0]} -> X ( ! p{patrol_regions[0]} U p{r}))" for idx, r in enumerate(patrol_regions)]
                self.LTL_formula[idx] = ord_patrol_ltl + " & " + "".join(not_until)  + "".join(no_repeat_until)

            elif ltl[0] == 'patrol_while_avoid':
                assert (ltl[1][0] == 'patrol') & (ltl[2][0] == 'avoid'), "Invalid command"
                patrol_regions = ltl[1][1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"
                patrol_ltl = [f"F p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                # patrol_ltl = "G (" + "".join(patrol_ltl) + ")"
                patrol_ltl = "(" + "".join(patrol_ltl) + ")"

                avoid_regions = ltl[2][1:]
                assert (max(avoid_regions) < self.ws.NUM_REGIONS) & (min(avoid_regions) >= 0), "Invalid region index"
                avoid_ltl = [f"G ! p{r} & " if idx < len(avoid_regions)-1 else f"G ! p{r}" for idx, r in enumerate(avoid_regions)]
                avoid_ltl = "".join(avoid_ltl)

                patrol_while_avoid_ltl = patrol_ltl + " & " + avoid_ltl
                self.LTL_formula[idx] = patrol_while_avoid_ltl
                patrol_mission_checker[idx] = 1                
                
            elif ltl[0] == 'seq_patrol_while_avoid':
                assert (ltl[1][0] == 'seq_patrol') & (ltl[2][0] == 'avoid'), "Invalid command"                
                patrol_regions = ltl[1][1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"
                seq_patrol_ltl = [f"F ( p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                # seq_patrol_ltl = "G (" + "".join(seq_patrol_ltl) + "".join([")"] *len(patrol_regions))
                seq_patrol_ltl = "(" + "".join(seq_patrol_ltl) + "".join([")"] *len(patrol_regions))

                avoid_regions = ltl[2][1:]
                assert (max(avoid_regions) < self.ws.NUM_REGIONS) & (min(avoid_regions) >= 0), "Invalid region index"
                avoid_ltl = [f"G ! p{r} & " if idx < len(avoid_regions)-1 else f"G ! p{r}" for idx, r in enumerate(avoid_regions)]
                avoid_ltl = "".join(avoid_ltl)

                seq_patrol_while_avoid_ltl = seq_patrol_ltl + " & " + avoid_ltl
                self.LTL_formula[idx] = seq_patrol_while_avoid_ltl
                patrol_mission_checker[idx] = 1                
                
            elif ltl[0] == 'ord_patrol_while_avoid':
                assert (ltl[1][0] == 'ord_patrol') & (ltl[2][0] == 'avoid'), "Invalid command"                                
                patrol_regions = ltl[1][1:]
                assert (max(patrol_regions) < self.ws.NUM_REGIONS) & (min(patrol_regions) >= 0), "Invalid region index"
                ord_patrol_ltl = [f"F ( p{r} & " if idx < len(patrol_regions)-1 else f"F p{r}" for idx, r in enumerate(patrol_regions)]
                ord_patrol_ltl = "G (" + "".join(ord_patrol_ltl) + "".join([")"] *len(patrol_regions))
                not_until = [f"( ! p{patrol_regions[idx+1]} U p{patrol_regions[idx]} ) & " for idx, _ in enumerate(patrol_regions[:-1])]
                no_repeat_until = [f"G ( p{patrol_regions[idx+1]} -> X ( ! p{patrol_regions[idx+1]} U p{patrol_regions[idx]})) & " if idx < len(patrol_regions)-1 else f"G ( p{patrol_regions[0]} -> X ( ! p{patrol_regions[0]} U p{r}))" for idx, r in enumerate(patrol_regions)]
                ord_patrol_ltl = ord_patrol_ltl + " & " + "".join(not_until)  + "".join(no_repeat_until)
                
                avoid_regions = ltl[2][1:]
                assert (max(avoid_regions) < self.ws.NUM_REGIONS) & (min(avoid_regions) >= 0), "Invalid region index"
                avoid_ltl = [f"G ! p{r} & " if idx < len(avoid_regions)-1 else f"G ! p{r}" for idx, r in enumerate(avoid_regions)]
                avoid_ltl = "".join(avoid_ltl)

                ord_patrol_while_avoid_ltl = ord_patrol_ltl + " & " + avoid_ltl
                self.LTL_formula[idx] = ord_patrol_while_avoid_ltl                
                
        # Regions do not share the same location
        comb = combinations(range(0,self.ws.NUM_REGIONS), 2)

        self.restriction = ['G ! (p'+ str(c[0]) + ' & p' + str(c[1]) + ') & ' for c in comb]
        self.restriction = "".join(self.restriction)
        self.LTL_formula = ["".join([self.restriction, ltl]) for ltl in self.LTL_formula]        
        self.patrol_mission_checker = patrol_mission_checker
        self.kitchen_mission_checker = kitchen_mission_checker
        
    def generate_intent_set(self):
        self.intent_set = deque()
        for idx, ltl in enumerate(self.LTL_formula):
            self.intent_set.append(Intent(ltl, self.patrol_mission_checker[idx], self.kitchen_mission_checker[idx], self.ws, alpha = self.alpha, beta = self.beta_high, tree_depth = self.tree_depth, n_intents = len(self.LTL_formula)))
    
    def print_probab_intents(self):
        print("LTL: probability\n")
        for idx, LTL in enumerate(self.LTL_formula):
            restriction_removed_LTL = LTL.replace(self.restriction, "")
            print(f"{idx}- {restriction_removed_LTL}: {self.hm.probab_intent[idx]}")
        print("-------------------------------------------------------------------")        
  
    def print_probab_short_term_goals(self):
        print("AP(immediate): probability\n")        
        for idx, p in enumerate(self.lm.prior):
            print(f"{idx}: {p}")
        # for idx, (key, v) in enumerate(self.hm.probability_tree[1].items()):
        #     print(f"{idx}: {v}")            
        print("-------------------------------------------------------------------")        
 
    def initialize_monitors(self):
        self.initialize_high_lv_mointor()
        self.initialize_low_lv_monitor()     
        
    def initialize_high_lv_mointor(self):
        self.hm = HighLvMonitor(self.intent_set, self.ws, lb_prob=0.01, tree_depth = self.tree_depth)
    
    def initialize_low_lv_monitor(self):
        self.lm = LowLvMonitor(self.hm, self.ws, self.ws.start_position, beta=self.beta_low, epsilon=self.epsilon)

    def add_observations(self, observed_state_list):
        if len(observed_state_list) > 1:
            observed_state_list = self.fill_in_the_gap(observed_state_list)
        for o in observed_state_list:
            self.hm.add_observation([o])
            self.lm.add_observation(o)
            updated_prior = self.lm.prior
            self.hm.update_short_term_goal(updated_prior)
        self.print_probab_intents()
        self.print_probab_short_term_goals()

    def add_observations_thor(self, observed_state_list, print_probab=False, accept_dist=REGION_ACCEPT_DIST):
        if len(observed_state_list) > 1:
            observed_state_list = self.fill_in_the_gap(observed_state_list)
        for o in observed_state_list:
            self.hm.add_observation_thor([o], accept_dist)
            self.lm.add_observation_thor(o, accept_dist)
            updated_prior = self.lm.prior
            self.hm.update_short_term_goal(updated_prior)
            
        if print_probab:
            self.print_probab_intents()
            self.print_probab_short_term_goals()
            
    def add_observations_running(self, observed_state_list, print_probab=False):
        observed_state_list = self.fill_in_the_gap(observed_state_list)
        for o in observed_state_list:
            self.hm.add_observation([o])
            self.lm.add_observation(o)
            updated_prior = self.lm.prior
            self.hm.update_short_term_goal(updated_prior)
            
        if print_probab:
            self.print_probab_intents()
            self.print_probab_short_term_goals()            
        
    def fill_in_the_gap(self,observation):
        [observation.pop(idx) for idx, _ in enumerate(observation) if idx < len(observation)-1 and observation[idx] == observation[idx+1]]
    
        cnt = 0
        for idx, p in enumerate(observation[1:]):
            current_x, current_y = p[0], p[1]
            s_x, s_y = 1, 1
            prev_x, prev_y = observation[cnt+idx][0], observation[cnt+idx][1]
            if current_x < prev_x:
                s_x = -1
            if current_y < prev_y:
                s_y = -1        

            if prev_x == current_x:
                xlist = [prev_x]
            else:
                xlist = list(range(prev_x, current_x, 1*s_x))
            if prev_y == current_y:
                ylist = [prev_y]          
            else:
                ylist = list(range(prev_y, current_y, 1*s_y))

            xlist_rightmost = xlist[-1]
            ylist_rightmost = ylist[-1]    

            while len(xlist) != len(ylist):
                if len(xlist) < len(ylist):
                    xlist.insert(len(xlist), xlist_rightmost)
                if len(ylist) < len(xlist):
                    ylist.insert(len(ylist), ylist_rightmost)                
            observation.pop(cnt+idx)
            [observation.insert(cnt+idx+i, x) for i, x in enumerate(list(zip(xlist, ylist)))]
            cnt += len(list(zip(xlist, ylist)))-1
        return observation
            
    def prediction(self, n_simulation=300, steps = 1):
        self.lm.predict(n_simulation, steps)
        
    def __tree_to_newick(self, g, root=None):
        if root is None:
            roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
            assert 1 == len(roots)
            root = roots[0][0]    
        subgs = []
        for child in g[root]:
            if len(g[child]) > 0:
                s = self.__tree_to_newick(g, root=child)
                subgs.append(f"{s}{child}:{child}")
            else:
                subgs.append(f"{child}:{child}")
        return "(" + ','.join(subgs) + ")"        

    def plot_map(self, saveimg = False, save_filename = "./result/map.svg"):

        fig = plt.figure(figsize=(18, 12))
        plt.cla()
        ax = plt.gca()
        # ax.set_xticks(np.arange(0, self.MAP_SIZE['x'], 1))
        # ax.set_yticks(np.arange(0, self.MAP_SIZE['y'], 1))
        ax.set_aspect('equal')
        ax.grid(color='w', linestyle='-', linewidth=0.1)
        
        for idx, r in enumerate(self.ws.region):
            plt.plot(r[0], r[1], 'sy', ms = 20)            
            # plt.plot(r[0], r[1], 'sy', ms = 10)
            plt.text(r[0], r[1], idx, fontsize=15)

        # for o in self.lm.observation[-5:]:
            # plt.plot(o[0], o[1], 'or', ms = 20)
            # plt.plot(o[0], o[1], 'or', ms = 10)            
        # plt.plot(self.lm.observation[0][0], self.lm.observation[0][1], '*w', ms = 15)
        plt.plot(self.lm.observation[-5:,0], self.lm.observation[-5:,1], '-or', ms = 5)
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'or', ms = 20)        
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'ow', ms = 5)        
        # plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'ow', ms = 2)        
        
        self.lm.grid[self.lm.grid<0.05] = 0
        plt.imshow(self.lm.grid, cmap='binary')
        ax.invert_yaxis()                
        
        plt.clim(0, 1)
        plt.colorbar()

        # plt.show()
        
        if saveimg:
            plt.savefig(save_filename, bbox_inches="tight")
            plt.close(fig)        
        
    def plot_map_running(self, observation, saveimg = False, save_filename = "./result/map.svg"):

        fig = plt.figure(figsize=(18, 12))
        plt.cla()
        ax = plt.gca()
        # ax.set_xticks(np.arange(0, self.MAP_SIZE['x'], 1))
        # ax.set_yticks(np.arange(0, self.MAP_SIZE['y'], 1))
        ax.set_aspect('equal')
        
        label = ['0:pot', '1:water', '2:noodle', '3:meat', '4:cooktop', '5:oven', '6:delivery location']
        for idx, r in enumerate(self.ws.region):
            plt.plot(r[0], r[1], 'sk', ms = 20)            
            # plt.plot(r[0], r[1], 'sy', ms = 10)
            plt.text(r[0]-0.5, r[1]+0.3, label[idx], fontsize=15)

        # for o in self.lm.observation[-5:]:
            # plt.plot(o[0], o[1], 'or', ms = 20)
            # plt.plot(o[0], o[1], 'or', ms = 10)            
        # plt.plot(self.lm.observation[0][0], self.lm.observation[0][1], '*w', ms = 15)
        observation = np.array(observation)
        plt.plot(observation[-6:,0], observation[-6:,1], '--or', ms=5, linewidth=2)
        plt.plot(observation[-1,0], observation[-1,1], '-*r', ms=15, linewidth=2)        
        plt.plot(self.lm.observation[-4:,0], self.lm.observation[-4:,1], '-or', ms = 5, linewidth=2)
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'or', ms = 20)        
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'ow', ms = 5)        
        # plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'ow', ms = 2)        

        self.lm.grid[self.lm.grid<0.03] = 0
        plt.imshow(self.lm.grid, cmap='binary')
        plt.clim(0, 0.5)
        plt.colorbar()
        
        # plot kitchen
        img = mpimg.imread('kitchen.png')
        img_cropped = img[10:510, 30:1018, :]
        plt.imshow(img, extent=[-1,10, -1, 10])
        
        minor_ticks = np.arange(0, 10, 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)        
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.1)
        # plt.show()
        
        # ax.invert_yaxis()                
                
        if saveimg:
            plt.savefig(save_filename, bbox_inches="tight")
            plt.close(fig)        
                
    def plot_map_thor(self, observation, region_accept_dist, saveimg = False, save_filename = "./result/map.svg"):
        
        fig = plt.figure(figsize=(18, 6))
        plt.cla()
        ax = plt.gca()

        # ax.set_xticks(np.arange(0, self.MAP_SIZE['x'], 1))
        # ax.set_yticks(np.arange(0, self.MAP_SIZE['y'], 1))
        ax.set_aspect('equal')
        ax.grid(color='w', linestyle='-', linewidth=0.1)
        
        for idx, r in enumerate(self.ws.region):
            # plt.plot(r[0], r[1], 'sy', ms = 20)            
            plt.plot(r[0], r[1], 'sy', ms = 10)
            plt.text(r[0], r[1], idx, fontsize=15)

        # for o in self.lm.observation[-5:]:
            # plt.plot(o[0], o[1], 'or', ms = 20)
        observation = np.array(observation)
        plt.plot(observation[-6:,0], observation[-6:,1], '--r', ms=5, linewidth=2)
        plt.plot(observation[-1,0], observation[-1,1], '-*r', ms=15, linewidth=2)        
        plt.plot(self.lm.observation[-10:,0], self.lm.observation[-10:,1], '-or', ms = 3)
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'or', ms = 10)        
        plt.plot(self.lm.observation[-1][0], self.lm.observation[-1][1], 'ow', ms = 2)        
        
        self.lm.grid[self.lm.grid<0.01] = 0
        plt.imshow(self.lm.grid, cmap='binary')
        plt.clim(0, 0.1)
        plt.colorbar()
        
        # plot office map
        img = mpimg.imread('map1.png')
        img_cropped = img[10:510, 30:1018, :]
        plt.imshow(img_cropped, extent=[0,100, 0, 50])
        
        # plot head
        # im = OffsetImage(plt.imread('head2.png'), zoom=0.03)
        # ab = AnnotationBbox(im, (self.lm.observation[-1][0],self.lm.observation[-1][1]), xycoords='data', frameon=False)
        # plt.gca().add_artist(ab)
        
        
        plt.ylim([50,0])
        ax.invert_yaxis()                
        
        # plot circle around region
        for idx, r in enumerate(self.ws.region):
            radius = region_accept_dist[idx]
            circle = plt.Circle((r[0], r[1]), region_accept_dist[idx], color='b', fill=False, linestyle='--')
            ax.add_patch(circle)
        
        if saveimg:
            plt.savefig(save_filename, bbox_inches="tight")
            plt.close(fig)        
                
    def __remove_node(self, nodeID):
        if self.G.has_node(nodeID):
            succ = list(self.G.successors(nodeID))
            if len(succ)!=0:
                for s in succ:
                    self.__remove_node(s)
            self.G.remove_node(nodeID)
    
    def plot_tree(self, saveimg = False, tree_prob_threshold=0.1, tree_height=500, tree_width=500, save_filename = "./result/tree-plot.svg"):
        self.G = nx.DiGraph()

        plot_node_ID = dict()
        plot_node_prob = []
        plot_node_key = []
        cnt = 0

        for d in range(self.tree_depth):
            for key in self.hm.probability_tree[d]:
                plot_node_ID[key] = cnt
                plot_node_key.append(key[1:])
                plot_node_prob.append(self.hm.probability_tree[d][key])
                self.G.add_node(cnt)
                cnt+=1

        colors = []
        remove_list = []
        for newKey in plot_node_ID:
            parent = newKey[:-1]
            if parent in plot_node_ID:
                newKeyID = plot_node_ID[(newKey)]
                parentID = plot_node_ID[(parent)]
                self.G.add_edge(parentID, newKeyID)

                prob = plot_node_prob[newKeyID]
                if prob < tree_prob_threshold:
                    remove_list.append(newKeyID)

            AP = newKey[-1]
            colors.append(AP)

        for i in remove_list:
            self.__remove_node(i)
            
        newick = f"{self.__tree_to_newick(self.G, None)}0:0;"
        tre0 = toytree.tree(newick, tree_format=3)
        tre0 = tre0.set_node_values(feature = 'dist', values={idx:1 for idx in tre0.idx_dict})        
        G_node = tre0.get_node_dict(return_internal=True)
        G_node = {n: int(G_node[n]) for n in reversed(range(len(G_node)))}

        node_size_scaler = 30
        node_size_lb = 10 # lower bound
        sizes = [(plot_node_prob[p]*node_size_scaler) + node_size_lb for p in G_node.values()]
        sizes[0] = 30
        APs = [colors[c] for c in G_node.values()]
        APs[0] = ""
        colors = [colors[c] for c in G_node.values()]
        plot_node_key = [plot_node_key[p] for p in G_node.values()]
        colors = ['lightgray' if x ==0 else x for x in colors]
        colors = ['lightgreen' if x ==1 else x for x in colors]
        colors = ['lightblue' if x ==2 else x for x in colors]
        colors = ['khaki' if x ==3 else x for x in colors]
        colors = ['lightseagreen' if x ==4 else x for x in colors]
        colors = ['lightsalmon' if x ==5 else x for x in colors]        
        colors[0]='pink'
        # e_colors = list(reversed(colors[1:]))
        ewidths = [((x-node_size_lb)/node_size_scaler) * 5 + 1 for x in list(reversed(sizes[1:]))]        
        
        labels_idx = [int(label) for label in tre0.get_tip_labels()]
        labels_idx = [list(G_node.values()).index(x) for x in labels_idx]
        labels = [f"{plot_node_key[l]}: {(sizes[l]-node_size_lb)/node_size_scaler:.4f}" for l in labels_idx]        

        self.tre0 = tre0
        
        canvas, axes, mark = tre0.draw(node_sizes=sizes, node_colors=colors, node_hover=True, tree_style='n', 
          node_labels=APs,  
          # edge_colors = e_colors,
          edge_widths=ewidths,
          node_labels_style={
                "fill": "black",
                "font-size": "20px",
            },
          tip_labels=labels,
          tip_labels_align=True,
          height=tree_height,
          width=tree_width); 
        
        if saveimg:
            toyplot.svg.render(canvas, save_filename)

        
    def plot_probability_of_APs(self, saveimg = False, save_filename = "./result/hist.svg"):
        fig = plt.figure(figsize=(18, 3)) 

        for d in range(1, self.tree_depth):
            # print(f"Goal #{d}: ")
            sum_probab = []
            ax = plt.subplot(1, 4, d)   
            for AP in range(self.ws.NUM_REGIONS):
                sum_probab_print = sum([list(self.hm.probability_tree[d].values())[idx] for idx, x in enumerate(list(self.hm.probability_tree[d].keys())) if x[-1] == AP])
                sum_probab.append(sum_probab_print)
            #     print(f"{AP}: {sum_probab_print:.4f}")
            # print("-------------------------")
            plt.bar(range(self.ws.NUM_REGIONS), sum_probab)
            plt.ylim([0,1])
            plt.xticks(range(self.ws.NUM_REGIONS))
            plt.xlabel("Atomic proposition label")
            plt.ylabel("Probability")  
            if d == 1:
                plt.title(f"1st goal")    
            elif d == 2:
                plt.title(f"2nd goal")
            elif d == 3:
                plt.title(f"3rd goal")
            else:
                plt.title(f"{d}th goal")                
            for container in ax.containers:
                plt.bar_label(container,  fmt='%.3f' )
        
        if saveimg:
            plt.savefig(save_filename, bbox_inches="tight")
            plt.close(fig)        


    def plot(self, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map()
        self.plot_tree(tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs()
        
    def plot_thor(self, observation, region_accept_dist, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map_thor(observation, region_accept_dist)
        self.plot_tree(tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs()
        
    def plot_running(self, observation, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map_running(observation)
        self.plot_tree(tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs()        
        
    def save_plot(self, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map(saveimg = True)
        self.plot_tree(saveimg = True, tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs(saveimg = True)    
        self.plot_probab_intents()        
        
    def save_plot_thor(self, observation, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map_thor(observation, saveimg = True)
        self.plot_tree(saveimg = True, tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs(saveimg = True)                
        self.plot_probab_intents_thor()
        
    def save_plot_running(self, observation, tree_prob_threshold=0.1, tree_height=500, tree_width=500):
        self.plot_map_running(observation, saveimg = True)
        self.plot_tree(saveimg = True, tree_prob_threshold=tree_prob_threshold, tree_height=tree_height, tree_width=tree_width)
        self.plot_probability_of_APs(saveimg = True)    
        self.plot_probab_intents()               
        
    def plot_probab_intents_thor(self):
        probable_ltl = np.array(self.LTL_formula)[np.argsort(self.hm.probab_intent)[:10]]
        probable_ltl_probab = np.array(self.hm.probab_intent)[np.argsort(self.hm.probab_intent)[:10]]
        
        label_ltl = ['default', 'visitor', 'visitor', 'visitor', 'visitor', 'visitor', 'visitor', 'lab worker', 'utility worker']
        label_ltl = np.array(label_ltl)[np.argsort(self.hm.probab_intent)[:10]]
        
        cnt = 0
        for i, LTL in enumerate(probable_ltl):
            if probable_ltl_probab[i] > 0.01:
                restriction_removed_LTL = LTL.replace(self.restriction, "")
                ltl_with_probab = f"{restriction_removed_LTL} : {probable_ltl_probab[i]:.3f} -- {label_ltl[i]}"
                plt.text(0,0.95-cnt*0.1, ltl_with_probab) 
                cnt += 1
        plt.axis('off')
        
        save_filename = "./result/ltl.svg"
        plt.savefig(save_filename, bbox_inches="tight")        
        plt.close()   
        
    def plot_probab_intents(self):
        probable_ltl = np.array(self.LTL_formula)[np.argsort(self.hm.probab_intent)[:10]]
        probable_ltl_probab = np.array(self.hm.probab_intent)[np.argsort(self.hm.probab_intent)[:10]]
        
        cnt = 0
        for i, LTL in enumerate(probable_ltl):
            if probable_ltl_probab[i] > 0.01:
                restriction_removed_LTL = LTL.replace(self.restriction, "")
                ltl_with_probab = f"{restriction_removed_LTL} : {probable_ltl_probab[i]:.3f}"
                plt.text(0,0.95-cnt*0.1, ltl_with_probab) 
                cnt += 1
        plt.axis('off')
        
        save_filename = "./result/ltl.svg"
        plt.savefig(save_filename, bbox_inches="tight")        
        plt.close()             