import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mesa
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.agent import Agent
from mesa.space import MultiGrid
import networkx as nx

# === Model Code ===

def cE(model):
    E1 = np.count_nonzero([a.E == a.B for a in model.all_agents])
    Em1 = np.count_nonzero([a.E == -a.B for a in model.all_agents])
    E0 = np.count_nonzero([a.E == 0 for a in model.all_agents])
    return [E1/model.N, E0/model.N, Em1/model.N]

def cC(model):
    return (np.mean([a.C for a in model.all_agents]) + 1) / 2

def cB(model):
    return (np.mean([a.B for a in model.all_agents]) + 1) / 2

def false_believers(model):
    return np.count_nonzero([a.B == -1 and a.C == 1 for a in model.all_agents]) / model.N

class Agent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.neighbors = [] # list(self.model.graph.neighbors(self.unique_id))
        self.N_neighbors = None # len(self.neighbors) 
        # belief B
        if np.random.random() < model.N_believers/100:
            self.B = 1
        else:
            self.B = -1
        # strength of conviction S
        self.S = 1 if self.B==1 else np.random.uniform(0, 0.38) # check if this is correct
        # compliance C
        self.C = np.random.choice([-1, 1]) # -self.B if -B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S else B
        # need for enforcement W
        self.W = np.random.random() # (1 - self.B/self.N_neighbors*np.sum([a.C for a in self.neighbors]) )/2
        # cost K
        self.K = model.K
        # enforcement E
        self.E = np.random.choice([-1, 0, 1]) 

        # if (-B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S + self.K) and B != C:
        #     self.E = B
        # elif S*W>K and B==C:
        #     self.E = -B
        # else: 
        #     self.E = 0 

    def if_enforce(self):
        if (-self.B/self.N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S + self.K) and self.B != self.C:
            self.E = -self.B
        elif self.S*self.W>self.K and self.B==self.C:
            self.E = self.B
        else: 
            self.E = 0 

    def if_enforce_message(self):
        self.if_enforce()
        if self.E == 1:
            mess = f"Enforcing the norm: E = {self.E}." 
        elif self.E == -1:
            mess = f"Enforcing deviation from the norm: E = {self.E}."
        else:
            mess = f"Not enforcing the norm: E = {self.E}."

        return mess


    def if_comply(self):
        # print(self.neighbors)
        self.C = -self.B if -self.B/self.N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S else self.B

    def if_comply_message(self):
        mess = f"Information from neighbors: {-self.B/self.N_neighbors*np.sum([a.E for a in self.neighbors]):.2f}; Strength of conviction: {self.S:.2f}; "
        self.C = -self.B if -self.B/self.N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S else self.B
        if self.C == 1:
            mess += f"Complying with the norm: C = {self.C}."    
        else:
            mess += f"Not complying with the norm: C = {self.C}."
        return mess

    # def update_strength_of_conviction(self):
    #     if self.B == 1:
    #         self.S = 1
    #     else:
    #         self.S = np.random.random()

    def update_need_for_enforcement(self):
        self.W = (1 - self.B/self.N_neighbors*np.sum([a.C for a in self.neighbors]) )/2

    def update_need_for_enforcement_message(self):
        mess = f"Need for enforcement: W = {self.W:.2f}; "
        return mess


class EmperorsDilemma(Model):
    def __init__(self, N, K, N_bel, network_type="square lattice", k=0.125, beta=0.5, neighborhood_type="Moore"):
        super().__init__()
        self.N = N
        self.K = K
        self.N_believers = N_bel
        if network_type == "square lattice":
            self.graph = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
            if neighborhood_type == "Moore":
                for x in range(int(np.sqrt(N))):
                    for y in range(int(np.sqrt(N))):
                        if x+1 < int(np.sqrt(N)) and y+1 < int(np.sqrt(N)):
                            self.graph.add_edge((x, y), (x+1, y+1))  # diagonal down-right
                        if x+1 < int(np.sqrt(N)) and y-1 >= 0:
                            self.graph.add_edge((x, y), (x+1, y-1))  # diagonal up-right
            else:
                for x in range(int(np.sqrt(N))):
                    for y in range(int(np.sqrt(N))):
                        if x+1 < int(np.sqrt(N)):
                            self.graph.add_edge((x, y), (x+1, y))  # right
                        if y+1 < int(np.sqrt(N)):
                            self.graph.add_edge((x, y), (x, y+1))  # down
            print(neighborhood_type)
        elif network_type == "Watts-Strogatz":
            self.graph = nx.watts_strogatz_graph(N, k=k, p=beta)
        self.all_agents = []
        for node in self.graph.nodes:
            agent = Agent(self)
            self.all_agents.append(agent)
            self.graph.nodes[node]["agent"] = agent
        self.datacollector = DataCollector(
            model_reporters={"cC": cC, "cE": cE, "cB": cB}, 
            agent_reporters={"B": "B", "C": "C", "E": "E", "S": "S", "W": "W"})

        # print(self.graph.nodes)

        for agent, node in zip(self.all_agents, self.graph.nodes):
            nodes_neighbors = self.graph.neighbors(node)
            for neighbor in nodes_neighbors:
                if neighbor in self.graph.nodes:
                    agent.neighbors.append(self.graph.nodes[neighbor]["agent"])

            # agent.neighbors = [node["agent"] for node in nodes_neighbors]
            agent.N_neighbors = len(agent.neighbors)

    def step(self):
        self.datacollector.collect(self)
        for agent in self.all_agents:
            agent.if_comply()   
            agent.update_need_for_enforcement()
            agent.if_enforce()

    def run_model(self, steps):
        for _ in range(steps):
            self.step()
