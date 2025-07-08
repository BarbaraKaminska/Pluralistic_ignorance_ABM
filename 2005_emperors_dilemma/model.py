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

class Agent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.neighbors = list(self.model.graph.neighbors(self.unique_id-1))
        self.N_neighbors = len(self.neighbors) 
        # belief B
        self.B = np.random.choice([-1, 1])   
        # strength of conviction S
        self.S = 1 if B==1 else np.random.random()
        # compliance C
        self.C = -B if -B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S else B
        # need for enforcement W
        self.W = (1 - self.B/self.N_neighbors*np.sum([a.C for a in self.neighbors]) )/2
        # cost K
        self.K = model.K
        # enforcement E
        if (-B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S + self.K) and B != C:
            self.E = B
        elif S*W>K and B==C:
            self.E = -B
        else: 
            self.E = 0 

    def if_enforce():
        if (-B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S + self.K) and B != C:
            self.E = B
        elif S*W>K and B==C:
            self.E = -B
        else: 
            self.E = 0 

    def if_comply():
        self.C = -B if -B/N_neighbors*np.sum([a.E for a in self.neighbors]) > self.S else B        


class Model(Model):
    def __init__(self, N, K):
        super().__init__()
        self.N = N
        self.K = K