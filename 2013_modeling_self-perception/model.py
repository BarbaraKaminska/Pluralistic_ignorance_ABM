import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import mesa
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
import networkx as nx
from sklearn.cluster import KMeans

def average_opinion(model):
    """Calculate the average opinion of all agents in the model."""
    return np.mean([agent.op for agent in model.agents])

def average_attitude(model):
    """Calculate the average attitude of all agents in the model."""
    return np.mean([agent.att for agent in model.agents])

def std_opinion(model):
    """Calculate the standard deviation of opinions of all agents in the model."""
    return np.std([agent.op for agent in model.agents])

def std_attitude(model):
    """Calculate the standard deviation of attitudes of all agents in the model."""
    return np.std([agent.att for agent in model.agents])


class Agent(Agent):
    def __init__(self, model, x, y, u = 0.5):
        super().__init__(model)
        self.op = np.random.rand()
        self.att = self.op
        self.u = u
        self.green = 1 - self.u
        self.red = model.RED if self.green < model.RED else self.green
        self.x = x
        self.y = y
        self.model = model

    def update_opinion(self):
        
        neighbors = self.model.grid.get_neighbors((self.x, self.y), moore=True, include_center=False)
        neighbor_opinions = np.array([agent.op for agent in neighbors])
        g = float('inf')
        for n_clusters in range(1, 5):  
            try:    
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(neighbor_opinions.reshape(-1, 1))
                cluster_centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_
                # print(f"Cluster centers for {n_clusters} clusters: {cluster_centers.flatten()}")
                g_temp = 0
                for i, op in enumerate(neighbor_opinions):
                    center = cluster_centers[cluster_labels[i]]
                    g_temp += np.linalg.norm(op - center) ** 2

                if g_temp < g:
                    g = g_temp
                    best_n_clusters = n_clusters
                    best_labels = cluster_labels
                    best_centers = cluster_centers
            except:
                continue

        labels_count = np.bincount(best_labels)
        max_label = np.argmax(labels_count)
        if g < self.model.G:
            if self.u > self.model.U:
                self.op = best_centers[max_label][0]
                self.att = self.op
            else:
                closest_group = np.argmin(np.abs(best_centers[:, 0] - self.att))
                x_ = labels_count[closest_group]
                if 1 <= x_ < 2:
                    f_a = 0.1 * x_
                elif 2 <= x_ < 3:
                    f_a = 0.2 * x_
                elif 3 <= x_ < 4:
                    f_a = 0.5 * x_
                elif 4 <= x_ <= 5:
                    f_a = 0.2 * x_
                else:
                    f_a = 1

                x_1 = np.std(neighbor_opinions[best_labels == closest_group])
                f_b = 1/(1+ np.exp(24*x_1 - 6))

                x_2 = np.min([np.abs(self.att - center) for center in best_centers])
                f_c = 1/(1 + np.exp(-12*x_2 + 6))

                O = np.mean([f_a, f_b, f_c])

                if O >= self.red: 
                    self.op = best_centers[closest_group][0]

                elif O >= self.green:
                    self.op = best_centers[closest_group][0]
                    self.att = self.op

        else:
            # Step 4: fallback individual averaging
            compatible_neighbors = [a.op for a in neighbors if abs(a.op - self.op) < self.model.BC]
            if compatible_neighbors:
                avg_op = np.mean(compatible_neighbors + [self.op])
                self.op = avg_op 
                self.att = avg_op


class PluralisticIgnoranceModel(Model):
    def __init__(self, N = 24, G = 2, U = 0.8, RED = 0.6):

        super().__init__()
        self.N = N
        self.L = int(np.sqrt(N))
        self.grid = MultiGrid(self.L, self.L, True)
        self.data_collector = DataCollector(
            agent_reporters={"opinion": "op", "attitude": "att"}, 
            model_reporters={"average_opinion": average_opinion, "average_attitude": average_attitude, 
            "std_opinion": std_opinion, "std_attitude": std_attitude}
        )
        self.G = G
        self.U = U
        self.RED = RED
        self.BC = 0.5

        for x in range(self.L):
            for y in range(self.L):
                agent = Agent(self, x, y)
                self.grid.place_agent(agent, (x, y))

        
    def step(self):
        self.data_collector.collect(self)
        for agent in self.agents:
            agent.update_opinion()

    def run_model(self, steps):
        for _ in range(steps):
            self.step()