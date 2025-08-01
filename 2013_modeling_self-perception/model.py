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

        self.best_n_clusters = None
        self.best_labels = None 
        self.best_centers = None
        self.g = float('inf')
        self.neighbors = []
        

    def update_opinion(self):
        
        
        neighbor_opinions = np.array([agent.op for agent in self.neighbors])
        self.best_n_clusters = None
        self.best_labels = None 
        self.best_centers = None
        self.g = float('inf')
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

                if g_temp < self.g:
                    self.g = g_temp
                    self.best_n_clusters = n_clusters
                    self.best_labels = cluster_labels
                    self.best_centers = cluster_centers
            except:
                continue

        labels_count = np.bincount(self.best_labels)
        max_label = np.argmax(labels_count)
        if self.g < self.model.G:
            if self.u > self.model.U:
                self.op = self.best_centers[max_label][0]
                self.att = self.op
            else:
                closest_group = np.argmin(np.abs(self.best_centers[:, 0] - self.att))
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

                x_1 = np.std(neighbor_opinions[self.best_labels == closest_group])
                f_b = 1/(1+ np.exp(24*x_1 - 6))

                x_2 = np.min([np.abs(self.att - center) for center in self.best_centers])
                f_c = 1/(1 + np.exp(-12*x_2 + 6))

                O = np.mean([f_a, f_b, f_c])

                if O >= self.red: 
                    self.op = self.best_centers[closest_group][0]

                elif O >= self.green:
                    self.op = self.best_centers[closest_group][0]
                    self.att = self.op

        else:
            # Step 4: fallback individual averaging
            compatible_neighbors = [a.op for a in neighbors if abs(a.op - self.op) ]
            if compatible_neighbors:
                avg_op = np.mean(compatible_neighbors + [self.op])
                self.op = avg_op 
                self.att = avg_op

    def update_step1_message(self):
        # Step 1: Calculate the best clustering of neighbors' opinions
        neighbor_opinions = np.array([agent.op for agent in self.neighbors])
        self.g = float('inf')
        # print(f"Initial group number g = {self.g}")
        for n_clusters in range(1, 5):  
            try:    
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(neighbor_opinions.reshape(-1, 1))
                cluster_centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_
                # print(f"Cluster centers for {n_clusters} clusters: {cluster_centers.flatten()}")
                g_temp = float(0)
                # print(f"{g_temp=}, {self.g=}")
                print(f"Calculating group number g for {n_clusters} clusters")
                print(f"Neighbor opinions: {neighbor_opinions}")
                for i, op in enumerate(neighbor_opinions):
                    center = cluster_centers[cluster_labels[i]]
                    g_temp += np.linalg.norm(op - center) ** 2
                    print(f"{g_temp}")

                if g_temp < self.g:
                    self.g = g_temp
                    self.best_n_clusters = n_clusters
                    self.best_labels = cluster_labels
                    self.best_centers = cluster_centers
            except:
                print(f"Failed to fit KMeans for {n_clusters} clusters, continuing with next n_clusters")
                continue
        print(f"Final group number g = {self.g}")
        message = f"Number of clusters: {self.best_n_clusters}, group number g = {self.g:.4f}; "
        return message, "2"

    def update_step2_message(self):
        if self.g < self.model.G:
            message = f"Group number g = {self.g:.2f} is less than model G = {self.model.G:.2f}; "
            if self.u > self.model.U:
                message += f", individual uncertainty u = {self.u:.2f} is greater than model U = {self.model.U:.2f}"
                return message, "2a"
            else:
                message += f", individual uncertainty u = {self.u:.2f} is less than model U = {self.model.U:.2f}"
                return message, "3a"
        else:
            message = f"Group number g = {self.g:.4f} is not less than model G = {self.model.G:.1f}"
            return message, "4"

    def update_step2a_message(self):
        message = f"Private acceptance;"
        labels_count = np.bincount(self.best_labels)
        max_label = np.argmax(labels_count)
        self.op = self.best_centers[max_label][0]
        self.att = self.op
        message += f" agent's opinion and attitude are updated to center of largest cluster: {self.op:.2f}"
        return message, "0"

    def update_step3a_message(self):
        message = f"Decide whether to follow the group;"

        closest_group = np.argmin(np.abs(self.best_centers[:, 0] - self.att))
        x_ = np.bincount(self.best_labels)[closest_group]
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

        neighbor_opinions = np.array([agent.op for agent in self.neighbors])
        x_1 = np.std(neighbor_opinions[self.best_labels == closest_group])
        f_b = 1/(1+ np.exp(24*x_1 - 6))

        x_2 = np.min([np.abs(self.att - center) for center in self.best_centers])
        f_c = 1/(1 + np.exp(-12*x_2 + 6))

        O = np.mean([f_a, f_b, f_c])

        if O >= self.red: 
            self.op = self.best_centers[closest_group][0]
            message += f"only agent's opinion is updated to center of closest cluster: {self.op:.2f}"

        elif O >= self.green:
            self.op = self.best_centers[closest_group][0]
            self.att = self.op
            message += f" agent's opinion and attitude are updated to center of closest cluster: {self.op:.2f}"
        else:
            message += f" agent's opinion and attitude remain unchanged: {self.op:.2f}"
        return message, "0"

    def update_step4_message(self):
        compatible_neighbors = [a.op for a in self.neighbors if abs(a.op - self.op) < self.model.U]
        if compatible_neighbors:    
            avg_op = np.mean(compatible_neighbors + [self.op])
            self.op = avg_op 
            self.att = avg_op
        message = f"Agent's opinion and attitude are updated to average of compatible neighbors: {self.op:.2f}"
        return message, "0"

class PluralisticIgnoranceModel(Model):
    def __init__(self, N = 24, G = 2, U = 0.8, RED = 0.6):

        super().__init__()
        self.N = N
        self.L = int(np.sqrt(N))
        self.G = G
        self.U = U
        self.RED = RED
        self.grid = MultiGrid(self.L, self.L, True)
        self.data_collector = DataCollector(
            agent_reporters={"opinion": "op", "attitude": "att"}, 
            model_reporters={"average_opinion": average_opinion, "average_attitude": average_attitude, 
            "std_opinion": std_opinion, "std_attitude": std_attitude}
        )

        for x in range(self.L):
            for y in range(self.L):
                agent = Agent(self, x, y)
                self.grid.place_agent(agent, (x, y))

        for agent in self.grid.agents:
            agent.neighbors = self.grid.get_neighbors((agent.x, agent.y), moore=True, include_center=False)
            


    def step(self):
        self.data_collector.collect(self)
        # agent = np.random.choice(self.agents)
        
        for _ in range(self.N):
            agent = self.random.choice(self.grid.agents)
            agent.update_opinion()

