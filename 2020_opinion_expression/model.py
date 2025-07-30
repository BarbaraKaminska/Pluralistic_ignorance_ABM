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

def E1(model):
    return np.count_nonzero([a.behavior == 1 for a in model.all_agents if a.opinion == 1]) / model.N1

def E2(model):
    return np.count_nonzero([a.behavior == 1 for a in model.all_agents if a.opinion == 2]) / model.N2

def S1(model):
    return np.count_nonzero([a.behavior == 0 for a in model.all_agents if a.opinion == 1]) / model.N1

def S2(model):
    return np.count_nonzero([a.behavior == 0 for a in model.all_agents if a.opinion == 2]) / model.N2

class Agent(Agent):
    def __init__(self, model, group):
        super().__init__(model)
        self.opinion = group
        self.behavior = np.random.choice([0, 1])
        self.Q = np.random.rand()  # Q-values for actions (express, silence) for each opinion

    def update_behavior(self):
        if self.opinion == 1:
            LHS = self.model.q11*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 1 and agent != self])/((self.model.N1 - 1)*self.model.q11 + self.model.N2*self.model.q12)
            RHS = self.model.q12*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 2])/((self.model.N1 - 1)*self.model.q11 + self.model.N2*self.model.q12) + self.model.c

        else:
            LHS = self.model.q22*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 2 and agent != self])/((self.model.N2 - 1)*self.model.q22 + self.model.N1*self.model.q12)
            RHS = self.model.q12*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 1 ])/((self.model.N2 - 1)*self.model.q22 + self.model.N1*self.model.q12) + self.model.c

        if LHS > RHS:
            self.behavior = 1  
        else:
            self.behavior = 0

    def update_behavior_q_learning(self):
        neighbor = None
        
        prob_express = 1 / (1 + np.exp(-self.Q*self.model.beta))  # Sigmoid function to convert Q-value to probability

        if np.random.rand() < prob_express:
            self.behavior = 1 
            # print(self.model.graph.neighbors(self.unique_id))
            neighbors_list = list(self.model.graph.neighbors(self.unique_id-1))
            neighbor_id = np.random.choice(np.array(neighbors_list), size = 1)
            # print(neighbor_id)
            neighbor = self.model.graph.nodes[neighbor_id[0]]["agent"]
            if neighbor.behavior == 0:
                r = -self.model.c
            else:
                if self.opinion == neighbor.opinion:
                    r = 1 - self.model.c
                else:
                    r = -1 - self.model.c
            self.Q = (1 - self.model.alpha) * self.Q + self.model.alpha * r
        else:
            self.behavior = 0

        return prob_express, neighbor

    def update_behavior_message(self):
        if self.opinion == 1:
            LHS = self.model.q11*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 1 and agent != self])/((self.model.N1 - 1)*self.model.q11 + self.model.N2*self.model.q12)
            RHS = self.model.q12*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 2])/((self.model.N1 - 1)*self.model.q11 + self.model.N2*self.model.q12) + self.model.c
            message = f"Agreement {LHS:.2f} vs disagreement & cost {RHS:.2f} for opinion 1"

        else:
            LHS = self.model.q22*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 2 and agent != self])/((self.model.N2 - 1)*self.model.q22 + self.model.N1*self.model.q12)
            RHS = self.model.q12*np.count_nonzero([self.behavior == 1 for agent in self.model.all_agents if agent.opinion == 1 ])/((self.model.N2 - 1)*self.model.q22 + self.model.N1*self.model.q12) + self.model.c
            message = f"Agreement {LHS:.2f} vs disagreement & cost {RHS:.2f} for opinion 2"

        if LHS > RHS:
            self.behavior = 1  
            message += " - Agent expresses opinion"
        else:
            self.behavior = 0
            message += " - Agent remains silent"
        
        return message

    def update_behavior_q_learning_message(self):
        prob_express, neighbor = self.update_behavior_q_learning()
        message = f"Probability of expressing opinion: {prob_express:.2f}; "
        
        if self.behavior == 1:
            message += "\nagent expresses opinion."
        else:
            message += "\nagent remains silent."

        return message, neighbor

    def q_learning_message(self, neighbor):
        if neighbor.behavior == 0:
            r = -self.model.c       
            message = f"Neighbor is silent; reward = {r:.2f}; "
        else:
            if self.opinion == neighbor.opinion:
                r = 1 - self.model.c
                message = f"Neighbor agrees; reward = {r:.2f}; "
            else:
                r = -1 - self.model.c
                message = f"Neighbor disagrees; reward = {r:.2f}; "
        message += f"\nQ-value before update = {self.Q:.2f}; "
        self.Q = (1 - self.model.alpha) * self.Q + self.model.alpha * r
        message += f"\nQ-value after update = {self.Q:.2f}"

        return message


class OpinionExpression(Model):
    def __init__(self, N1, N2, q11, q12, q22, c, alpha, beta, width=20, height=20):
        super().__init__()
        self.N1 = N1
        self.N2 = N2
        self.N = N1 + N2
        self.q11 = q11
        self.q12 = q12
        self.q22 = q22
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.graph = nx.stochastic_block_model(
            sizes=[N1, N2],
            p=[[q11, q12], [q12, q22]],
            directed=False,
            selfloops=False
        )
        self.all_agents = []
        for idx, node in enumerate(self.graph.nodes):
            if node < N1:
                agent = Agent(self, group=1)
            else:
                agent = Agent(self, group=2)
            self.all_agents.append(agent)
            self.graph.nodes[node]["agent"] = agent
        self.datacollector = DataCollector(
            model_reporters={"E1": E1, "E2": E2, "S1": S1, "S2": S2},
            agent_reporters={"opinion": "opinion", "behavior": "behavior"}
        )
        

    def step(self):
        self.datacollector.collect(self)
        for agent in self.all_agents:
            agent.update_behavior()

    def step_q_learning(self):
        self.datacollector.collect(self)
        for agent in self.all_agents:
            agent.update_behavior_q_learning()