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
    return (np.mean([a.expressed for a in model.all_agents]) + 1) / 2

def cP(model):
    return (np.mean([a.private for a in model.all_agents]) + 1) / 2

class Agent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.private = np.random.choice([-1, 1])
        self.expressed = np.random.choice([-1, 1])
        # self.private = 1
        # self.expressed = 1


    # def update_private(self, q, p):
    #     if np.random.rand() < p:
    #         if np.random.rand() < 0.5:
    #             self.private *= (-1)
    #     else:
    #         neighbors = [self.model.graph.neighbors(self.unique_id-1)]
    #         if len(neighbors) > q:
    #             q_panel = np.random.choice(neighbors, size=q, replace=False)
    #             if np.sum([a.expressed for a in q_panel]) == q * self.expressed:
    #                 self.private = self.expressed

    # def update_expressed(self, q, p):
    #     if np.random.rand() < p:
    #         self.expressed = self.private
    #     else:
    #         neighbors = [self.model.graph.neighbors(self.unique_id-1)]
    #         if len(neighbors) > q:
    #             q_panel = np.random.choice(neighbors, size=q, replace=False)
    #             if self.private == self.expressed:
    #                 if np.abs(np.sum([a.expressed for a in q_panel])) == q:
    #                     self.expressed = q_panel[0].expressed
    #             else:
    #                 if np.sum([a.expressed for a in q_panel]) != (-1) * q * self.private:
    #                     self.expressed = self.private

    def update_expressed(self, q, p):
        if np.random.rand() < p:
            self.update_expressed_independence()
        else:
            q_panel_ = self.choose_q_panel(q)
            self.update_expressed_conformity(q_panel_)

    def update_private(self, q, p):
        if np.random.rand() < p:
            self.update_private_independence()
        else:
            q_panel_ = self.choose_q_panel(q)
            self.update_private_conformity(q_panel_)

    def update_private_independence(self):
        if np.random.rand() < 0.5:
            self.private *= (-1)

    def update_expressed_independence(self):
        self.expressed = self.private

    def update_private_conformity(self, q_panel_):
        q_panel = [self.model.graph.nodes[n]["agent"] for n in q_panel_]
        if np.sum([a.expressed for a in q_panel]) == len(q_panel) * self.expressed:
            self.private = self.expressed

    def update_expressed_conformity(self, q_panel_):
        # print (f'previous expressed: {self.expressed}')
        q = len(q_panel_)
        q_panel = [self.model.graph.nodes[n]["agent"] for n in q_panel_]
        if self.private == self.expressed:
            if np.abs(np.sum([a.expressed for a in q_panel])) == q:
                self.expressed = q_panel[0].expressed
        else:
            if np.sum([a.expressed for a in q_panel]) != (-1) * q * self.private:
                self.expressed = self.private       
        # print (f'new expressed: {self.expressed}') 

    def get_private(self):
        return self.private
        
    def get_expressed(self):
        return self.expressed

    def get_neighbors(self):
        return list(self.model.graph.neighbors(self.unique_id-1))

    def choose_q_panel(self, q):
        neighbors = self.get_neighbors()
        if len(neighbors) > q:
            return np.random.choice(neighbors, size=q, replace=False)
        else:
            return neighbors


class QVoterEPO(Model):
    def __init__(self, N, q, p, k, beta, width=20, height=20):
        super().__init__()
        self.N = N
        self.q = q
        self.p = p
        self.graph = nx.watts_strogatz_graph(N, k=k, p=beta)
        self.all_agents = []
        for node in self.graph.nodes:
            agent = Agent(self)
            self.all_agents.append(agent)
            self.graph.nodes[node]["agent"] = agent
        self.datacollector = DataCollector(
            model_reporters={"cE": cE, "cP": cP},
            agent_reporters={"expressed": "expressed", "private": "private"}
        )

    def step(self):
        self.datacollector.collect(self)
        for agent in self.all_agents:
            agent.update_expressed(self.q, self.p)
            agent.update_private(self.q, self.p)

    def run_model(self, steps):
        for _ in range(steps):
            self.step()

    # def update_single_agent(self, agent_id):
    #     agent = self.graph.nodes[agent_id]["agent"]
    #     agent.update_expressed(self.q, self.p)
    #     agent.update_private(self.q, self.p)
    #     self.datacollector.collect(self)


# m = QVoterEPO(N=100, q=5, p=0.1, k=4, beta=0.1)
# a = m.graph.nodes[0]["agent"]
# print(f"Agent 0 - Private: {a.get_private()}, Expressed: {a.get_expressed()}")
# a.update_private_conformity(a.choose_q_panel(5))
# print(f"Agent 0 after independence - Private: {a.get_private()}, Expressed: {a.get_expressed()}")