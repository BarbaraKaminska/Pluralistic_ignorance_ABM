# app.py


# to do 
# V ensure that k <= N-1 (both if user changes k or N)
# - random sequential update
# V implementation of the model so that it works on the network
# V plot and ntwork should evolve in time
# - step by step algorithm exlplanation
# V prevent user from changing parameters while the model is running
# - possibly, add different initial conditions for agents' opinions
# - stop button -> stop the model 

import solara
import numpy as np
import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.agent import Agent
import networkx as nx
from model import *
import time
import asyncio
from solara.lab import task

# === Solara App ===

steps = solara.reactive(50)
N = solara.reactive(10)
q = solara.reactive(3)
p = solara.reactive(0.2)
k = solara.reactive(5)
beta = solara.reactive(0.1)
model = solara.reactive(None)
history = solara.reactive(([], [])) 


current_step = solara.reactive(0)
is_running = solara.reactive(False)
opinion_to_update = solara.reactive("Private Opinion") 
response = solara.reactive("Conformity")
target = solara.reactive(None)
agent_id = solara.reactive(None)  # For selecting an agent
q_panel = solara.reactive(None)
rerendere = solara.reactive(True)  # To trigger re-rendering


@solara.component
def Page():

    solara.Title("q-voter EPO model")
    solara.Markdown("## Interactive q-voter simulation")

    with solara.ColumnsResponsive(2, small = [1, 1]):  
        with solara.Card():
            solara.Markdown("### Model Parameters")
            solara.SliderInt("Agents (N)", value=N, min=10, max=1000, disabled=is_running.value,
            on_value=lambda _ : N.set(max((k.value+1), N.value)))  # Ensure k <= N-1
            solara.SliderInt("Group size (q)", value=q, min=1, max=10, disabled=is_running.value)
            solara.SliderFloat("Probability (p)", value=p, min=0.0, max=1.0, step=0.01, disabled=is_running.value)
            solara.SliderInt("Steps", value=steps, min=1, max=2000, disabled=is_running.value)
        with solara.Card():
            solara.Markdown("### Network Parameters (Watts-Strogatz)")
            solara.SliderInt("Average degree (k)", value=k, min=5, max=100, disabled=is_running.value, 
            on_value=lambda _ : k.set(min(k.value, N.value - 1)))  # Ensure k <= N-1
            solara.SliderFloat("Rewiring probability (beta)", value=beta, min=0.0, max=1.0, step=0.01, disabled=is_running.value)

    with solara.Card():
        solara.Markdown("### Model Control")
        with solara.Row():
            solara.Button("Setup Model", on_click=setup_model, disabled=is_running.value)
            solara.Button("Run Model", on_click=run_animated_model, disabled=is_running.value or model.value is None)
            solara.Button("Run Model (no animation)", on_click=run_model_app, disabled=is_running.value or model.value is None)
            solara.Button("Stop", on_click=stop_model, disabled=not is_running.value)
            solara.Text(f"Step: {current_step.value} / {steps.value}")

    with solara.Card():
        solara.Markdown("### Algorithm - Step-by-Step")
        with solara.Row():
            solara.Button("Target", on_click= select_target, disabled=is_running.value or (model.value is None))
            solara.Select("Opinion to be updated", value=opinion_to_update, values=["Private Opinion", "Expressed Opinion"], disabled=is_running.value)
            solara.Select("Response", value=response, values=["Conformity", "Independence"], disabled=is_running.value)
            solara.Button("q-panel", on_click=select_q_panel, disabled=response.value == "Independence" or is_running.value or target.value is None)
            solara.Button("Update", on_click=update_opinion, disabled=(target.value is None) or (q_panel.value is None and response.value =='Conformity') or is_running.value)

    
    with solara.Row():  # Side-by-side layout
        with solara.Card(title="Time evolution of opinions"):
            plt.figure(figsize=(6, 6))
            if history.value[0]:  # Check if history has data
                cE_list, cP_list = history.value
                plt.plot(cE_list, label="Expressed")
                plt.plot(cP_list, label="Private")
                plt.legend()
            else:
                plt.text(steps.value/2, 0.5, "Run the model \nto see results", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.xlabel("Step")
            plt.ylabel("Fraction of agents with positive opinion")
            plt.xlim(0, steps.value)
            plt.ylim(-0.05, 1.05)
            plt.title(f"q-voter EPO model - N: {N.value}, q: {q.value}, p: {p.value}")
            plt.show()
        

        with solara.Card(title="Network Visualization"):
            # solara.Markdown("### Network Visualization")
            if model.value is not None and rerendere.value:    
                plt.figure(figsize=(6, 6))
                graph = model.value.graph
                color_map_edge = []
                color_map_face = []
                for node in graph.nodes:
                    agent = graph.nodes[node]["agent"]
                    if agent.expressed == 1:
                        color_map_edge.append("green")
                    else:
                        color_map_edge.append("red")
                    if agent.private == 1:
                        color_map_face.append("lime")
                    else:
                        color_map_face.append("tomato")           
                nodes_sizes = [70 for node in graph.nodes]  # Scale node sizes by degree 
                if target.value is not None:
                    nodes_sizes[target.value] = 350
                if q_panel.value is not None:
                    for node in q_panel.value:
                        nodes_sizes[node] = 250
                pos = nx.spring_layout(graph, seed=42)
                nx.draw(graph, pos, node_size = nodes_sizes, node_color=color_map_face, edgecolors=color_map_edge, linewidths=3, with_labels=False)
                plt.title(f"<k> = {k.value}, " +r"$\beta$" +f" = {beta.value}")
            
            else:
                # draw_only_ntw_structure(nx.watts_strogatz_graph(N.value, k=k.value, p=beta.value))
                plt.figure(figsize=(6, 6))
                plt.text(0., 0., "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.title(f"<k> = {k.value}, " +r"$\beta$" +f" = {beta.value}")
            plt.xlabel("")
            plt.ylabel("")
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            # plt.axis('off')  # Hide the axes

            plt.show()
        

def setup_model():
    """Initialize the model with the current parameters."""
    model.value = None  # Reset the model
    target.value = None  # Reset target
    q_panel.value = None  # Reset q_panel
    m = QVoterEPO(N=N.value, q=q.value, p=p.value, k=k.value, beta=beta.value)
    model.value = m
    model_data = m.datacollector.get_model_vars_dataframe()
    history.value = ([], [])
    current_step.value = 0

@task
def run_model_app():
    current_step.value = 0
    is_running.value = True
    rerendere.value = False  # Disable re-rendering during the run
    target.value = None  # Reset target
    q_panel.value = None  # Reset q_panel
    m = QVoterEPO(N=N.value, q=q.value, p=p.value, k=k.value, beta=beta.value)
    model.value = m
    # m.run_model(steps.value)
    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        current_step.value += 1
    model_data = m.datacollector.get_model_vars_dataframe()
    history.value = (model_data["cE"].tolist(), model_data["cP"].tolist())
    is_running.value = False
    rerendere.value = True  # Re-enable re-rendering after the run

@task
def run_animated_model():
    target.value = None  # Reset target
    q_panel.value = None  # Reset q_panel
    if model.value is None:
        m = QVoterEPO(N=N.value, q=q.value, p=p.value, k=k.value, beta=beta.value)
        model.value = m
    else:
        m = model.value
    print(f"model.value: {model.value}")
    history.value = ([], [])
    current_step.value = 0
    is_running.value = True

    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        df = m.datacollector.get_model_vars_dataframe()
        history.value = (df["cE"].tolist(), df["cP"].tolist())
        current_step.value += 1
        # time.sleep(0.3)  # Delay between frames to animate

    is_running.value = False



def draw_network(graph):
    color_map_edge = []
    color_map_face = []
    for node in graph.nodes:
        agent = graph.nodes[node]["agent"]
        if agent.expressed == 1:
            color_map_edge.append("green")
        else:
            color_map_edge.append("red")
        if agent.private == 1:
            color_map_face.append("lightgreen")
        else:
            color_map_face.append("lightcoral")
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(graph, pos, node_color=color_map_face, edgecolors = color_map_edge, linewidths=3, with_labels=False, node_size=70)

def draw_only_ntw_structure(graph):
    pos = nx.spring_layout(graph, seed=42)
    plot.figure(figsize=(6, 6))
    nx.draw(graph, pos, with_labels=False, node_size=50)



def select_target():
    target.value = np.random.randint(0, N.value)
    q_panel.value = None  # Reset q_panel
    agent = model.value.graph.nodes[target.value]["agent"]
    agent_id.value = agent.unique_id

def select_q_panel():
    if model.value is not None:
        agent = model.value.graph.nodes[target.value]["agent"]
        q_panel.value = agent.choose_q_panel(q.value)


def update_opinion():
    agent = model.value.graph.nodes[target.value]["agent"]
    solara.Info(f"Updating {opinion_to_update.value} for agent {target.value} with response {response.value} and q-panel {q_panel.value}")
    if opinion_to_update.value == "Private Opinion":
        if response.value == "Conformity":
            agent.update_private_conformity(q_panel.value)
        else:
            agent.update_private_independence()
    else:  # Expressed Opinion
        if response.value == "Conformity":
            agent.update_expressed_conformity(q_panel.value)   
        else:
            agent.update_expressed_independence()
    prev_target = target.value
    target.value = None  # Reset target
    target.value = prev_target  # Set it back to the previous target


def stop_model():
    if run_animated_model.pending:
        run_animated_model.cancel()
    if run_model_app.pending:
        run_model_app.cancel()
    is_running.value = False
    solara.Info("Model stopped.")