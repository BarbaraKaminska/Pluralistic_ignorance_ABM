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

steps = solara.reactive(20)
N1 = solara.reactive(10)
N2 = solara.reactive(10)
q11 = solara.reactive(0.5)
q12 = solara.reactive(0.5)
q22 = solara.reactive(0.5)
c = solara.reactive(0.1)
Q_learning = solara.reactive(False)  # Whether to use Q-learning
alpha = solara.reactive(0.1) 
beta = solara.reactive(0.1)
gamma = solara.reactive((N1.value - 1)*q11.value / (N2.value*q12.value))
delta = solara.reactive((N2.value - 1)*q22.value / (N1.value*q12.value))
model = solara.reactive(None)
history = solara.reactive(([], [], [], []))  # Store history of E1, E2, S1, S2
step_message = solara.reactive("")
neighbors_ = solara.reactive(None)  # Store neighbors of the selected agent


current_step = solara.reactive(0)
is_running = solara.reactive(False)
target = solara.reactive(None)
agent_id = solara.reactive(None)  # For selecting an agent
rerendere = solara.reactive(True)  # To trigger re-rendering



@solara.component
def Page():

    solara.Title("Dynamics of opinion expression")
    solara.Markdown("## Dynamics of opinion expression")

    with solara.ColumnsResponsive(2, small = [1, 1]):  
        with solara.Card():
            solara.Markdown("## Model Parameters")
            solara.SliderInt("Agents with opinion = 1 (N1)", value=N1, min=10, max=1000, disabled=is_running.value) 
            solara.SliderInt("Agents with opinion = 2 (N2)", value=N2, min=10, max=1000, disabled=is_running.value)
            solara.SliderFloat("Cost of expression (c)", value=c, min=-1.0, max=0.99, step=0.01, disabled=is_running.value)
            solara.SliderInt("Steps", value=steps, min=1, max=500, disabled=is_running.value) 
            solara.SliderFloat("alpha", value=alpha, min=0.0, max=1.0, step=0.01, disabled=is_running.value) 
            solara.SliderFloat("beta", value=beta, min=0.0, max=1.0, step=0.01, disabled=is_running.value)
            solara.Select("Q-learning ", value=Q_learning, values=[True, False], disabled=is_running.value)
            solara.Markdown(f"### N1 = {N1.value},\t N2 = {N2.value}, \t cost = {c.value}, \t steps = {steps.value}")
            solara.Markdown(f"### alpha = {alpha.value}, beta = {beta.value}")

        with solara.Card():
            solara.Markdown("## Network Parameters - Stochastic Block Model")
            solara.SliderFloat("q11", value=q11, min=0.0, max=1.0, step=0.01, disabled=is_running.value, on_value=lambda v: gamma.set((N1.value - 1) * v / (N2.value * q12.value)))
            solara.SliderFloat("q12", value=q12, min=0.0, max=1.0, step=0.01, disabled=is_running.value, on_value=lambda v: update_values(((N2.value - 1) * q22.value / (N1.value * v)), ((N1.value - 1) * q11.value / (N2.value * v))))
            solara.SliderFloat("q22", value=q22, min=0.0, max=1.0, step=0.01, disabled=is_running.value, on_value=lambda v: delta.set((N2.value - 1) * v / (N1.value * q12.value)))
            solara.Markdown(f"q11 = {q11.value}, q12 = {q12.value}, q22 = {q22.value}")
            solara.Markdown(f"delta = {delta.value:.2f}, gamma = {gamma.value:.2f}, c/(1-c) = {c.value/(1-c.value):.2f}, (c+1)/(1-c) = {(c.value + 1)/(1 - c.value):.2f}")
            solara.Markdown(f"d+1 = {delta.value + 1:.2f}, d-1 = {delta.value - 1:.2f}, g+1 = {gamma.value + 1:.2f}, g-1 = {gamma.value - 1:.2f}")
            solara.Markdown(f"(g-1)/(g+1) - c = {(gamma.value - 1)/(gamma.value + 1) - c.value:.2f}, (d-1)/(d+1) - c = {(delta.value - 1)/(delta.value + 1) - c.value:.2f}")
            solara.Markdown(f"g/(g+1) - c = {gamma.value/(gamma.value + 1) - c.value:.2f}, 1/(d+1) + c = {1/(delta.value + 1) + c.value:.2f}")
            solara.Markdown(f"1/(g+1) + c = {1/(gamma.value + 1) + c.value:.2f}, d/(d+1) - c = {delta.value /(delta.value + 1) - c.value:.2f}")

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
            solara.Button("Update", on_click=update_opinion, disabled=(target.value is None) or is_running.value)
            solara.Button("Q-learning", on_click=update_opinion_q_learning, disabled=is_running.value or target.value is None or Q_learning.value is False)
        solara.Text(f"{step_message.value}")
    
    with solara.Row():  # Side-by-side layout
        with solara.Card(title="Time evolution of opinions"):
            plt.figure(figsize=(6, 6))
            if history.value[0]:  # Check if history has data
                E1_list, E2_list, S1_list, S2_list = history.value
                plt.plot(E1_list, label="Expressed 1")
                plt.plot(E2_list, label="Expressed 2")
                # plt.plot(S1_list, label="Silenced 1")
                # plt.plot(S2_list, label="Silenced 2")
                plt.legend()
            else:
                plt.text(steps.value/2, 0.5, "Run the model \nto see results", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.xlabel("Step")
            plt.ylabel("Fraction of agents who express opinion")
            plt.xlim(0, steps.value)
            plt.ylim(-0.05, 1.05)
            plt.title(f"Dynamics of opinion expression")
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
                    if agent.opinion == 1:
                        color_map_face.append("lime")
                        if agent.behavior == 1:
                            color_map_edge.append("green")
                        else:
                            color_map_edge.append("gray")
                    else:
                        color_map_face.append("tomato")   
                        if agent.behavior == 1:
                            color_map_edge.append("red")    
                        else:
                            color_map_edge.append("gray")        
                nodes_sizes = [70 for node in graph.nodes]  # Scale node sizes by degree 
                if target.value is not None:
                    nodes_sizes[target.value] = 350
                if neighbors_.value is not None:
                    for neighbor in neighbors_.value:
                        nodes_sizes[neighbor] = 200
                # pos = nx.spring_layout(graph, seed=42)
                pos = nx.kamada_kawai_layout(graph)
                nx.draw(graph, pos, node_size = nodes_sizes, node_color=color_map_face, edgecolors=color_map_edge, linewidths=3, with_labels=False)
                
            
            else:
                # draw_only_ntw_structure(nx.watts_strogatz_graph(N.value, k=k.value, p=beta.value))
                plt.figure(figsize=(6, 6))
                plt.text(0., 0., "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.title(f"Network Visualization")
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
    step_message.value = ""
    m = OpinionExpression(N1=N1.value, N2=N2.value, q11=q11.value, q12=q12.value, q22=q22.value, c=c.value, alpha=alpha.value, beta=beta.value)
    model.value = m
    model_data = m.datacollector.get_model_vars_dataframe()
    history.value = ([], [], [], [])
    current_step.value = 0

@task
def run_model_app():
    step_message.value = ""
    current_step.value = 0
    is_running.value = True
    rerendere.value = False  # Disable re-rendering during the run
    target.value = None  # Reset target
    m = OpinionExpression(N1=N1.value, N2=N2.value, q11=q11.value, q12=q12.value, q22=q22.value, c=c.value, alpha=alpha.value, beta=beta.value)
    model.value = m
    # m.run_model(steps.value)
    if Q_learning.value:
        for _ in range(steps.value):
            if not is_running.value:
                break
            m.step_q_learning()
            current_step.value += 1
    else:
        for _ in range(steps.value):
            if not is_running.value:
                break
            m.step()
            current_step.value += 1
    model_data = m.datacollector.get_model_vars_dataframe()
    # print(model_data)
    history.value = (model_data["E1"].tolist(), model_data["E2"].tolist(), model_data["S1"].tolist(), model_data["S2"].tolist())
    is_running.value = False
    rerendere.value = True  # Re-enable re-rendering after the run
    print(history.value)

@task
def run_animated_model():
    target.value = None  # Reset target
    step_message.value = ""
    if model.value is None:
        m = OpinionExpression(N1=N1.value, N2=N2.value, q11=q11.value, q12=q12.value, q22=q22.value, c=c.value, alpha=alpha.value, beta=beta.value)
        model.value = m
    else:
        m = model.value
    history.value = ([], [], [], [])
    current_step.value = 0
    is_running.value = True
    if Q_learning.value:
        for _ in range(steps.value):
            if not is_running.value:
                break
            m.step_q_learning()
            df = m.datacollector.get_model_vars_dataframe()
            # print(df)
            history.value = (df["E1"].tolist(), df["E2"].tolist(), df["S1"].tolist(), df["S2"].tolist())
            current_step.value += 1
    else:
        for _ in range(steps.value):
            if not is_running.value:
                break
            m.step()
            df = m.datacollector.get_model_vars_dataframe()
            # print(df)
            history.value = (df["E1"].tolist(), df["E2"].tolist(), df["S1"].tolist(), df["S2"].tolist())
            current_step.value += 1

    is_running.value = False



def select_target():
    target.value = np.random.randint(0, N1.value+N2.value)
    neighbors_.value = list(model.value.graph.neighbors(target.value)) if model.value is not None else None
    rerendere.value = True  # Trigger re-rendering
    agent = model.value.graph.nodes[target.value]["agent"]
    agent_id.value = agent.unique_id
    step_message.value = f"Selected agent holds opinion {agent.opinion} and {'expresses it' if agent.behavior == 1 else 'remains silent'}."


def update_opinion():
    agent = model.value.graph.nodes[target.value]["agent"]
    if Q_learning.value:
        step_message.value, neighbor = agent.update_behavior_q_learning_message()
    else:
        step_message.value = agent.update_behavior_message()
    prev_target = target.value
    target.value = None  # Reset target
    target.value = prev_target  # Set it back to the previous target


def update_opinion_q_learning():
    agent = model.value.graph.nodes[target.value]["agent"]
    if agent.behavior == 1:
        message_, neighbor = agent.update_behavior_q_learning_message()
        step_message.value = agent.q_learning_message(neighbor)
    else:
        step_message.value = "Agent remains silent, no Q-learning update."
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


def update_values(x, y):
    delta.set(x)
    gamma.set(y)