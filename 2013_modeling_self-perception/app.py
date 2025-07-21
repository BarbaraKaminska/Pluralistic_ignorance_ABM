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
L = solara.reactive(10)
N = solara.reactive(L.value**2)
G = solara.reactive(2)  # Normative social influence threshold
U = solara.reactive(0.6)  # Attitude-lacking agent threshold
RED = solara.reactive(0.6)  # Red color threshold
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
opinion_evolution = solara.reactive(np.empty((steps.value, N.value)))  # To store opinion evolution
attitude_evolution = solara.reactive(np.empty((steps.value, N.value)))  # To store attitude evolution



@solara.component
def Page():

    solara.Title("Modeling Self-Perception")
    solara.Markdown("## Modeling Self-Perception")


    with solara.ColumnsResponsive(2, small = [1, 1]):  
        with solara.Card():
            solara.Markdown("### Model Parameters")
            solara.SliderInt("System size (L)", value=L, min=5, max=25, disabled=is_running.value, 
                             on_value=lambda v: N.set(v**2))  
            solara.SliderInt("Normative social influence threshold (G)", value=2, min=1, max=5, disabled=is_running.value)
            solara.SliderFloat("Attitude-lacking agent threshold (U)", value=0.6, min=0.1, max=1, step=0.01, disabled=is_running.value)
            solara.SliderInt("Steps", value=steps, min=1, max=200, disabled=is_running.value)
            solara.Markdown(f"##N = {N},\t steps = {steps.value}")


    with solara.Card():
        solara.Markdown("### Model Control")
        with solara.Row():
            solara.Button("Setup Model", on_click=setup_model, disabled=is_running.value)
            solara.Button("Run Model", on_click=run_animated_model, disabled=is_running.value or model.value is None)
            solara.Button("Run Model (no animation)", on_click=run_model_app, disabled=is_running.value or model.value is None)
            solara.Button("Stop", on_click=stop_model, disabled=not is_running.value)
            solara.Text(f"Step: {current_step.value} / {steps.value}")




    with solara.Row():  # Side-by-side layout
        with solara.Card(title="Time evolution"):
            plt.figure(figsize=(6, 6))
            if history.value[0]:  # Check if history has data
                op_list, att_list, std_op_list, std_att_list = history.value
                plt.plot(op_list, label="Opinion")
                plt.plot(att_list, label="Attitude")
                plt.legend()
            else:
                plt.text(steps.value/2, 0.5, "Run the model \nto see results", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.xlabel("Step")
            plt.ylabel("Fraction of agents with positive opinion")
            plt.xlim(0, steps.value)
            plt.ylim(-0.05, 1.05)
            plt.title(f"Modeling self-perception - N: {N.value}")
            plt.show()
        

        with solara.Card(title="Network Visualization"):
            # solara.Markdown("### Network Visualization")
            if model.value is not None and rerendere.value:    
                data = get_opinion_grid(model.value)

                plt.figure(figsize=(6, 6))
                sns.heatmap(data, cmap="RdBu", vmin=0, vmax=1, cbar_kws={'label': 'Opinion'})
                plt.title("Opinions of agents")
                plt.tight_layout()
                plt.xlim(-0.5, data.shape[1] + 0.5)
                plt.ylim(-0.5, data.shape[0] + 0.5)
            else:
                
                plt.figure(figsize=(6, 6))
                plt.text(0., 0., "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
            plt.xlabel("")
            plt.ylabel("")
            # plt.axis('off')  # Hide the axes

            plt.show()

        with solara.Card(title="Network Visualization"):
            # solara.Markdown("### Network Visualization")
            if model.value is not None and rerendere.value:    
                for i in range(N.value):
                    plt.scatter(np.arange(steps.value), opinion_evolution.value[:, i], s=1, color='red')
                print(f"opinion_evolution: {opinion_evolution.value[0, :]}")
                plt.xlabel("time step")
                plt.ylabel("value")
                plt.ylim(0, 1)
            else:
                
                plt.figure(figsize=(6, 6))
                plt.text(0., 0., "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
            plt.xlabel("")
            plt.ylabel("")
            # plt.axis('off')  # Hide the axes

            plt.show()
        
    with solara.Row():  # Side-by-side layout
        with solara.Card(title=""):
            plt.figure(figsize=(6, 6))
            if history.value[0]:  # Check if history has data                
                op_list, att_list, std_op_list, std_att_list = history.value
                plt.plot(std_op_list, label="Std Opinion")
                plt.plot(std_att_list, label="Std Attitude")
                plt.legend()
            else:
                plt.text(steps.value/2, 0.5, "Run the model \nto see results", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.xlabel("Step")
            plt.ylabel("Fraction of agents with positive opinion")
            plt.xlim(0, steps.value)
            plt.ylim(-0.05, 1.05)
            plt.title(f"Modeling self-perception - N: {N}")
            plt.show()
        

        with solara.Card(title=""):
            # solara.Markdown("### Network Visualization")
            if model.value is not None and rerendere.value:    
                data = get_attitude_grid(model.value)

                plt.figure(figsize=(6, 6))
                sns.heatmap(data, cmap="RdBu", vmin=0, vmax=1, cbar_kws={'label': 'Attitude'})
                plt.title("Attitudes of agents")
                plt.tight_layout()
                plt.xlim(-0.5, data.shape[1] + 0.5)
                plt.ylim(-0.5, data.shape[0] + 0.5)
            else:
                
                plt.figure(figsize=(6, 6))
                plt.text(0., 0., "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
            plt.xlabel("")
            plt.ylabel("")
            # plt.axis('off')  # Hide the axes

            plt.show()


def setup_model():
    """Initialize the model with the current parameters."""
    model.value = None  # Reset the model
    target.value = None  # Reset target
    m = PluralisticIgnoranceModel(N.value, G.value, U.value, RED.value)
    model.value = m
    # model_data = m.datacollector.get_model_vars_dataframe()
    history.value = ([], [], [], [])  # Reset history
    current_step.value = 0


def get_opinion_grid(model):
    width, height = model.grid.width, model.grid.height
    opinion_grid = np.zeros((height, width))

    for x in range(width):
        for y in range(height):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                opinion_grid[height - y - 1][x] = agents[0].op  # flip y to match plot orientation

    return opinion_grid

def get_attitude_grid(model):
    width, height = model.grid.width, model.grid.height
    attitude_grid = np.zeros((height, width))

    for x in range(width):
        for y in range(height):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                attitude_grid[height - y - 1][x] = agents[0].att  # flip y to match plot orientation

    return attitude_grid


@task
def run_model_app():
    current_step.value = 0
    is_running.value = True
    rerendere.value = False  # Disable re-rendering during the run
    target.value = None  # Reset target
    q_panel.value = None  # Reset q_panel
    m = PluralisticIgnoranceModel(N=N.value, G=G.value, U=U.value, RED=RED.value)
    model.value = m
    # m.run_model(steps.value)
    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        current_step.value += 1
    model_data = m.data_collector.get_model_vars_dataframe()
    history.value = (model_data["average_opinion"].tolist(), model_data["average_attitude"].tolist(), 
    model_data["std_opinion"].tolist(), model_data["std_attitude"].tolist())
    is_running.value = False
    rerendere.value = True  # Re-enable re-rendering after the run

@task
def run_animated_model():
    target.value = None  # Reset target
    q_panel.value = None  # Reset q_panel
    if model.value is None:
        m = PluralisticIgnoranceModel(N=N.value, G=G.value, U=U.value, RED=RED.value)
        model.value = m
    else:
        m = model.value
    print(f"model.value: {model.value}")
    history.value = ([], [], [], [])  # Reset history
    opinion_evolution = np.empty((steps.value, N.value))  # Reset opinion evolution
    attitude_evolution = np.empty((steps.value, N.value))  # Reset attitude evolution
    current_step.value = 0
    is_running.value = True

    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        df = m.data_collector.get_model_vars_dataframe()
        history.value = (df["average_opinion"].tolist(), df["average_attitude"].tolist(), 
        df["std_opinion"].tolist(), df["std_attitude"].tolist())
        opinion_evolution[current_step.value, :] = np.array([agent.op for agent in m.agents])
        attitude_evolution[current_step.value, :] = np.array([agent.att for agent in m.agents])
        current_step.value += 1
        # time.sleep(0.3)  # Delay between frames to animate

    is_running.value = False

def stop_model():
    if run_animated_model.pending:
        run_animated_model.cancel()
    if run_model_app.pending:
        run_model_app.cancel()
    is_running.value = False
    solara.Info("Model stopped.")
