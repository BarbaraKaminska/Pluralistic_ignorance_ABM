# To do 
# 1. Moore vs von Neumann neighborhood
# 2. Add enforcement and compliance to the network visualization
# 3. Legend to network visualization
# 4. Markdown: real fraction of believers
# 5. step by step algorithm



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
current_step = solara.reactive(0)
is_running = solara.reactive(False)
N = solara.reactive(25)
N_believers = solara.reactive(10)  # Percentage of believers
real_fraction_believers = solara.reactive(N_believers.value)  # Placeholder for real fraction of believers
k = solara.reactive(0.125)
beta = solara.reactive(0.5)
target = solara.reactive(None)
rerendere = solara.reactive(True)  # To trigger re-rendering
network = solara.reactive("square lattice")
neighborhood = solara.reactive("Moore")
model = solara.reactive(None)
history = solara.reactive(([], [])) 
step_message = solara.reactive("")

networks = ["square lattice", "Watts-Strogatz"]
neighborhoods = ["Moore", "von Neumann"]
substeps = ['compliance', 'need for enforcement', 'enforcement']
current_substep = solara.reactive(0)

@solara.component
def Page():

    solara.Title("Emperor's dilemma model")
    solara.Markdown("### Emperor's dilemma model")    
    with solara.ColumnsResponsive(2, small = [1, 1]):  
        with solara.Card():
            solara.Markdown("### Model Parameters")
            solara.SliderInt("Agents (N)", value=N, min=10, max=1000, disabled=is_running.value,
            on_value=lambda _ : N.set(max((k.value+1), N.value)))  # Ensure k <= N-1
            solara.SliderInt("Percentage of believers", value=N_believers, min=0, max=100, disabled=is_running.value)
            solara.SliderInt("Steps", value=steps, min=1, max=2000, disabled=is_running.value)
            solara.Markdown(f"### Selected N = {N.value}; \t Steps = {steps.value}; \t Percentage of believers = {real_fraction_believers} %")

        with solara.Card():
            solara.Markdown("### Network Parameters")
            solara.Select(label="Network structure", value=network, values=networks, disabled=is_running.value, on_value=setup_model)  # Trigger re-rendering on change
            solara.SliderInt("Average degree (k)", value=k, min=5, max=100, disabled=is_running.value or network.value != "Watts-Strogatz", on_value=lambda _ : k.set(min(k.value, N.value - 1)))  # Ensure k <= N-1
            solara.SliderFloat(f"Rewiring probability (beta))", value=beta, min=0.0, max=1.0, step=0.01, disabled=is_running.value or network.value != "Watts-Strogatz")
            solara.Select(label="Type of neighborhood", value=neighborhood, values=neighborhoods, disabled=is_running.value or network.value != "square lattice", on_value=setup_model) 

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
            solara.Button("Next step", on_click=lambda: next_single_step(current_substep.value), disabled=is_running.value or (model.value is None))
            solara.Markdown(f"{step_message.value}")


    with solara.Row():  # Side-by-side layout
        with solara.Card(title="Time evolution of opinions"):
            plt.figure(figsize=(6, 6))
            if history.value[0]:  # Check if history has data
                cE_list, cC_list = history.value
                cE_arr = np.array(cE_list)

                print(f"cE_list: {cE_arr.shape}")
                plt.plot(cC_list, label="Compliance with the norm")
                plt.plot(cE_arr[:, 0], label="True enforcement")
                plt.plot(cE_arr[:, 1], label="No enforcement")
                plt.plot(cE_arr[:, 2], label="False enforcement")
                plt.legend()
            else:
                plt.text(steps.value/2, 0.5, "Run the model \nto see results", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.xlabel("Step")
            plt.ylabel("Fraction of agents", fontsize=16)
            plt.xlim(0, steps.value)
            plt.ylim(-0.05, 1.05)
            plt.title(f"Emperor's dilemma model", fontsize=16)
            plt.show()
     

        with solara.Card(title="Network Visualization"):
            # solara.Markdown("### Network Visualization")
            if model.value is not None and rerendere.value:    
                plt.figure(figsize=(6, 6))
                graph = model.value.graph
                color_map_edge = []
                color_map_face = []
                nodes_shapes = []
                for node in graph.nodes:
                    agent = graph.nodes[node]["agent"]
                    if agent.E == 1:
                        color_map_edge.append("green")
                    elif agent.E == -1:
                        color_map_edge.append("red")
                    else:
                        color_map_edge.append("gray")
                    if agent.C == 1:
                        color_map_face.append("lime")
                    else:
                        color_map_face.append("tomato")    
                    if agent.B == 1:
                        nodes_shapes.append("o")   
                        graph.nodes[node]['shape'] = ["o"]
 
                    else:
                        nodes_shapes.append("d")
                        graph.nodes[node]['shape'] = ["d"]

                # for node in graph.nodes:
                #     print(f"Node {node}: shape = {graph.nodes[node]['shape']}, agent = {graph.nodes[node]['agent']}")
                # print(graph.nodes)           

                nodes_sizes = [100 for node in graph.nodes]  # Scale node sizes by degree 
                if target.value is not None:
                    nodes_sizes[target.value] = 400

                    all_nodes = list(graph.nodes)
                    node = all_nodes[target.value]  # safe, no KeyError here


                    for neighbor in graph.nodes[node]["agent"].neighbors:
                        print(f"Neighbor: {neighbor.unique_id}")
                        nodes_sizes[neighbor.unique_id-1] = 200

                if network.value == "square lattice":
                    pos = dict((n, n) for n in graph.nodes())
                    plt.xlim(-1, np.sqrt(graph.number_of_nodes()))
                    plt.ylim(-1, np.sqrt(graph.number_of_nodes()))
                else:
                    pos = nx.spring_layout(graph, seed=42)
                    plt.xlim(-1.1, 1.1)
                    plt.ylim(-1.1, 1.1)
                # nx.draw(graph, pos, node_size = nodes_sizes, node_color=color_map_face, edgecolors=color_map_edge, node_shape= nodes_shapes, linewidths=3, with_labels=False)
                
                nx.draw_networkx_edges(graph, pos, edge_color='k', width=1)
                for shape in [['d'], ['o']]:
                    nodes_with_shape = [n for n in graph.nodes if graph.nodes[n]['shape'] == shape]
                    nodes_sizes_temp = [nodes_sizes[i] for i, n in enumerate(graph.nodes) if graph.nodes[n]['shape'] == shape]
                    color_map_face_temp = [color_map_face[i] for i, n in enumerate(graph.nodes) if graph.nodes[n]['shape'] == shape]
                    color_map_edge_temp = [color_map_edge[i] for i, n in enumerate(graph.nodes) if graph.nodes[n]['shape'] == shape]
                    nx.draw_networkx_nodes(
                        graph,
                        pos,
                        nodelist=nodes_with_shape,
                        node_shape=shape[0],
                        node_color=color_map_face_temp,
                        edgecolors=color_map_edge_temp,
                        node_size=nodes_sizes_temp,
                        linewidths=3
                    )
                
                # nx.draw_networkx_nodes(graph, pos, node_size=nodes_sizes, node_color=color_map_face, edgecolors=color_map_edge, linewidths=3)
                # plt.title(f"<k> = {k.value}, " +r"$\beta$" +f" = {beta.value}")
            
            else:
                # draw_only_ntw_structure(nx.watts_strogatz_graph(N.value, k=k.value, p=beta.value))
                plt.figure(figsize=(6, 6))
                if network.value == "square lattice":
                    text_pos_x = np.sqrt(N.value) / 2
                    text_pos_y = np.sqrt(N.value) / 2
                    plt.xlim(0, np.sqrt(N.value))
                    plt.ylim(0, np.sqrt(N.value))
                else:
                    text_pos_x = 0
                    text_pos_y = 0
                    plt.xlim(-1.1, 1.1)
                    plt.ylim(-1.1, 1.1)
                plt.text(text_pos_x, text_pos_y, "Setup the model \nto see network", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.title(f"{int(np.sqrt(N.value))} x {int(np.sqrt(N.value))} grid" if network.value == "square lattice" else f"N = {N.value}, k = {k.value}, beta = {beta.value}", fontsize=16)
            plt.xlabel("")
            plt.ylabel("")
            # plt.axis('off')  # Hide the axes

            plt.show()


def setup_model(*args):
    """Initialize the model with the current parameters."""
    step_message.value = ""
    model.value = None  # Reset the model
    target.value = None  # Reset target
    m = EmperosDilemma(N=N.value, K=steps.value, N_bel = N_believers.value, network_type=network.value, k=k.value, beta=beta.value)
    model.value = m
    model_data = m.datacollector.get_model_vars_dataframe()
    # history.value = (model_data["cE"].tolist(), model_data["cC"].tolist(), model_data["cB"].tolist())
    real_fraction_believers.value = int((np.mean([a.B for a in m.all_agents]) + 1) / 2 * 100)
    print(f"Real fraction of believers: {real_fraction_believers}")
    history.value = ([], [])
    current_step.value = 0
    current_substep.value = 0

@task
def run_model_app():
    step_message.value = ""
    current_step.value = 0
    is_running.value = True
    rerendere.value = False  # Disable re-rendering during the run
    target.value = None  # Reset target
    m = EmperosDilemma(N=N.value, K=steps.value, N_bel = N_believers.value, network_type=network.value, k=k.value, beta=beta.value)
    model.value = m
    print(f"Running model with N={N.value}, K={steps.value}, network_type={network.value}, k={k.value}, beta={beta.value}")
    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        current_step.value += 1
    model_data = m.datacollector.get_model_vars_dataframe()
    history.value = (model_data["cE"].tolist(), model_data["cC"].tolist())
    is_running.value = False
    rerendere.value = True  # Re-enable re-rendering after the run
    current_substep.value = 0

@task
def run_animated_model():
    step_message.value = ""
    target.value = None  # Reset target
    if model.value is None:
        m = EmperosDilemma(N=N.value, K=steps.value, network_type=network.value, k=k.value, beta=beta.value)
        model.value = m
    else:
        m = model.value
    history.value = ([], [])
    current_step.value = 0
    is_running.value = True

    for _ in range(steps.value):
        if not is_running.value:
            break
        m.step()
        df = m.datacollector.get_model_vars_dataframe()
        history.value = (df["cE"].tolist(), df["cC"].tolist())
        current_step.value += 1
        # time.sleep(0.3)  # Delay between frames to animate

    is_running.value = False
    current_substep.value = 0


def stop_model():
    if run_animated_model.pending:
        run_animated_model.cancel()
    if run_model_app.pending:
        run_model_app.cancel()
    is_running.value = False
    solara.Info("Model stopped.")
    current_substep.value = 0
    step_message.value = ""

def select_target():
    if model.value is not None:
        all_nodes = list(model.value.graph.nodes)
        target.value = np.random.choice(len(all_nodes))
        current_substep.value = 0
        step_message.value = f"Selected target agent: {target.value}."


def next_single_step(current_substep_value):
    current_substep.value = current_substep_value+1
    if model.value is not None and target.value is not None:        
        graph = model.value.graph
        all_nodes = list(graph.nodes)
        node_key = all_nodes[target.value]  # Get the actual node key
        target_agent = graph.nodes[node_key]["agent"]
        
        if current_substep.value == 1:
            step_message.value = target_agent.if_comply_message()
            target_agent.if_comply()
        elif current_substep.value == 2:
            step_message.value = target_agent.update_need_for_enforcement_message()
            target_agent.update_need_for_enforcement()
        elif current_substep.value == 3:
            step_message.value = target_agent.if_enforce_message()
            target_agent.if_enforce()
        else:
            step_message.value = "Update completed. Click 'Target' to select a new agent." 