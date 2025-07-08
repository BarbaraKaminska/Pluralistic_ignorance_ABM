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
N = solara.reactive(10)
K = solara.reactive(0.125)
target = solara.reactive(None)
rerendere = solara.reactive(True)  # To trigger re-rendering
network = solara.reactive("square lattice")

networks = ["square lattice", "Watts-Strogatz"]


@solara.component
def Page():

    solara.Title("Emperor's dilemma model")
    solara.Markdown("Emperor's dilemma model")    
    with solara.ColumnsResponsive(2, small = [1, 1]):  
        with solara.Card():
            solara.Markdown("### Model Parameters")
            solara.SliderInt("Agents (N)", value=N, min=10, max=1000, disabled=is_running.value,
            on_value=lambda _ : N.set(max((k.value+1), N.value)))  # Ensure k <= N-1
            solara.SliderInt("Steps", value=steps, min=1, max=2000, disabled=is_running.value)

        with solara.Card():
            solara.Markdown("### Network Parameters")
            solara.Select(label="Network structure", value=network, values=networks)
            solara.SliderInt("Average degree (k)", value=k, min=5, max=100, disabled=is_running.value or network.value =, 
            on_value=lambda _ : k.set(min(k.value, N.value - 1)))  # Ensure k <= N-1
            solara.SliderFloat(f"Rewiring probability (beta))", value=beta, min=0.0, max=1.0, step=0.01, disabled=is_running.value)
            solara.Markdown(f"##k = {k.value},\t" + r"$\beta$" + f" = {beta.value}")






