import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pgmpy.models import BayesianNetwork


def get_boundary(model, target_variable):
    """Computes the Markov blanket of a target variable in a Bayesian network model.

    Args:
        model (pgmpy.models.BayesianModel): The Bayesian network model.
        target_variable (str): The name of the target variable.

    Returns:
        list: The list of variables in the Markov blanket of the target variable.
    """
    # Get the parents of the target variable
    parents = model.get_parents(target_variable)

    # Get the children of the target variable
    children = model.get_children(target_variable)

    # Get the other parents of the children (excluding the target variable)
    other_parents = []
    for child in children:
        other_parents.extend([parent for parent in model.get_parents(child) if parent != target_variable])

    # Combine parents, children, and other parents to form the Markov blanket
    markov_blanket = list(set(parents + children + other_parents))

    return markov_blanket


def draw_bayesian_network(model: BayesianNetwork):
    """Draws a Bayesian network using NetworkX and Matplotlib.

    Args:
        model (pgmpy.models.BayesianNetwork): The Bayesian network to draw.

    Returns:
        None
    """

    # Create a new directed graph using NetworkX
    graph = nx.DiGraph()

    # Add the nodes to the graph
    for node in model.nodes:
        graph.add_node(node)

    # Add the edges to the graph
    for edge in model.edges:
        graph.add_edge(edge[0], edge[1])

    # Set the positions of the nodes using a spring layout
    pos = nx.spring_layout(graph)

    # Create a grid for placing CPD tables and the main graph
    gs = gridspec.GridSpec(len(model.nodes) + 1, 2)
    main_ax = plt.subplot(gs[:-1, :])

    # Draw the nodes and edges using Matplotlib
    nx.draw_networkx_nodes(graph, pos, ax=main_ax)
    nx.draw_networkx_edges(graph, pos, ax=main_ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif", ax=main_ax)

    # Draw the CPDs as tables
    """for idx, node in enumerate(model.nodes):
        cpd = model.get_cpds(node)

        if len(cpd.values.shape) == 1:
            table = np.array(cpd.values, dtype=str).reshape(-1, 1)  # Convert values to strings and ensure 2D array
            columns = [cpd.variable]
        else:
            table = np.array(cpd.values, dtype=str).reshape(cpd.values.shape[0],
                                                            -1)  # Convert values to strings and flatten to 2D array
            # Generate column names for multi-dimensional CPD
            column_values = [f"{k}({v})" for k, values in cpd.state_names.items() if k != cpd.variable for v in values]
            columns = [' '.join(col) for col in itertools.product(column_values, repeat=cpd.values.ndim - 1)]

        ax = plt.subplot(gs[idx, -1])
        ax.axis("off")
        ax.table(cellText=table, colLabels=columns, loc="center")
        ax.set_title(node)"""

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


class DummyTrial:

    def suggest_int(self, name, _min, _max):
        return random.randint(_min, _max)

    def suggest_float(self, name, _min, _max):
        return random.uniform(_min, _max)

    def suggest_categorical(self, name, cats):
        return random.choice(cats)

    def set_user_attr(self, v0, v1):
        pass
