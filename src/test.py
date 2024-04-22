import numpy as np
import networkx as nx
from random import random, randint
from math import floor, log

np.random.seed(44)


def nearest_neigbor(vec_pos, query_vec):
    nearest_neighbor_index = -1
    nearest_dist = float('inf')

    nodes = []
    edges = []
    for i in range(np.shape(vec_pos)[0]):
        nodes.append((i, {"pos": vec_pos[i, :]}))
        if i < np.shape(vec_pos)[0] - 1:
            edges.append((i, i + 1))
        else:
            edges.append((i, 0))

        dist = np.linalg.norm(query_vec - vec_pos[i])
        if dist < nearest_dist:
            nearest_neighbor_index = i
            nearest_dist = dist

    G_lin = nx.Graph()
    G_lin.add_nodes_from(nodes)
    G_lin.add_edges_from(edges)

    nodes = []
    nodes.append(("*", {"pos": vec_pos[nearest_neighbor_index, :]}))
    G_best = nx.Graph()
    G_best.add_nodes_from(nodes)
    return G_lin, G_best


def layer_num(max_layers: int):
    # new element's topmost layer: notice the normalization by mL
    mL = 1.5
    layer_i = floor(-1 * log(random()) * mL)
    # ensure we don't exceed our allocated layers.
    layer_i = min(layer_i, max_layers - 1)
    return layer_i
    # return randint(0,max_layers-1)


def construct_HNSW(vec_pos, m_nearest_neighbor):
    max_layers = 4

    vec_num = np.shape(vec_pos)[0]
    dist_mat = np.zeros((vec_num, vec_num))

    for i in range(vec_num):
        for j in range(i, vec_num):
            dist = np.linalg.norm(vec_pos[i, :] - vec_pos[j, :])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    node_layer = []
    for i in range(np.shape(vec_pos)[0]):
        node_layer.append(layer_num(max_layers))

    max_num_of_layers = max(node_layer) + 1  ## layer indices start from 0
    GraphArray = []
    for layer_i in range(max_num_of_layers):
        nodes = []
        edges = []
        edges_nn = []
        for i in range(np.shape(vec_pos)[0]):  ## Number of Vectors
            if node_layer[i] >= layer_i:
                nodes.append((i, {"pos": vec_pos[i, :]}))

        G = nx.Graph()
        G.add_nodes_from(nodes)

        pos = nx.get_node_attributes(G, 'pos')

        for i in range(len(G.nodes)):
            node_i = nodes[i][0]
            nearest_edges = -1
            nearest_distances = float('inf')
            candidate_edges = range(0, i)
            candidate_edges_indices = []

            #######################
            for j in candidate_edges:
                node_j = nodes[j][0]
                candidate_edges_indices.append(node_j)

            dist_from_node = dist_mat[node_i, candidate_edges_indices]
            num_nearest_neighbor = min(m_nearest_neighbor, i)  ### Add note comment

            if num_nearest_neighbor > 0:
                indices = np.argsort(dist_from_node)
                for nn_i in range(num_nearest_neighbor):
                    edges_nn.append((node_i, candidate_edges_indices[indices[nn_i]]))

            for j in candidate_edges:
                node_j = nodes[j][0]
                dist = np.linalg.norm(pos[node_i] - pos[node_j])
                if dist < nearest_distances:
                    nearest_edges = node_j
                    nearest_distances = dist

            if nearest_edges != -1:
                edges.append((node_i, nearest_edges))

        G.add_edges_from(edges_nn)

        GraphArray.append(G)

    return GraphArray


## Search the Graph
def search_HNSW(GraphArray, G_query):
    max_layers = len(GraphArray)
    G_top_layer = GraphArray[max_layers - 1]
    num_nodes = G_top_layer.number_of_nodes()
    entry_node_r = randint(0, num_nodes - 1)
    nodes_list = list(G_top_layer.nodes)
    entry_node_index = nodes_list[entry_node_r]
    # entry_node_index = 26

    SearchPathGraphArray = []
    EntryGraphArray = []
    for l_i in range(max_layers):
        layer_i = max_layers - l_i - 1
        G_layer = GraphArray[layer_i]

        G_entry = nx.Graph()
        nodes = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes.append((entry_node_index, {"pos": p}))
        G_entry.add_nodes_from(nodes)

        nearest_node_layer = entry_node_index
        nearest_distance_layer = np.linalg.norm(G_layer.nodes[entry_node_index]['pos'] - G_query.nodes['Q']['pos'])
        current_node_index = entry_node_index

        G_path_layer = nx.Graph()
        nodes_path = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes_path.append((entry_node_index, {"pos": p}))

        cond = True
        while cond:
            nearest_node_current = -1
            nearest_distance_current = float('inf')
            for neihbor_i in G_layer.neighbors(current_node_index):
                vec1 = G_layer.nodes[neihbor_i]['pos']
                vec2 = G_query.nodes['Q']['pos']
                dist = np.linalg.norm(vec1 - vec2)
                if dist < nearest_distance_current:
                    nearest_node_current = neihbor_i
                    nearest_distance_current = dist

            if nearest_distance_current < nearest_distance_layer:
                nearest_node_layer = nearest_node_current
                nearest_distance_layer = nearest_distance_current
                nodes_path.append((nearest_node_current, {"pos": G_layer.nodes[nearest_node_current]['pos']}))
            else:
                cond = False

        entry_node_index = nearest_node_layer

        G_path_layer.add_nodes_from(nodes_path)
        SearchPathGraphArray.append(G_path_layer)
        EntryGraphArray.append(G_entry)

    SearchPathGraphArray.reverse()
    EntryGraphArray.reverse()

    return SearchPathGraphArray, EntryGraphArray


from random import random, randint
from math import floor, log
import networkx as nx
import numpy as np
import matplotlib as mtplt
from matplotlib import pyplot as plt
from utils import *

vec_num = 40  # Number of vectors (nodes)
dim = 2  ## Dimention. Set to be 2. All the graph plots are for dim 2. If changed, then plots should be commented.
m_nearest_neighbor = 2  # M Nearest Neigbor used in construction of the Navigable Small World (NSW)

vec_pos = np.random.uniform(size=(vec_num, dim))

## Query
query_vec = [0.5, 0.5]

nodes = []
nodes.append(("Q", {"pos": query_vec}))

G_query = nx.Graph()
G_query.add_nodes_from(nodes)

print("nodes = ", nodes, flush=True)

pos_query = nx.get_node_attributes(G_query, 'pos')

(G_lin, G_best) = nearest_neigbor(vec_pos, query_vec)

pos_lin = nx.get_node_attributes(G_lin, 'pos')
pos_best = nx.get_node_attributes(G_best, 'pos')

fig, axs = plt.subplots()

nx.draw(G_lin, pos_lin, with_labels=True, node_size=150, node_color=[[0.8, 0.8, 1]], width=0.0, font_size=7, ax=axs)
nx.draw(G_query, pos_query, with_labels=True, node_size=200, node_color=[[0.5, 0, 0]], font_color='white', width=0.5,
        font_size=7, font_weight='bold', ax=axs)
nx.draw(G_best, pos_best, with_labels=True, node_size=200, node_color=[[0.85, 0.7, 0.2]], width=0.5, font_size=7,
        font_weight='bold', ax=axs)

GraphArray = construct_HNSW(vec_pos, m_nearest_neighbor)

for layer_i in range(len(GraphArray) - 1, -1, -1):
    fig, axs = plt.subplots()

    print("layer_i = ", layer_i)

    if layer_i > 0:
        pos_layer_0 = nx.get_node_attributes(GraphArray[0], 'pos')
        nx.draw(GraphArray[0], pos_layer_0, with_labels=True, node_size=120, node_color=[[0.9, 0.9, 1]], width=0.0,
                font_size=6, font_color=(0.65, 0.65, 0.65), ax=axs)

    pos_layer_i = nx.get_node_attributes(GraphArray[layer_i], 'pos')
    nx.draw(GraphArray[layer_i], pos_layer_i, with_labels=True, node_size=150, node_color=[[0.7, 0.7, 1]], width=0.5,
            font_size=7, ax=axs)
    nx.draw(G_query, pos_query, with_labels=True, node_size=200, node_color=[[0.8, 0, 0]], width=0.5, font_size=7,
            font_weight='bold', ax=axs)
    nx.draw(G_best, pos_best, with_labels=True, node_size=200, node_color=[[0.85, 0.7, 0.2]], width=0.5, font_size=7,
            font_weight='bold', ax=axs)
    plt.show()

(SearchPathGraphArray, EntryGraphArray) = search_HNSW(GraphArray, G_query)

for layer_i in range(len(GraphArray) - 1, -1, -1):
    fig, axs = plt.subplots()

    print("layer_i = ", layer_i)
    G_path_layer = SearchPathGraphArray[layer_i]
    pos_path = nx.get_node_attributes(G_path_layer, 'pos')
    G_entry = EntryGraphArray[layer_i]
    pos_entry = nx.get_node_attributes(G_entry, 'pos')

    if layer_i > 0:
        pos_layer_0 = nx.get_node_attributes(GraphArray[0], 'pos')
        nx.draw(GraphArray[0], pos_layer_0, with_labels=True, node_size=120, node_color=[[0.9, 0.9, 1]], width=0.0,
                font_size=6, font_color=(0.65, 0.65, 0.65), ax=axs)

    pos_layer_i = nx.get_node_attributes(GraphArray[layer_i], 'pos')
    nx.draw(GraphArray[layer_i], pos_layer_i, with_labels=True, node_size=100, node_color=[[0.7, 0.7, 1]], width=0.5,
            font_size=6, ax=axs)
    nx.draw(G_path_layer, pos_path, with_labels=True, node_size=110, node_color=[[0.8, 1, 0.8]], width=0.5, font_size=6,
            ax=axs)
    nx.draw(G_query, pos_query, with_labels=True, node_size=80, node_color=[[0.8, 0, 0]], width=0.5, font_size=7,
            ax=axs)
    nx.draw(G_best, pos_best, with_labels=True, node_size=70, node_color=[[0.85, 0.7, 0.2]], width=0.5, font_size=7,
            ax=axs)
    nx.draw(G_entry, pos_entry, with_labels=True, node_size=80, node_color=[[0.1, 0.9, 0.1]], width=0.5, font_size=7,
            ax=axs)
    plt.show()