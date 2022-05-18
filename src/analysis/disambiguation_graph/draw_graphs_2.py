import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# create graph from data
with open("dis_graph.csv", "r") as f:
   G = nx.parse_edgelist(f.readlines(), delimiter = ",")

# centrality
#deg_centrality = nx.degree_centrality(G)
#centrality = np.fromiter(deg_centrality.values(), float)
# plot
#pos = nx.kamada_kawai_layout(G)
#nx.draw(G, pos, node_color = centrality, node_size = centrality * 2e3)
#nx.draw_networkx_labels(G, pos)
#plt.show()