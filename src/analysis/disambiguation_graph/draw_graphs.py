import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


ents = {
 
    "basal cell carcinoma": [(2671, "red")],

    "autosomal dominant": [(6, "blue"),
                        (1470, "blue"),
                        (1475, "blue"),
                        (1444, "blue"),
                        (1452, "blue"),
                        (5274, "blue"),
                        (9921, "blue"),
                        (12447, "blue"),
                        (25352, "blue"),
                        (7, "blue")],

    "dominant": [(1488, "black"),
                 (1082, "black")],

    "variable": [(1806, "yellow"),
                 (1363, "yellow")],
    
    "variable expressivity": [(3828, "brown")],

    "jaw cysts": [(2909, "purple"),
                  (2028, "purple")],

    "palmar and/or plantar pits": [(10612, "orange"),
                                   (10506, "orange"),
                                   (982, "orange"),
                                   (12769, "orange"),
                                   (25530, "orange"),
                                   (10610, "orange"),
                                   (100701, "orange"),
                                   (100767, "orange"),
                                   (6529, "orange"),
                                   (1134, "orange"),
                                   (982, "orange")],
    
    "plantar pits": [(10612, "grey")],

    "pits": [(2000, "black"),
            (1259, " black")],

    "calcification of the falx cerebri": [(5462, "blue"),
                                          (2514, "blue"),
                                          (10654, "blue"),
                                          (5849, "blue"),
                                          (10653, "blue"),
                                          (31309, "blue"),
                                          (100593, "blue"),
                                          (5146, "blue"),
                                          (7862, "blue"),
                                          (31311, "blue")],

    "skin tumors": [(8069, "black"),
                    (100245, "black"),
                    (10609, "black"),
                    (200042, "black"),
                    (2322, "black"),
                    (10302, "black"),
                    (30186, "black"),
                    (100570, "black"),
                    (963, "black"),
                    (977, "black"),
                    (10889, "black"),
                    (10302, "black"),
                    (30691, "black")],

    "tumors": [(2664, "light purple"),
               (1337, "light purple"),
               (31947, "light purple"),
               (322, "light purple"),
               (32006, "light purple"),
               (32445, "light purple"),
               (100245, "light purple"),
               (2174, "light purple"),
               (2346, "light purple"),
               (2378, "light purple"),
               (10307, "light purple")],
    
    "increased skin pigmentation": [(2075, "orange"),
                                    (1413, "orange")]
    
    }
#12447;2664;100767;10612;30186;10653;8069;2671;100245;1488;6;982;5462;10654;7;963;1337;2514;10609;322;100570;1413;977;30692;1363;5146;3828;1806;200042;10610;2028
#edges_nilinker = [(2671, 12447), (2671, 2664), (2671,100767), (2671,10612), (2671,30186), (2671,10653), (2671,8069), (2671, 2671), (2671,100245), (2671,1488), (2671,6), (2671,982), (2671,5462), (2671,10654), (2671,7), (2671,963), (2671,1337),(2671,2514), (2671,10609), (2671,322), (2671,100570), (2671,1413), (2671,977), (2671,30692), (2671,1363), (2671,5146), (2671,3828), (2671,1806), (2671, 200042), (2671, 10610), (2671,2028),
#    (6, 2664), (6,2909), (6,10612), (6,100767), (6,30186), (6,10653), (6, 8069), (6, 2671), (6, 1134), (6, 100245), (6, 1259), (6, 2174), (6, 10307), (6, 1488), (6, 7862), (6, 982), (6, 5462), (6, 10654), (6, 963), (6, 1337), (6, 2514), (6, 31947), (6, 6529), (6, 32445), (6, 12769), (6, 10609), (6, 2322), (6, 2346), (6, 322), (6, 10302), (6, 2000), (6,100570), (6, 1413), (6, 100593), (6, 977), (6, 30692), (6, 1363), (6, 1082), (6, 5146), (6, 3828), (6, 1806), (6, 2378), (6, 200042), (6, 10610), (6, 2075), (6, 2028)
#    ]
with open("edges", "r") as edges_file:
    edges = edges_file.read()
    print(edges.split("\n"))
    dis_graph = nx.parse_edgelist(edges.split("\n"))

#dis_graph = nx.Graph()
color_map = list()

for ent in ents:

    for node in ents[ent]:
        #dis_graph.add_node(node[0], color=node[1])
        color_map.append(node[1])

#for edge in edges_nilinker:
    #print(edge)
#    dis_graph.add_edge(*edge)


#colors = [node[1]['color'] for node in dis_graph.nodes(data=True)]

#colors2 = list()

#for node in colors:
#    if node not in colors2:
#        colors2.append(node)

#colors2 = np.array(colors2)
#print(colors2)
#nx.draw(dis_graph, node_color=colors2, with_labels=True, font_color='white')
#plt.show()
#plt.savefig("nilinker.png")
# centrality
deg_centrality = nx.degree_centrality(dis_graph)
#centrality = np.fromiter(deg_centrality.values(), float)
# plot
pos = nx.kamada_kawai_layout(dis_graph)
#nx.draw(dis_graph, pos, node_color = centrality, node_size = centrality * 2e3)
#nx.draw_networkx_labels(dis_graph, pos)
#plt.show()

#nx.draw(dis_graph, node_color=color_map, with_labels=True)
#plt.show()

cent = np.fromiter(deg_centrality.values(), float)
sizes = cent / np.max(cent) * 200
normalize = mcolors.Normalize(vmin=cent.min(), vmax=cent.max())
colormap = cm.viridis

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(cent)
print(nx.info(dis_graph))
plt.colorbar(scalarmappaple)
nx.draw(dis_graph, pos, node_size=sizes, node_color=sizes, cmap=colormap, with_labels=True)
plt.show()