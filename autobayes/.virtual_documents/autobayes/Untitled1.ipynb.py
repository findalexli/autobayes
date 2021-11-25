get_ipython().getoutput("pip uninstall pygraphviz")


get_ipython().getoutput("pip install --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz")


import cdt


data, graph = cdt.data.load_dataset('sachs')


from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE


graph.edges


viz = plot_structure(
    graph,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
Image(viz.draw(format='png'))


import matplotlib.pyplot as plt
import networkx as nx
fig, ax = plt.subplots()
nx.draw_circular(graph, ax=ax)
fig.show()



