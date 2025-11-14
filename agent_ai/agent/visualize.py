from graph import analysis_graph
from IPython.display import Image, display

# Visualize graph
display(Image(analysis_graph.get_graph().draw_mermaid_png()))