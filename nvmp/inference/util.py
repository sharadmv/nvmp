from ..core import get_current_graph

def top_sort(graph):
    fringe = set()
    ordering = []
    for node in graph:
        if len(node.parents()) == 0:
            fringe.add(node)

    while len(fringe) > 0:
        for node in fringe:
            if len(node.parents() - set(ordering)) == 0:
                fringe.remove(node)
                ordering.append(node)
                for child in node.children():
                    fringe.add(child)
                break
    return [o for o in ordering if o in graph]
