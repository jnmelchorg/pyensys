from networkx import DiGraph, is_tree

class DirectedGraph:
    def __init__(self):
        self._graph = DiGraph()
    
    def __len__(self):
        return len(self._graph)

    def add_edge(self, from_node:int, to_node:int):
        self._graph.add_edge(from_node, to_node)
    
    def add_node(self, node:int):
        self._graph.add_node(node)
    
    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()
    
    def neighbours(self, node:int):
        return self._graph.neighbors(node)
    
    def predecessors(self, node:int):
        return self._graph.predecessors(node)
    
    def get_first_node(self) -> int:
        return [x for c, x in enumerate(self._graph.nodes()) if c == 0][0]
    
    def find_node_with_greatest_numbering(self) -> int:
        node_to_be_explored = self.get_first_node()
        return self._recursive_search_for_greatest_numbering(node_to_be_explored, node_to_be_explored)

    def _recursive_search_for_greatest_numbering(self, node_to_be_explored:int, greatest_node_number:int) -> int:
        for node in self.neighbours(node_to_be_explored):
            if node > greatest_node_number:
                greatest_node_number = node
            greatest_node_number = self._recursive_search_for_greatest_numbering(node, greatest_node_number)
        return greatest_node_number
    
    def nodes_iterator(self):
        return self._graph.nodes()
    
    def is_tree(self) -> bool:
        return is_tree(self._graph)
    
    def has_node(self, node:int) -> bool:
        return self._graph.has_node(node)
    
    def has_edge(self, from_node:int, to_node:int) -> bool:
        return self._graph.has_edge(from_node, to_node)