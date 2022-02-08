from pyensys.Interfaces.networkx_interface import DirectedGraph


def test_add_edge_to_directed_graph():
    G = DirectedGraph()
    G.add_edge(0,1)
    assert G._graph.number_of_nodes() == 2

def test_len_of_directed_graph():
    G = DirectedGraph()
    G.add_edge(0,1)
    assert len(G._graph) == 2
    
def test_get_neighbours_in_directed_graph():
    G = DirectedGraph()
    G.add_edge(100, 19)
    G.add_edge(100, 30)
    assert [19, 30] == [x for x in G.neighbours(100)]

def test_recursive_search_for_greatest_numbering_in_directed_graph():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    assert G._recursive_search_for_greatest_numbering(19, 19) == 100

def test_find_node_with_greatest_numbering():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    assert G.find_node_with_greatest_numbering() == 100

def test_nodes_iterator_in_directed_graph():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    assert [19, 100, 30] == [x for x in G.nodes_iterator()]

def test_is_tree_directed_graph():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    assert G.is_tree()

def test_is_not_tree_directed_graph():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    G.add_edge(100, 30)
    assert not G.is_tree()

def test_get_first_node_in_directed_graph():
    G = DirectedGraph()
    G.add_edge(19, 100)
    G.add_edge(19, 30)
    assert G.get_first_node() == 19

def test_get_predecessors_directed_graph():
    G = DirectedGraph()
    G.add_edge(100, 19)
    G.add_edge(30, 19)
    assert [100, 30] == [x for x in G.predecessors(19)]

def test_add_node():
    G = DirectedGraph()
    G.add_node(4)
    assert len(G) == 1

def test_has_node():
    G = DirectedGraph()
    G.add_edge(100, 19)
    assert G.has_node(100)

def test_does_not_have_node():
    G = DirectedGraph()
    G.add_edge(100, 19)
    assert not G.has_node(40)

def test_has_edge():
    G = DirectedGraph()
    G.add_edge(100, 19)
    assert G.has_edge(100, 19)