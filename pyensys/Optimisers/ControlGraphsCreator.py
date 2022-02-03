from typing import List, Dict
from dataclasses import dataclass, field
from networkx import DiGraph
from pandas import DataFrame

from pyensys.readers.ReaderDataClasses import OptimisationProfileData, OptimisationProfilesData, \
    Parameters
from pyensys.data_processing.clustering import TimeSeriesClustering, Birch_Settings

@dataclass
class ClusterData:
    scenarios: List[int] = field(default_factory=list)
    centroid: float = -1.0
    level: int = -1
    element_type: str = ''
    variable_name: str = ''

class GraphNodesCreator:
    def __init__(self):
        self._profile = OptimisationProfileData()
        self._nodes_in_dataframe: List[List[ClusterData]] = []
        self._graph_nodes: Dict[int, ClusterData] = {}

    def determine_graph_nodes(self, profile: OptimisationProfileData) -> Dict[int, ClusterData]:
        self._profile = profile
        self._determine_nodes_in_profile()
        self._add_final_information_to_nodes()
        return self._graph_nodes
    
    def _determine_nodes_in_profile(self):
        for row in self._profile.data.index:
            clusters_centroids: Dict[float, int] = {}
            new_nodes: List[ClusterData] = []
            for column, (_, value) in enumerate(self._profile.data.loc[row].iteritems()):
                if clusters_centroids.get(value, None) is not None:
                    new_nodes[clusters_centroids[value]].scenarios.append(column)
                else:
                    clusters_centroids[value] = len(new_nodes)
                    new_nodes.append(ClusterData(scenarios=[column], centroid=value))
            self._nodes_in_dataframe.append(new_nodes)

    def _add_final_information_to_nodes(self):
        level = -1
        node_counter = -1
        for nodes_per_row in self._nodes_in_dataframe:
            level += 1
            for node in nodes_per_row:
                node.level = level
                node.element_type = self._profile.element_type
                node.variable_name = self._profile.variable_name
                node_counter += 1
                self._graph_nodes[node_counter] = node
@dataclass
class EdgeData:
    end1: int = -1
    end2: int = -1

class GraphEdgesCreator:
    def __init__(self):
        self._graph_nodes: Dict[int, ClusterData] = {}
        self._graph_edges: List[EdgeData] = []
        self._nodes_all_levels: Dict[int, List[int]] = {}
    
    def determine_graph_edges(self, graph_nodes: Dict[int, ClusterData]) -> List[EdgeData]:
        self._graph_nodes = graph_nodes
        self._get_nodes_all_levels()
        for level in range(len(self._nodes_all_levels) - 1):
            for current_node in self._nodes_all_levels[level]:
                for sucessor in self._nodes_all_levels[level + 1]:
                    if any(item in graph_nodes[current_node].scenarios for item in \
                        graph_nodes[sucessor].scenarios):
                        self._graph_edges.append(EdgeData(end1=current_node, end2=sucessor))
        return self._graph_edges
    
    def _get_nodes_all_levels(self):
        for position, node_info in self._graph_nodes.items():
            if self._nodes_all_levels.get(node_info.level, None) is None:
                self._nodes_all_levels[node_info.level] = [position]
            else:
                self._nodes_all_levels[node_info.level].append(position)
        return self._nodes_all_levels

@dataclass
class ProfileGraphData:
    nodes_data: Dict[int, ClusterData] = field(default_factory=dict)
    graph: DiGraph = field(default_factory=DiGraph)

class ProfileGraphCreator:
    def __init__(self):
        self._edges: List[EdgeData] = []
        self._profile_graph = ProfileGraphData()
        self._profile = OptimisationProfileData()

    def create_graph(self, profile: OptimisationProfileData) -> ProfileGraphData:
        self._profile = profile
        self._create_nodes_and_edges()
        for edge in self._edges:
            self._profile_graph.graph.add_edge(edge.end1, edge.end2)
        return self._profile_graph
    
    def _create_nodes_and_edges(self):
        nodes_creator = GraphNodesCreator()
        self._profile_graph.nodes_data = nodes_creator.determine_graph_nodes(self._profile)
        edges_creator = GraphEdgesCreator()
        self._edges = edges_creator.determine_graph_edges(self._profile_graph.nodes_data)

class AllProfilesGraphsCreator:
    def __init__(self):
        self._graphs: List[ProfileGraphData] = []
    
    def create_all_profile_graphs(self, profiles: OptimisationProfilesData) -> \
        List[ProfileGraphData]:
        for profile in profiles.data:
            graph_creator = ProfileGraphCreator()
            self._graphs.append(graph_creator.create_graph(profile))
        return self._graphs

def create_profile_clusters(profile: OptimisationProfileData) -> DataFrame:
    clusters = TimeSeriesClustering()
    clusters.set_time_series(profile.data)
    clusters.initialise_birch_clustering_algorithm(Birch_Settings())
    clusters.perform_clustering()
    clusters.calculate_clusters_centroids()
    return DataFrame(data=clusters.clusters_centroids, columns=profile.data.columns)

def create_profiles_clusters(parameters: Parameters) -> OptimisationProfilesData:
    series_to_clusters: OptimisationProfilesData = OptimisationProfilesData(initialised=True)
    if parameters.optimisation_profiles_data.initialised:
        for profile in parameters.optimisation_profiles_data.data:
            data = create_profile_clusters(profile)
            series_to_clusters.data.append(OptimisationProfileData(element_type=profile.element_type, \
                variable_name=profile.variable_name, data=data))
    return series_to_clusters

@dataclass
class ControlGraphData:
    nodes_data: Dict[int, List[ClusterData]] = field(default_factory=dict)
    graph: DiGraph = field(default_factory=DiGraph)

class RecursiveFunctionGraphCreator:
    def __init__(self):
        self._control_graph = ControlGraphData()
        self._profiles_graphs: List[ProfileGraphData] = []
        self._profiles = Parameters()
    
    def create_recursive_function_graph(self, parameters: Parameters) -> ControlGraphData:
        self._profiles = create_profiles_clusters(parameters)
        self._create_all_profiles_graphs()
        self._create_control_graph()
        return self._control_graph
        
    def _create_all_profiles_graphs(self):
        profiles_graphs_creator = AllProfilesGraphsCreator()
        self._profiles_graphs = profiles_graphs_creator.create_all_profile_graphs(self._profiles)
    
    def _create_control_graph(self):
        if len(self._profiles_graphs) == 1:
            self._control_graph.graph = self._profiles_graphs[0].graph
            for key, value in self._profiles_graphs[0].nodes_data.items():
                self._control_graph.nodes_data[key] = [value]
        else:
            print("THIS OPTION NEEDS TO BE DEVELOPED FOR MORE THAN ONE GRAPH")
