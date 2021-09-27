from pandas import DataFrame, Series
from pyensys.wrappers.PandaPowerManager import PandaPowerManager
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData, \
    OptimisationProfileData, OptimisationProfilesData
from pyensys.data_processing.clustering import TimeSeriesClustering, Birch_Settings
from typing import List, Dict
from dataclasses import dataclass, field
from networkx import DiGraph

@dataclass
class DataMultiplier:
    data_multiplier: float = 0.0
    element_type: str = ''
    variable_name: str = ''

@dataclass
class DataMultipliers:
    multipliers: List[DataMultiplier] = field(default_factory=list)
    initialised: bool = False

@dataclass
class GraphNode:
    scenarios: List[int] = field(default_factory=list)
    multiplier: float = -1.0
    level: int = -1

@dataclass
class GraphNodes:
    nodes: Dict[int, GraphNode] = field(default_factory=dict)
    initialised: bool = False

class RecursiveFunction:

    def __init__(self):
        self.graphs_multipliers: List[DataMultipliers] = []
        self.original_pp_profiles_data = PandaPowerProfilesData()
        self.graphs: List[DiGraph] = []
        self.parameters = Parameters()
        
    def operational_check(self):
        if self.parameters.problem_settings.opf_optimizer == "pandapower" and \
            self.parameters.problem_settings.intertemporal:
            self.pp_opf.run_timestep_opf_pandapower()
    
    def update_pandapower_controllers(self, current_node: int):
        new_profiles = PandaPowerProfilesData()
        new_profiles.initialised = True
        data_multipliers = self.get_data_multipliers_current_node(current_node)
        for multiplier in data_multipliers.multipliers:
            profile_position_to_update = self.get_profile_position_to_update(multiplier)
            profile = self.original_pp_profiles_data.data[profile_position_to_update]
            profile.data = profile.data * multiplier.data_multiplier
            new_profiles.data.append(profile)
        self.pp_opf.update_network_controllers(new_profiles)
    
    def get_profile_position_to_update(self, multiplier: DataMultiplier) -> int:
        profile_position_to_update = -1
        for position, pp_profile in enumerate(self.original_pp_profiles_data.data):
            if multiplier.element_type == pp_profile.element_type and \
                multiplier.variable_name == pp_profile.variable_name:
                profile_position_to_update = position
        return profile_position_to_update

    def get_data_multipliers_current_node(self, current_node: int) -> DataMultipliers:
        return self.graphs_multipliers[current_node]
        
    def initialise(self, parameters: Parameters):
        self.parameters = parameters
        if parameters.problem_settings.opf_optimizer == "pandapower":
            self.pp_opf = PandaPowerManager()
            self.pp_opf.initialise_pandapower_network(parameters)
            self.original_pp_profiles_data = parameters.pandapower_profiles_data
            profiles_clusters = create_profiles_clusters(parameters)
            for profile in profiles_clusters.data:
                self.create_graph(profile)

    def solve(self):
        self.operational_check()

    def create_graph(self, profile: OptimisationProfileData):
        graph_nodes = determine_graph_nodes(profile.data)
        self.graphs_multipliers.append(get_graph_multipliers(graph_nodes, profile))
        edges = determine_graph_edges(graph_nodes)
        graph = DiGraph()
        graph.add_edges_from(edges)
        self.graphs.append(graph)

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

def determine_graph_nodes_in_row(dataframe_row: Series) -> List[GraphNode]:
    stored_multipliers: Dict[float, int] = {}
    new_nodes: List[GraphNode] = []
    for column, (_, value) in enumerate(dataframe_row.iteritems()):
        if stored_multipliers.get(value, None) is not None:
            new_nodes[stored_multipliers[value]].scenarios.append(column)
        else:
            stored_multipliers[value] = len(new_nodes)
            new_nodes.append(GraphNode(scenarios=[column], multiplier=value))
    return new_nodes

def get_graph_multipliers(graph_nodes: GraphNodes, profile: OptimisationProfileData) -> DataMultipliers:
    multipliers = DataMultipliers()
    multipliers.multipliers = [DataMultiplier() for _ in graph_nodes.nodes.keys()]
    if graph_nodes.initialised:
        for position, node_info in graph_nodes.nodes.items():
            multipliers.multipliers[position].data_multiplier = node_info.multiplier
            multipliers.multipliers[position].element_type = profile.element_type
            multipliers.multipliers[position].variable_name = profile.variable_name
    return multipliers

def determine_graph_nodes(data: DataFrame) -> GraphNodes:
    nodes = GraphNodes()
    nodes.initialised = True
    node_counter = -1
    level = -1
    for index in data.index:
        new_nodes = determine_graph_nodes_in_row(data.loc[index])
        level += 1
        for node in new_nodes:
            node_counter += 1
            node.level = level
            nodes.nodes[node_counter] = node
    return nodes

def get_nodes_per_level(graph_nodes: GraphNodes) -> Dict[int, List[int]]:
    nodes_per_level: Dict[int, List[int]] = {}
    for position, node_info in graph_nodes.nodes.items():
        if nodes_per_level.get(node_info.level, None) is None:
            nodes_per_level[node_info.level] = [position]
        else:
            nodes_per_level[node_info.level].append(position)
    return nodes_per_level

def determine_graph_edges(graph_nodes: GraphNodes) -> List[List[int]]:
    edges: List[List[int]] = []
    if graph_nodes.initialised:
        nodes_per_level = get_nodes_per_level(graph_nodes)
        for level in range(len(nodes_per_level) - 1):
            for current_node in nodes_per_level[level]:
                for sucessor in nodes_per_level[level + 1]:
                    if any(item in graph_nodes.nodes[current_node].scenarios for item in \
                        graph_nodes.nodes[sucessor].scenarios):
                        edges.append([current_node, sucessor])
    return edges


        

