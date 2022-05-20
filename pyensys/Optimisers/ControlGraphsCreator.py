from typing import List, Dict, Set
from dataclasses import dataclass, field

from pandas import DataFrame, concat

from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.readers.ReaderDataClasses import OptimisationProfileData, OptimisationProfilesData, \
    Parameters
from pyensys.data_processing.clustering import TimeSeriesClustering, Birch_Settings
from pyensys.wrappers.networkx_wrapper import DirectedGraph


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
class GraphandClusterData:
    nodes_data: Dict[int, ClusterData] = field(default_factory=dict)
    graph: DirectedGraph = field(default_factory=lambda: DirectedGraph())
    map_node_to_data_power_system: Dict[int, AbstractDataContainer] = field(default_factory=dict)


class GraphtoTreeConverter:
    def __init__(self, graph_and_data: GraphandClusterData):
        self._nodes_data = graph_and_data.nodes_data
        self._graph = graph_and_data.graph
        self._tree = DirectedGraph()

    @dataclass
    class CommonNodeData:
        predecessors: Set[int] = field(default_factory=set())
        node_under_analysis: int = -1

    def convert(self) -> GraphandClusterData:
        if not self._graph.is_tree():
            greatest_node = self._graph.find_node_with_greatest_numbering()
            current_layer = self._find_last_nodes_in_graph()
            all_layers_analysed = False
            while not all_layers_analysed:
                next_layer = set()
                for node in current_layer:
                    predecessors = set(self._graph_predecessors_list(node))
                    next_layer.update(predecessors)
                    if len(next_layer) == 0: break
                    greatest_node = self._create_duplicates_from_common_node(
                        self.CommonNodeData(predecessors, node), greatest_node)
                if len(next_layer) == 0:
                    all_layers_analysed = True
                else:
                    current_layer = next_layer
            return GraphandClusterData(self._nodes_data, self._tree)
        else:
            return GraphandClusterData(self._nodes_data, self._graph)

    def _create_duplicates_from_common_node(self, common_node: CommonNodeData, greatest_node_number: int) -> int:
        self._tree.add_edge(common_node.predecessors.pop(), common_node.node_under_analysis)
        for node in common_node.predecessors:
            greatest_node_number += 1
            self._tree.add_edge(node, greatest_node_number)
            self._nodes_data[greatest_node_number] = self._nodes_data[common_node.node_under_analysis]
            greatest_node_number = self._recursive_duplicates_for_sucessors_of_common_node( \
                common_node.node_under_analysis, greatest_node_number)
        return greatest_node_number

    def _recursive_duplicates_for_sucessors_of_common_node(self, node_under_analysis: int,
                                                           greatest_node_number: int) -> int:
        current_node = greatest_node_number
        for successor in self._tree.neighbours(node_under_analysis):
            greatest_node_number += 1
            self._tree.add_edge(current_node, greatest_node_number)
            self._nodes_data[greatest_node_number] = self._nodes_data[successor]
            greatest_node_number = self._recursive_duplicates_for_sucessors_of_common_node(successor,
                                                                                           greatest_node_number)
        return greatest_node_number

    def _graph_predecessors_list(self, node_to_be_explored: int) -> List[int]:
        return [x for x in self._graph.predecessors(node_to_be_explored)]

    def _find_last_nodes_in_graph(self) -> Set[int]:
        first_node = self._graph.get_first_node()
        return self._recursive_search_of_last_nodes_in_graph(first_node, set())

    def _recursive_search_of_last_nodes_in_graph(self, node_to_be_explored: int, last_nodes: Set[int]) -> Set[int]:
        is_last_node = True
        for node in self._graph.neighbours(node_to_be_explored):
            is_last_node = False
            last_nodes = self._recursive_search_of_last_nodes_in_graph(node, last_nodes)
        if is_last_node:
            last_nodes.add(node_to_be_explored)
        return last_nodes


class ProfileGraphCreator:
    def __init__(self):
        self._edges: List[EdgeData] = []
        self._profile_graph = GraphandClusterData()
        self._profile = OptimisationProfileData()

    def create_graph(self, profile: OptimisationProfileData) -> GraphandClusterData:
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
        self._graphs: List[GraphandClusterData] = []

    def create_all_profile_graphs(self, profiles: OptimisationProfilesData) -> \
            List[GraphandClusterData]:
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
    return DataFrame(data=clusters.clusters_centroids, columns=profile.data.columns, index=profile.data.index)


def create_profiles_clusters(parameters: Parameters) -> OptimisationProfilesData:
    series_to_clusters: OptimisationProfilesData = OptimisationProfilesData(initialised=True)
    if parameters.optimisation_profiles_data.initialised:
        for profile in parameters.optimisation_profiles_data.data:
            data = create_profile_clusters(profile)
            series_to_clusters.data.append(OptimisationProfileData(element_type=profile.element_type,
                                                                   variable_name=profile.variable_name, data=data))
    return series_to_clusters


class RecursiveFunctionGraphCreator:
    def __init__(self):
        self._control_graph = GraphandClusterData()
        self._profiles_graphs: List[GraphandClusterData] = []
        self._profiles = OptimisationProfilesData()

    def create_recursive_function_graph(self, parameters: Parameters) -> GraphandClusterData:
        self._profiles = create_profiles_clusters(parameters)
        self._create_all_profiles_graphs()
        self._create_control_graph()
        if parameters.problem_settings.non_anticipative:
            self._create_control_tree_graph()
        if len(parameters.optimisation_profiles_dataframes) > 0:
            self._create_map_node_to_data_power_system(parameters)
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

    def _initialise_class_graphtreeconverter(self) -> GraphtoTreeConverter:
        return GraphtoTreeConverter(self._control_graph)

    def _create_control_tree_graph(self):
        converter = self._initialise_class_graphtreeconverter()
        self._control_graph = converter.convert()

    def _create_map_node_to_data_power_system(self, parameters: Parameters):
        scenarios_name_raw = list(self._profiles.data[0].data.columns)
        scenarios_number = []
        for scenario in scenarios_name_raw:
            scenarios_number.append(int(scenario.split(" ")[1]))
        years_names = list(self._profiles.data[0].data.index)
        years_number = []
        for year in years_names:
            years_number.append(int(year))
        for number_node, node in self._control_graph.nodes_data.items():
            self._control_graph.map_node_to_data_power_system[number_node] = AbstractDataContainer()
            self._control_graph.map_node_to_data_power_system[number_node].create_dictionary()
            for group, data in parameters.optimisation_profiles_dataframes:
                if group == "buses":
                    related_data_to_node = data[data["scenario"].isin([scenarios_number[y] for y in node[0].scenarios])]
                    related_data_to_node = related_data_to_node[related_data_to_node["year"] ==
                                                                years_number[node[0].level]]
                    elements_numbers_in_data =list(related_data_to_node["bus_index"].unique())
                    averaged_data_for_elements = DataFrame()
                    for number in elements_numbers_in_data:
                        specific_bus_data = related_data_to_node[related_data_to_node["bus_index"] == number]
                        averaged_data_for_elements = concat([averaged_data_for_elements, DataFrame(
                            data=[[years_number[node[0].level], number, specific_bus_data["p_mw"].mean(),
                                   specific_bus_data["q_mvar"].mean()]],
                            columns=["year", "bus_index", "p_mw", "q_mvar"])], ignore_index=True)
                    self._control_graph.map_node_to_data_power_system[number_node].append(group,
                                                                                          averaged_data_for_elements)
