from unittest import TestCase

import numpy as np
import networkx as nx

from causallearn.graph.Dag import Dag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag


class TestGeneralGraph(TestCase):
    def get_graph1(self):
        nodes = [GraphNode(str(i)) for i in range(0, 5)]
        truth_dag = Dag(nodes)
        truth_dag.add_directed_edge(nodes[0], nodes[2])
        truth_dag.add_directed_edge(nodes[1], nodes[2])
        truth_dag.add_directed_edge(nodes[2], nodes[3])
        truth_dag.add_directed_edge(nodes[2], nodes[4])

        return truth_dag

    def get_graph2(self):
        nodes = [GraphNode(str(i)) for i in range(0, 5)]
        truth_dag = Dag(nodes)
        truth_dag.add_directed_edge(nodes[0], nodes[3])
        truth_dag.add_directed_edge(nodes[1], nodes[3])
        truth_dag.add_directed_edge(nodes[3], nodes[2])
        truth_dag.add_directed_edge(nodes[3], nodes[4])

        return truth_dag

    def test_pc_case1(self):
        truth_dag = self.get_graph1()
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()
        num_nodes_in_truth = truth_dag.get_num_nodes()

        true_dag_netx = nx.DiGraph()
        true_dag_netx.add_nodes_from(list(range(num_nodes_in_truth)))
        true_dag_netx.add_edges_from(set(map(tuple, np.argwhere(truth_dag.graph.T > 0))))

        data = np.zeros((100, len(truth_dag.nodes)))
        cg = pc(data, 0.05, d_separation, True, 0, -1, true_dag=true_dag_netx)
        print(cg.G)
        # Graph Nodes:
        # X1;X2;X3;X4;X5
        #
        # Graph Edges:
        # 1. X1 --> X3
        # 2. X2 --> X3
        # 3. X3 --> X4
        # 4. X3 --> X5

    def test_pc_case2(self):
        nodes = [GraphNode(str(i)) for i in range(0, 5)]
        truth_dag = Dag(nodes)
        truth_dag.add_directed_edge(nodes[0], nodes[3])
        truth_dag.add_directed_edge(nodes[1], nodes[3])
        truth_dag.add_directed_edge(nodes[3], nodes[2])
        truth_dag.add_directed_edge(nodes[3], nodes[4])

        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()
        num_nodes_in_truth = truth_dag.get_num_nodes()

        true_dag_netx = nx.DiGraph()
        true_dag_netx.add_nodes_from(list(range(num_nodes_in_truth)))
        true_dag_netx.add_edges_from(set(map(tuple, np.argwhere(truth_dag.graph.T > 0))))

        data = np.zeros((100, len(truth_dag.nodes)))
        cg = pc(data, 0.05, d_separation, True, 0, -1, true_dag=true_dag_netx)
        print(cg.G)
        # Graph Nodes:
        # X1;X2;X3;X4;X5
        #
        # Graph Edges:
        # 1. X1 --> X4
        # 2. X2 --> X4
        # 3. X3 --- X4
        # 4. X4 --> X5

    def test_case1(self):
        nodes = [GraphNode(str(i)) for i in range(0, 5)]
        graph = GeneralGraph(nodes)
        no_of_var = len(nodes)
        # Line 35 - 37 of the file causallearn/graph/GraphClass.py
        # start
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
        # end

        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                edge = graph.get_edge(nodes[i], nodes[j])
                graph.remove_edge(edge)

        graph.add_edge(Edge(nodes[0], nodes[3], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[1], nodes[3], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[3], nodes[2], Endpoint.TAIL, Endpoint.ARROW))
        graph.add_edge(Edge(nodes[3], nodes[4], Endpoint.TAIL, Endpoint.ARROW))

        print(graph.get_edge(nodes[3], nodes[2]))
        print(f'{nodes[3]} is ancestor of {nodes[2]}: {graph.is_ancestor_of(nodes[3], nodes[2])}')
        print(f'{nodes[2]} is ancestor of {nodes[3]}: {graph.is_ancestor_of(nodes[2], nodes[3])}')
        # 3 --> 2
        # 3 is ancestor of 2: True
        # 2 is ancestor of 3: True

        assert graph.is_ancestor_of(nodes[3], nodes[2])
        assert not graph.is_ancestor_of(nodes[2], nodes[3])