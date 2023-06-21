from parsing.dataset import build_test_dataset
from parsing.config import cfg
import os
import scipy
import random
import itertools
import os.path as osp
import matplotlib
from matplotlib.patches import Polygon
from descartes import PolygonPatch
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import shapely.geometry as sg
import shapely.ops as so
from shapely.strtree import STRtree
from tqdm import tqdm
from parsing.utils.visualization import ImagePlotter
from parsing.utils.comm import to_device
from parsing.utils.logger import CudaTimer
import seaborn as sns
import networkx as nx
import logging
import torch
import math

class DirectionalCycles:
    def __init__(self, edges, junctions):

        edges = edges.cpu().numpy()
        self.junctions = junctions.cpu().numpy()

        G = nx.Graph()
        G.add_nodes_from(range(self.junctions.shape[0]))
        G.add_edges_from(edges)
        di_G = G.to_directed()
        self.cycles = nx.simple_cycles(di_G)


    def generate_planes(self, number_of_planes = math.inf, timeout = math.inf):
        t = time.time()
        planes = []
        generated_plane_set = set()
        while True:
            if time.time() - t > timeout:
                print('Breaking due to timeout')
                break

            if len(planes) >= number_of_planes:
                break

            try:
                cycle_sample = next(self.cycles)
            except StopIteration:
                break

            if len(cycle_sample) < 3:
                # Polygon must have 3 nodes.
                continue

            poly_set = frozenset(cycle_sample)
            if poly_set in generated_plane_set:
                continue

            poly = cycle_sample + [cycle_sample[0]]

            # Verify its a valid polygon
            # t = time.time()
            sg_poly = sg.Polygon(junctions[np.array(poly)])
            # polygon_time += time.time()-t
            if not sg_poly.is_valid:
                continue

            planes.append(poly)
            generated_plane_set.add(poly_set)


        return planes

# Based an algorithm developed by X.Y. Jiang and H. Bunke (1993) in An Optimal Algorithm for Extracting the Regions of a Plane Graph
# see https://www.sciencedirect.com/science/article/abs/pii/016786559390104L
# Implementation inspired by https://github.com/paulgan98/polygon-detection/blob/main/graph.py
class RegionsCPU:
    def __init__(self, edges, junctions):

        edges = edges.cpu().numpy()
        junctions = junctions.cpu().numpy()
        self.junctions = junctions

        G = nx.Graph()
        G.add_nodes_from(range(junctions.shape[0]))
        G.add_edges_from(edges)

        # Remove nodes with degree 1 as they cannot form polygons and the method assumes degree > 1
        G_core = nx.k_core(G, k=2)
        # print(f'N: {G.number_of_nodes()}, 2-core N: {G_core.number_of_nodes()}')

        # Find strongly connected components, as a single region can not be formed across multiple components.
        S = [G.subgraph(c).copy() for c in nx.connected_components(G_core)]
        # print(f'Connected components, {len(S)}')

        # Algorithm requires directed edges
        # Step 1: duplicate each undirected edge to form two directed edges
        self.di_subgraphs = [g.to_directed() for g in S]

    def _make_planar(self, G):
        # edge_idx = np.array(G.edges)
        # lines = np.hstack([self.junctions[edge_idx], np.arange(edge_idx.shape[0]])
        # edge_checked = np.zeros([edge_idx.shape[0]]*2, dtype=np.bool)
        #
        # sg_lines = sg.MultiLineString(lines.tolist())
        # sg_points = sg.MultiPoint(self.junctions.tolist())
        # print('sg_lines',sg_lines)
        # planar_lines = so.unary_union(sg_lines)
        # print('planar_lines',planar_lines)
        # print('boundary',planar_lines.boundary)
        # print('difference',planar_lines.difference(sg_points))
        # for l in planar_lines.geoms:
        #     print(l.boundary.difference(sg_points))
        is_planar, P = nx.check_planarity(G)
        pos = nx.combinatorial_embedding_to_pos(P) if is_planar else None

        # junctions = np.array([pos[i] for i in range(G.number_of_nodes())])

        # plot_result(torch.tensor(junctions), torch.tensor(list(G.edges())), desc = 'make_planar')


        return is_planar, pos





    def _calculate_angles(self, G, pos):
        # Step 2: Complement each directed edge w/ angle theta of (vi, vj)
        # w/ respect to horizontal line passing through vi. Add to list
        # edge_vectors = []
        # for u,v in G.edges():
        #     edge_vectors.append(
        #         (pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
        #     )
        # edge_vectors = np.array(edge_vectors)


        edge_idx = np.array(G.edges)
        edge_vectors = self.junctions[edge_idx[:,1],:] - self.junctions[edge_idx[:,0],:]
        angles_pi_pi = np.arctan2(edge_vectors[:,0],edge_vectors[:,1])
        angles_0_2pi = angles_pi_pi + 2*np.pi*(angles_pi_pi < 0)
        return angles_0_2pi


    def _find_wedges(self, G, pos):
        angles = self._calculate_angles(G, pos)

        # Python sort
        # Step 3: Sort list ascending by index and theta as primary and secondary keys
        t = time.process_time()
        edges_angles = [(e[0],e[1],a) for e,a in zip(G.edges, angles)]
        edges_angles.sort(key=lambda x: (x[0], x[2]))

        # Step 4: Combine consecutive entries in each group into a wedge
        first_ind = 0
        wedges = []
        for i in range(1, len(edges_angles)):
            # Will always occur since each node has degree > 1
            # print('edges angle element',edges_angles[i])
            if edges_angles[i][0] == edges_angles[i-1][0]:
                tup = (edges_angles[i][1], edges_angles[i][0], edges_angles[i-1][1])
                wedges.append(tup)

            # last entry in group, add wedge
            if (i + 1 >= len(edges_angles)) or (edges_angles[i+1][0] != edges_angles[i][0]):
                tup = (edges_angles[first_ind][1], edges_angles[i][0], edges_angles[i][1])
                # tup = (edges_angles[i][0][1], edges_angles[i][0][0], edges_angles[firstInd][0][1])
                wedges.append(tup)
                first_ind = i + 1
        return wedges

    # binary search algorithm for finding next wedge from sorted wedge list
    # using v1 and v2 as primary and secondary search keys
    def _search_wedge(self, wedges, v1, v2):
        l, r = 0, len(wedges)
        m = -1
        while l < r:
            assert m != ((l+r) // 2)
            m = (l+r) // 2
            # if middle element is what we are looking for, return the wedge
            if wedges[m][0] == v1 and wedges[m][1] == v2:
                return wedges[m]
            # else if middle element > v1, shrink right bound
            elif wedges[m][0] > v1: r = m
            # else if middle element < v1, shrink left bound
            elif wedges[m][0] < v1: l = m
            # else v1 matches but v2 doesn't, we adjust bound based on v2
            else:
                if wedges[m][1] > v2: r = m
                else: l = m

        # if we reach here -> element not found, return None
        return None

    def _find_regions(self, wedges):

        def find_unused():
            for k, v in used.items():
                if v == 0:
                    return k
            return None

        # Step 5: Sort wedge list using vi and vj as primary and secondary key
        wedges.sort(key=lambda x: (x[0], x[1]))

        # Step 6: Mark all wedges as unused
        used = {w:0 for w in wedges}

        # Step 7: Find unused wedge W0 = (v1, v2, v3)
        w0 = find_unused() # initial wedge: w0
        used[w0] = 1 # set w0 to used
        next_first, next_second = w0[1], w0[2]
        region_wedge_list = [w0]

        # Step 8: Search for next wedge wi = (v2, v3, vn)
        regions = []
        while True:
            wi = self._search_wedge(wedges, next_first, next_second) # O(logn) binary search
            used[wi] = 1 # set wi to used
            next_first, next_second = wi[1], wi[2]
            region_wedge_list.append(wi)

            # keep searching for next wedge until w(i+1) and w(1) are contiguous
            if (next_first != w0[0]) and (next_second != w0[1]): continue
            else: # contiguous region found
                region = [x[1] for x in region_wedge_list]
                # if region contains no repeating elements
                if len(region) > 2 and len(region) == len(set(region)):

                    regions.append(region) # store region

                region_wedge_list = [] # clear list

                # Back to Step 7: Find next unused wedge
                w0 = find_unused() # initial wedge: w0
                if not w0: break
                used[w0] = 1 # set w0 to used
                next_first, next_second = w0[1], w0[2]
                region_wedge_list.append(w0)

        return regions


    def solve(self, number_of_planes = math.inf, timeout = math.inf):
        planes = []
        t = time.time()
        for G in self.di_subgraphs:
            if time.time() - t > timeout:
                print('Breaking due to timeout')
                break
            # is_planar, pos = self._make_planar(G)
            # assert is_planar
            pos=None
            wedges = self._find_wedges(G, pos)
            regions = self._find_regions(wedges)
            # Temporary
            planes += regions
        return planes



def polygonize(edges,junctions, number_of_planes =math.inf):
    edges = edges.cpu().numpy()
    junctions = junctions.cpu().numpy()
    #Encode index as z values
    # junctions = np.hstack([junctions, np.arange(junctions.shape[0])[:,None]])
    lines = junctions[edges,:]
    sg_lines = sg.MultiLineString(lines.tolist())
    polygons = list(so.polygonize(sg_lines))
    # adj_matrix = np.zeros([edges.shape[0],len(polygons)], dtype=np.bool)

    line_tree = STRtree(sg_lines)
    # keep_polys = []
    # print('nbr polys',len(polygons))
    # for sg_p in polygons:
    #     poly_boundary = sg_p.boundary
    #     print('poly_boundary',poly_boundary)
    #     print('poly_boundary.coords',poly_boundary.coords)
    #     nearby_lines = sg.MultiLineString(line_tree.query(sg_p))
    #     print('nearby_lines',nearby_lines)
    #
    #     # Check if the endpoints of the polygon exists within all valid enpoints
    #     # If not, we are not interested in the polygon
    #     if not nearby_lines.boundary.contains(poly_boundary.boundary):
    #         continue
    #
    #     # Store polygon and figure out the adjancency matrix
    #     # keep_polys.append(sg_p)
    #     poly_idx = [p[2] for p in poly_boundary.boundary]
    #     keep_polys.append(poly_idx)
    #     # for pl in sg_p.boundary.geoms:
    #     #     for l in nearby_lines:
    #     #         if pl.equals(l):
    #     #             poly_idx.append()



    return polygons

class CycleBasisGeneration:
    def __init__(self, edges, junctions, enable_timing = False):

        self.timer = CudaTimer(active=enable_timing)
        self.time_info = {}
        self.timer.start_timer()

        try:
            self.device = junctions.device
            edges = edges.detach().cpu().numpy()
            junctions = junctions.detach().cpu().numpy()
        except AttributeError:
            self.device = 'cpu'
        self.junctions = junctions
        self.edges = edges

        G = nx.Graph()
        G.add_nodes_from(range(junctions.shape[0]))
        G.add_edges_from([(e[0], e[1], {'idx':i}) for i, e in enumerate(edges)])
        G.remove_edges_from(nx.selfloop_edges(G))

        # Remove nodes with degree 1 as they cannot form polygons and the method assumes degree > 1
        G_core = nx.k_core(G, k=2)
        # print(f'N: {G.number_of_nodes()}, 2-core N: {G_core.number_of_nodes()}')

        # Find strongly connected components, as a single region can not be formed across multiple components.
        self.graph = G_core
        self.time_info['time_plane_generation_init'] = self.timer.end_timer()
        # self.subgraphs = [G]
        # assert len(self.subgraphs) == 1
        # Not yet implemented for subgraphs
        # print(f'Connected components, {len(S)}')


    def get_timings(self):
        return self.time_info

    # Takes grapg of cycles with edges describing if they overlap.
    # Based on https://stackoverflow.com/questions/15658245/efficiently-find-all-connected-induced-subgraphs
    def _generate_cycles(self, cycle_graph, random_pop = False):
        minimum_number_of_vertices = 1
        set_pop = self._random_set_pop if random_pop else self._standard_set_pop
        def GenerateConnectedSubgraphs(verticesNotYetConsidered, subsetSoFar, neighbors):
            if len(subsetSoFar) == 0:
                candidates = verticesNotYetConsidered
            else:
                candidates = verticesNotYetConsidered & neighbors
            if len(candidates)  == 0:
                if len(subsetSoFar) >= minimum_number_of_vertices:
                    yield list(subsetSoFar)
            else:
                # v = candidates.pop()
                v = set_pop(candidates)
                yield from GenerateConnectedSubgraphs(verticesNotYetConsidered - {v},
                                           subsetSoFar,
                                           neighbors)
                yield from GenerateConnectedSubgraphs(verticesNotYetConsidered - {v},
                                           subsetSoFar | {v},
                                           neighbors | set(cycle_graph.neighbors(v)))


        yield from GenerateConnectedSubgraphs(set(cycle_graph.nodes), set(), set())

    def _random_set_pop(self, candidates):
        v = random.choice(list(candidates))
        candidates.remove(v)
        return v

    def _standard_set_pop(self, candidates):
        return candidates.pop()

    def _join_cycle_edges(self, cycle):
        candidates = [set(c) for c in cycle]
        stack = [candidates.pop().pop()]
        while candidates:
            last_node = stack[-1]
            success = False
            for i,c in enumerate(candidates):
                if last_node in c:
                    c.discard(last_node)
                    stack.append(c.pop())
                    del candidates[i]
                    success = True
                    break
            if not success:
                break
        if success:
            stack.append(stack[0])
            return stack
        else:
            return None

    def find_fundamental_cycle_graphs(self, graph = None):
        input_graph = graph if graph else self.graph
        subgraphs = [input_graph.subgraph(c).copy() for c in nx.connected_components(input_graph)]
        nested_c_basis = [nx.cycle_basis(graph) for graph in subgraphs]
        c_basis = list(itertools.chain.from_iterable(nested_c_basis))

        edges = input_graph.edges

        # Construct edge logical array
        c_basis = [c + [c[0]] for c in c_basis]

        self.cycle_logical = np.zeros([len(c_basis), self.edges.shape[0]], dtype=bool)
        for i, b in enumerate(c_basis):
            for u,v in zip(b[:-1],b[1:]):

                edge_idx = edges[u,v]['idx']
                # print('Edge', u,v, 'Idx', edge_idx)
                self.cycle_logical[i,edge_idx] = True

        adj_matrix = np.zeros([len(c_basis)]*2, dtype=bool)
        for i in range(self.cycle_logical.shape[0]):
            for j in range(i, self.cycle_logical.shape[0]):
                adj_matrix[i,j] = np.any(np.logical_and(self.cycle_logical[i], self.cycle_logical[j]))

        cycle_relation_graph = nx.Graph(adj_matrix)
        self.cycle_subgraphs = [cycle_relation_graph.subgraph(c).copy() for c in nx.connected_components(cycle_relation_graph)]

    def generate_valid_permutations(self, number_of_planes = math.inf):
        generated_cycles_nodes = []
        generated_cycles_edges = []
        generated_cycles_set = set()
        for cycle_graph in self.cycle_subgraphs:
            for tuple_idx in self._generate_cycles(cycle_graph, random_pop = number_of_planes < math.inf):

                if time.time() > self.end_time:
                    print('Breaking due to timeout')
                    return generated_cycles_nodes, generated_cycles_edges

                if len(generated_cycles_nodes) >= number_of_planes:
                    return generated_cycles_nodes, generated_cycles_edges

                tuple_set = frozenset(tuple_idx)
                if tuple_set in generated_cycles_set:
                    continue

                self._add_polygon_from_fundamental_cycles(tuple_idx, generated_cycles_nodes, generated_cycles_edges)
                generated_cycles_set.add(tuple_set)


        return generated_cycles_nodes, generated_cycles_edges

    def _make_result(self, plane_nodes, plane_edges, return_edge_idx = False):
        if return_edge_idx:
            return plane_nodes, plane_edges
        else:
            return plane_nodes


    def generate_random_permutations_sample_edges(self, number_of_planes = 10, edges_to_sample = 20, timeout = math.inf, return_edge_idx = False):
        self.end_time = time.time() + timeout
        generated_cycles_set = set()
        generated_cycles_nodes = []
        generated_cycles_edges = []
        max_iter = 100*number_of_planes

        if edges_to_sample >= self.graph.number_of_edges():
            return self.generate_planes(number_of_planes = number_of_planes, timeout = timeout, return_edge_idx = return_edge_idx)


        for i in range(max_iter):

            # Sample edges
            edge_samples = random.sample(self.graph.edges(), edges_to_sample)
            subgraph = self.graph.subgraph(edge_samples)

            # Find components
            self.find_fundamental_cycle_graphs(subgraph)

            # Generate all permutations
            nbr_to_sample = math.ceil((number_of_planes - len(generated_cycles_nodes))*1.5)
            planes_nodes, planes_edges = self.generate_valid_permutations(number_of_planes = nbr_to_sample)

            for n, e in zip(planes_nodes, planes_edges):
                n_set = frozenset(n)
                if not (n in generated_cycles_set):
                    generated_cycles_nodes.append(n)
                    generated_cycles_edges.append(e)
                    if len(generated_cycles_nodes) >= number_of_planes:
                        return self._make_result(generated_cycles_nodes, generated_cycles_edges, return_edge_idx = return_edge_idx)

            if time.time() > self.end_time:
                print('Breaking due to timeout')
                break


        return self._make_result(generated_cycles_nodes, generated_cycles_edges, return_edge_idx = return_edge_idx)




    def generate_random_permutations_cuts(self, number_of_planes = 10):
        generated_cycles_nodes = []
        generated_cycles_edges = []
        generated_cycles_set = set()
        max_iter = 100*number_of_planes
        all_max_degree = []
        for g in self.cycle_subgraphs:
            all_max_degree.append(np.max([d for (_,d) in g.degree()]))

        all_cuts_set = [set() for _ in self.cycle_subgraphs]

        for i in range(max_iter):
            if time.time() > self.end_time:
                print('Breaking due to timeout')
                break
            # Sample graph
            graph_idx  = random.randrange(len(self.cycle_subgraphs))
            cycle_graph = self.cycle_subgraphs[graph_idx]
            max_degree = all_max_degree[graph_idx]
            cuts_set = all_cuts_set[graph_idx]

            # Sample cut
            nbr_cuts = random.randint(0, max_degree)
            cut_idx = random.sample(range(cycle_graph.number_of_edges()), k=nbr_cuts)
            cut_idx_set = frozenset(cut_idx)
            if cut_idx_set not in all_cuts_set:
                # Make cut if not done before
                cycle_graph = cycle_graph.copy()
                edges = list(cycle_graph.edges)
                cycle_graph.remove_edges_from([edges[eidx] for eidx in cut_idx])
                all_paths = nx.connected_components(cycle_graph)

                for tuple_idx in all_paths:
                    # Add path if new
                    assert tuple_idx
                    tuple_idx_set = frozenset(tuple_idx)
                    tuple_idx = list(tuple_idx)
                    if not (tuple_idx_set in generated_cycles_set):
                        generated_cycles_set.add(tuple_idx_set)
                        self._add_polygon_from_fundamental_cycles(tuple_idx, generated_cycles_nodes, generated_cycles_edges)

            if len(generated_cycles_nodes) >= number_of_planes:
                break

        return generated_cycles_nodes, generated_cycles_edges

    def generate_random_permutations_paths(self, number_of_planes = 10, method = 'shortest'):
        generated_cycles_nodes = []
        generated_cycles_edges = []
        generated_cycles_set = set()
        max_iter = 100*number_of_planes
        max_nbr_paths = 10
        if method == 'shortest':
            path_func = lambda cycle_graph, start_node, end_node: [nx.shortest_path(cycle_graph, start_node, end_node)]
        elif method == 'all_simple':
            path_func = lambda cycle_graph, start_node, end_node: nx.all_simple_paths(cycle_graph, start_node, end_node)
        else:
            raise NotImplementedError(f'Method {method} not implemented')


        for i in range(max_iter):

            cycle_graph = random.choice(self.cycle_subgraphs)
            nodes = list(cycle_graph)
            start_node, end_node = random.choices(nodes, k=2)
            # print(start_node, end_node)

            all_paths = path_func(cycle_graph, start_node, end_node)
            for i, tuple_idx in enumerate(all_paths):
                tuple_idx_set = frozenset(tuple_idx)
                if not (tuple_idx_set in generated_cycles_set):
                    generated_cycles_set.add(tuple_idx_set)
                    self._add_polygon_from_fundamental_cycles(tuple_idx, generated_cycles_nodes, generated_cycles_edges)

                if len(generated_cycles_nodes) >= number_of_planes:
                    return generated_cycles_nodes, generated_cycles_edges

                if time.time() > self.end_time:
                    print('Breaking due to timeout')
                    return generated_cycles_nodes, generated_cycles_edges

                # if i >= max_nbr_paths:
                #     break

        return generated_cycles_nodes, generated_cycles_edges

    def _add_polygon_from_fundamental_cycles(self, cycle_idx, generated_cycles_nodes, generated_cycles_edges):
        if len(cycle_idx) > 1:
            tuple_idx = np.array(cycle_idx)
            xor_mask = np.logical_xor.reduce(self.cycle_logical[tuple_idx])
        else:
            # Special case for a single cycle node
            xor_mask = self.cycle_logical[cycle_idx]

        # Require more than two edges since we want a polygon
        if not np.sum(xor_mask) > 2:
            return

        cycle_edge_idx = np.flatnonzero(xor_mask)
        cycle_node_idx = self._join_cycle_edges(self.edges[cycle_edge_idx].tolist())

        if cycle_node_idx and sg.Polygon(self.junctions[cycle_node_idx]).is_valid:
            generated_cycles_nodes.append(torch.tensor(cycle_node_idx).to(self.device))
            generated_cycles_edges.append(torch.tensor(cycle_edge_idx).to(self.device))

    def _get_upper_bound_cycles(self):
        nbr_possible_planes = 0
        for cg in self.cycle_subgraphs:
            nbr_possible_planes += 2**cg.number_of_nodes() - 1

        return nbr_possible_planes

    def generate_planes(self, number_of_planes = math.inf, timeout = math.inf, return_edge_idx = False):
        self.end_time = time.time() + timeout

        self.timer.start_timer()
        self.find_fundamental_cycle_graphs()
        self.time_info['time_plane_generation_fundamental_cycles'] = self.timer.end_timer()

        self.timer.start_timer()
        # Some randomness in case we are sampling
        random.shuffle(self.cycle_subgraphs)
        planes_nodes, planes_edges = self.generate_valid_permutations(number_of_planes = number_of_planes)
        self.time_info['time_plane_generation_generate_perm'] = self.timer.end_timer()
        if return_edge_idx:
            return planes_nodes, planes_edges
        else:
            return planes_nodes

    def generate_random_planes(self, number_of_planes = 10, method = 'shortest', timeout = math.inf, return_edge_idx = False):
        assert np.isfinite(number_of_planes)
        self.end_time = time.time() + timeout

        self.timer.start_timer()
        self.find_fundamental_cycle_graphs()
        self.time_info['time_plane_generation_fundamental_cycles'] = self.timer.end_timer()


        self.timer.start_timer()
        # Figure out if traversing the cycle tree is smarter than sampling
        nbr_cycles_ub = self._get_upper_bound_cycles()
        if nbr_cycles_ub < 2*number_of_planes:
            planes_nodes, planes_edges = self.generate_valid_permutations(number_of_planes = number_of_planes)
        # Otherwise choose algorithm according to input.
        elif method == 'shortest':
            planes_nodes, planes_edges = self.generate_random_permutations_paths(number_of_planes = number_of_planes, method = method)
        elif method == 'all_simple':
            planes_nodes, planes_edges = self.generate_random_permutations_paths(number_of_planes = number_of_planes, method = method)
        elif method == 'cuts':
            planes_nodes, planes_edges = self.generate_random_permutations_cuts(number_of_planes = number_of_planes)
        else:
            raise NotImplementedError(f'{method} not implemented')

        self.time_info['time_plane_generation_generate_perm'] = self.timer.end_timer()

        if return_edge_idx:
            return planes_nodes, planes_edges
        else:
            return planes_nodes






def prepare_data(ann, nbr_edges = None):
    #Take detections as both positive and negative edges
    junctions = ann['junctions']

    edges = torch.cat((ann['edges_positive'],ann['edges_negative']), dim=0)
    if nbr_edges and nbr_edges < edges.size(0):
        rand_idx = torch.randperm(edges.size(0), device=edges.device)[:nbr_edges]
        edges = edges[rand_idx]

    return junctions, edges

# Make sure that we at least found all positive planes
def verify(ann, planes):
    for p_gt in ann['planes']:
        p_gt = set(p_gt['junction_idx'])
        exist = False
        for p in planes:
            if p_gt == set(p):
                exist = True
                break
        if not exist:
            return False
    return True

def generate_simple_geometry():
    junctions = torch.tensor([(0,0),(1,0),(1,1),(0,1),(2,0)])
    edges = torch.tensor([
        (0,1),(1,2),(2,3),(3,0), #Square
        (1,4),(4,2), # Triangle right
        (0,2),(3,1) # split square x2
    ], dtype=torch.long)
    regions = [
        (0,1,2,0), #Square right
        (0,2,3,0), #Square left
        (0,1,3,0), #Square left 2
        (1,2,3,1), #Square right 2
        (1,4,2,1) #Triangle right
    ]
    return junctions,edges,regions

def plot_result(junctions, edges, planes = [], desc = 'simple_geometry'):

    def _plot_lines():
        plt.plot(junctions[:,0],junctions[:,1], 'o')
        for l in lines:
            plt.plot(l[:,0],l[:,1])

    edges = edges.cpu().numpy()
    junctions = junctions.cpu().numpy()
    lines = junctions[edges,:]
    plt.figure()
    nbr_plots = 1+len(planes)
    row = int(np.ceil(np.sqrt(nbr_plots/1.6)))
    col = int(np.ceil(nbr_plots/row))
    plt.subplot(row,col,1)
    _plot_lines()


    colors = sns.color_palette("husl", len(planes))

    for i, (p,c) in enumerate(zip(planes, colors)):
        plt.subplot(row,col,i+2)
        ax = plt.gca()
        _plot_lines()
        if isinstance(p, sg.Polygon):
            ax.add_patch(PolygonPatch(p,facecolor=c,edgecolor = c, alpha=0.5))
        else:
            poly = junctions[np.array(p,dtype=np.int)]
            ax.add_patch(Polygon(poly,facecolor=c,edgecolor = c, alpha=0.5, closed=True))


    plt.savefig(f'/host_home/plots/plane_generation/{desc}.svg')


def test_randomness(f,edges,junctions, nbr_planes, nbr_iterations):
    counts = {}
    for i in range(nbr_iterations):
        planes = f(edges, junctions)
        for p in planes:
            p_set = frozenset(p)
            if p_set in counts:
                counts[p_set] += 1
            else:
                counts[p_set] = 1

    return counts

if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Benchmark plane generation methods', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--nbr-samples', type=int, default = 1, help='Number of samples from dataset')
    parser.add_argument('-e', '--nbr-edges', type=int, default = None, help='Number of edges to sample')
    parser.add_argument('-p', '--nbr-planes', type=int, default = math.inf, help='Number of planes to generate')
    parser.add_argument('-t', '--timeout', type=float, default = math.inf, help='Maximum time allowed in seconds')
    parser.add_argument('-r', '--randomness', type=int, default = 0, help='Test randomness for this amount of iterations')

    args = parser.parse_args()
    cfg.merge_from_file(osp.join(script_path, '..', '..', 'config-files', 'Pred-simple-plane-S3D.yaml'))
    cfg.DATASETS.TEST = ("structured3D_perspective_planes_test_mini",)
    device = cfg.MODEL.DEVICE

    #Suppress shapely logger
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)



    functions = {#'Regions': lambda edges, junctions: RegionsCPU(edges, junctions).solve(timeout = args.timeout),
                 'Cycle Basis': lambda edges, junctions: CycleBasisGeneration(edges, junctions).generate_planes(args.nbr_planes, timeout = args.timeout),
                 #'Polygonize': polygonize,
                 #'Directional Cycles': lambda edges, junctions: DirectionalCycles(edges, junctions).generate_planes(args.nbr_planes, timeout = args.timeout)
                 }
    if np.isfinite(args.nbr_planes):
        # functions['Cycle Basis - Randomized - Simple'] =  lambda edges, junctions: CycleBasisGeneration(edges, junctions).generate_random_planes(args.nbr_planes, method= 'all_simple', timeout = args.timeout)
        functions['Cycle Basis - Randomized - Shortest'] =  lambda edges, junctions: CycleBasisGeneration(edges, junctions).generate_random_planes(args.nbr_planes, method= 'shortest', timeout = args.timeout)
        # functions['Cycle Basis - Randomized - Cuts'] =  lambda edges, junctions: CycleBasisGeneration(edges, junctions).generate_random_planes(args.nbr_planes, method= 'cuts', timeout = args.timeout)
        # functions['Cycle Basis - Randomized - Sample'] =  lambda edges, junctions: CycleBasisGeneration(edges, junctions).generate_random_permutations_sample_edges(args.nbr_planes, timeout = args.timeout)

        # del functions['Polygonize']
        # del functions['Regions']
    timings = {f:[] for f in functions}
    plane_count = {f:0 for f in functions}
    unique_counts = {f:{} for f in functions}

    # junctions, edges,regions = generate_simple_geometry()
    # plot_result(junctions, edges,regions, desc='gt')
    #
    # for name, f in functions.items():
    #     t = time.process_time()
    #     planes = f(edges, junctions)
    #     timings[name].append(time.process_time() - t)
    #     plot_result(junctions, edges,planes, desc=name)
    #     print(f'{name} found {len(planes)} planes')
    #     print(planes)

    datasets = build_test_dataset(cfg)
    for name, dataset in datasets:
        for i, (images, annotations) in enumerate(tqdm(dataset)):

            if i >= args.nbr_samples:
                break

            annotations = to_device(annotations, device)
            ann = annotations[0]
            junctions, edges = prepare_data(ann, args.nbr_edges)
            print(f'Number of junctions {junctions.size(0)}')
            print(f'Number of edges {edges.size(0)}')

            for name, f in functions.items():
                print('Running', name)
                t = time.process_time()
                planes = f(edges, junctions)
                timings[name].append(time.process_time() - t)
                plane_count[name] += len(planes)

                if i == 1 and np.isfinite(args.nbr_planes) and args.randomness > 0:
                    unique_counts[name] = test_randomness(f,edges,junctions, args.nbr_planes, args.randomness)

    print(f'Generate {args.nbr_planes} planes over {args.nbr_samples} sample images')
    all_durations = []
    all_plane_counts = []
    for method, durations in timings.items():
        print('=====================')
        print(method)
        print('---------------------')
        print('Median:', np.median(durations))
        print('Min:', np.min(durations))
        print('Max:', np.max(durations))
        print('# Planes:', plane_count[method])
        print('')
        all_durations.append(durations)
        all_plane_counts.append(plane_count[method])

    plt.figure()
    plt.subplot(2,1,1)
    plt.bar(range(len(all_plane_counts)),all_plane_counts)
    plt.ylabel('# planes generated')
    plt.gca().tick_params(labelbottom=False)
    plt.title(f'Generate {args.nbr_planes} planes for {args.nbr_samples} sample images')


    plt.subplot(2,1,2)
    plt.boxplot(all_durations, labels = timings.keys(), showfliers=False)
    plt.xticks(rotation = 30, ha='right')
    plt.yscale('log')
    plt.ylabel('Time [seconds]')


    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_generation/summary_{args.nbr_planes}planes_{args.nbr_samples}samples_{args.nbr_edges}edges.png')
    plt.close()

    # plot randomness
    plt.figure()
    bar_per_tick = len(unique_counts)+1
    all_polygons = set()
    for f,p in unique_counts.items():
        all_polygons.update(p)
    poly2idx = {p:idx for idx,p in enumerate(all_polygons)}
    for i, (f, counts) in enumerate(unique_counts.items()):
        nbr_instances = []
        idx = []
        for p, count in counts.items():
            nbr_instances.append(count)
            idx.append(poly2idx[p])
        idx = np.array(idx)*bar_per_tick + i

        plt.bar(idx,nbr_instances, label=f)
    plt.ylabel('# observations')
    plt.xlabel('Polygon instance')
    ax = plt.gca()
    all_ticks = np.arange(len(all_polygons))*bar_per_tick
    ax.set_xticks(all_ticks, minor = True)
    ax.set_xticks(all_ticks[::5], minor = False)
    ax.set_xticklabels(np.arange(0,len(all_polygons),5))

    # plt.gca().tick_params(labelbottom=False)
    plt.title(f'Observed polygons when randomizing {args.nbr_planes} planes for {args.randomness} iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_generation/randomness_{args.nbr_planes}planes_{args.randomness}iterations_{args.nbr_edges}edges.png')
    plt.close()


    plt.figure()
    plt.suptitle(f'Observed polygons when randomizing {args.nbr_planes} planes for {args.randomness} iterations')
    for i, (f, counts) in enumerate(unique_counts.items()):
        plt.subplot(3,2,i+1)
        nbr_instances = []
        all_idx = []

        for p, idx in poly2idx.items():
            nbr_obs = counts.get(p,0)
            nbr_instances.append(nbr_obs)
            all_idx.append(idx)
            if nbr_obs > args.randomness:
                print(f'Method {f} observed {p} too many times -> {nbr_obs} times')
        idx = np.array(all_idx)
        nbr_instances = np.array(nbr_instances)
        s_idx = np.argsort(idx)
        idx = idx[s_idx]
        nbr_instances = nbr_instances[s_idx]


        # plt.plot(idx,nbr_instances, label=f, marker=markers[i])
        plt.plot(idx,nbr_instances, marker='.', linestyle='none')
        plt.title(f, fontdict={'fontsize': 10})
        plt.ylabel('# observations')
        plt.xlabel('Polygon instance')

    # plt.gca().tick_params(labelbottom=False)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_generation/randomness_{args.nbr_planes}planes_{args.randomness}iterations_{args.nbr_edges}edges_lineplot.png')
    plt.close()
