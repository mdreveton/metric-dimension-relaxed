#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:45:19 2024

@author: dreveton
"""

import numpy as np
from collections import Counter
#import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import rustworkx as rx
import igraph as ig

import time
from itertools import combinations


def relaxedResolvingSet( g, k, print_detailed_running_time = False ):
    """
    Parameters
    ----------
    graph : TYPE
        igraph Graph.
    k : integer
        distance cutoff of resolved pairs.

    Returns
    -------
    resolving_set : set
        set of vertices that forms a k-relaxed resolving set.
    """
    resolving_set = set ( )
    
    #print("Start distance computations")
    start = time.time()
    #g = rx.networkx_converter( graph )
    #distances = rx.all_pairs_dijkstra_path_lengths( g , edge_cost_fn = lambda edge: 1 )
    #distances = dict(nx.all_pairs_shortest_path_length(graph))
    #g = ig.Graph.from_networkx(graph)
    distances = g.distances( )
    end = time.time()
    if print_detailed_running_time:
        print( 'Distances computations took : ' + str( end - start ) + ' seconds' )


    #print("Start subset generations")
    start = time.time()
    n = g.vcount()
    U = set( (u,v) for u in range( n ) for v in range( u+1, n ) if distances[u][v] > k )
    S = dict( )
    for vertex in range( n ):
        S_node = [ ]
        #distances_to_nodes = set( distances[ vertex ].values() )
        distances_to_nodes = set( distances[ vertex ] )
        other_vertices = [ u for u in range( n ) if u != vertex ]
        for dummy in distances_to_nodes:
            nodes_at_distances_dummy = [ u for u in other_vertices if distances[ vertex ][ u ] == dummy ]
            #all_possible_pairs = [ ( nodes_at_distances_dummy[i], nodes_at_distances_dummy[j] ) for i in range(len(nodes_at_distances_dummy))  for j in range(i+1, len(nodes_at_distances_dummy) ) ] 
            
            #all_possible_pairs = list( combinations(nodes_at_distances_dummy, 2) ) 
            
            for pair in combinations( nodes_at_distances_dummy, 2 ): #loop over all pair of vertices at sane distance of u
                if distances[ pair[0] ][ pair[1] ] > k: #Only add pair of vertices that are at distance more than k
                    S_node.append( pair )
                    #S_node.append( ( min(pair), max(pair) ) ) #To ensure list of edges is always the one with smallest value first.
        S[ vertex ] = set( S_node ) 
    end = time.time()
    if print_detailed_running_time:
        print( 'Subset generation took : ' + str( end - start ) + ' seconds' )
    
    #print( "Start greedy search of resolving vertices" )
    start = time.time( )
    while len( U ) > 0:
        new_resolving_node = np.argmin( [ len( S[ i ].intersection( U ) ) for i in range( n ) ] )
        resolving_set.add( new_resolving_node )
        resolved_pairs = U - S[ new_resolving_node ] 
        U = U.difference( resolved_pairs )
    end = time.time()
    if print_detailed_running_time:
        print( 'Greedy search of resolving vertices took : ' + str( end - start ) + ' seconds' )

    return resolving_set


# ##############################################
# ADDITIONAL FUNCTION
# ##############################################


def md_tree(Graph, k):
    """
    Old function to compute the metric dimension of trees
    G should be a networks graph.
    """
    G = Graph.copy()
    for i in range( int( k/2 ) ):
        degree_one_nodes = [node for node in G.nodes() if G.degree[node] == 1]

        # Step 2: Remove vertices of degree 1
        G.remove_nodes_from(degree_one_nodes)
        
    while(True):
        degree_two_nodes = [node for node in G.nodes() if G.degree[node] == 2]
        if (len(degree_two_nodes) == 0):
            break
        neighbors = list(G.neighbors(degree_two_nodes[0]))
        G.remove_node(degree_two_nodes[0])
        G.add_edge(neighbors[0], neighbors[1])
    
    degree_one_nodes = [node for node in G.nodes() if G.degree[node] == 1]
    leaf = len(degree_one_nodes)
    neighbors = set()
    for v in degree_one_nodes:
        neighbors.add([n for n in G.neighbors(v)][0])
    n_leaf = len(neighbors)
    
    return leaf - n_leaf
