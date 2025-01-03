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



def getIdentificationVectors( S, g = None, distances = None ):
    """
    Given a set of vertices and either the graph or the matrix of distances,
    this function returns the identification vectors of all vertices
    (that is, the element u of the returned list is the vector of the distances between vertex u and the source vertices in S)
    
    equivalent classes
    (recall that an equivalent class is a set of vertices with same identification vector)
    
    Parameters
    ----------
    S : set
        set of vertices.
    g : igraph Graph
        DESCRIPTION. The default is None.
    distances : matrix
        DESCRIPTION. The default is None.

    Returns
    -------
    nonResolvedSetsOfVertices : set of sets
        Set with the equivalent classes.
    """
    
    if distances == None and g == None:
        raise TypeError( 'You need to provide at least either the graph of the matrix of distances' )
    
    if distances == None:
        distances = g.distances( )
        n = g.vcount( )
    if g == None:
        n = len( distances )
    
    identification_vectors = [ ]
    
    for u in range( n ):
        vector_profile_u = [ distances[ u ][ source ] for source in S ]             
        identification_vectors.append( vector_profile_u )
        
    return identification_vectors



def getEquivalentClasses( identification_vectors ):
    
    equivalent_classes = dict( )
    unique_identification_vectors = [ ]
    n = len( identification_vectors )
    
    for u in range( n ):
        vector_profile_u = identification_vectors[ u ]
        if vector_profile_u not in unique_identification_vectors:
            unique_identification_vectors.append( vector_profile_u )
            equivalent_classes[ str( vector_profile_u ) ] = [ u ]
        else:
            equivalent_classes[ str( vector_profile_u ) ].append( u )
    
    return equivalent_classes



def getNonResolvedEquivalentClasses( equivalent_classes ):
    non_resolved_equivalent_classes = [ ]
    
    for equivalent_class in equivalent_classes.values( ):
        
        if len( equivalent_class ) >= 2:
            non_resolved_equivalent_classes.append( equivalent_class )
    
    return non_resolved_equivalent_classes



def getNonResolvedVertices( equivalent_classes ):
    non_resolved_vertices = [ ]
    
    for equivalent_class in equivalent_classes.values( ):
        
        if len( equivalent_class ) >= 2:
            non_resolved_vertices += equivalent_class
    
    return non_resolved_vertices
   
 

# ##############################################
# Helpers functions
# ##############################################


def is_resolving_set( G, nodes_in_subset, distances ):
    """Given a graph and the matrix with all shortest paths, 
    test if a set of node resolve the graph

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        distances (dict): Dictionary or matrix whose element (u,v) is the distance between vertices u and v

    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """

    dist = {}
    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes():
        distances_subset = {}
        for n in nodes_in_subset:
            distances_subset[n] = distances[node][n]
        dist[node] = tuple( distances_subset.values( ) )

    # Check if the vector with all the shortest paths to nodes nodes_in_subset is different for each node in G
    # If it is, then the set of nodes nodes_in_subset resolves the graph G, otherwise it does not
    res = len( ( set( list(dist.values( ) ) ) ) ) == G.number_of_nodes( )   

    return res


def is_resolving_set_d(G, nodes_in_subset, distances, k ):
    """Given a graph and the matrix with all shortest paths, 
    test if a set of node resolve the graph

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        distances (dict): Dictionary or matrix whose element (u,v) is the distance between vertices u and v
        k (int): threshold
        
    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """

    dist = {}

    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes():
        distances_subset = {}
        for n in nodes_in_subset:
            distances_subset[n] = distances[node][n]
        dist[node] = tuple(distances_subset.values())

    # Check if the vector with all the shortest paths to nodes nodes_in_subset is different for each node in G
    # If it is, then the set of nodes nodes_in_subset resolves the graph G, otherwise it does not
    res = True
    for iii in G.nodes():
        for jjj in G.nodes():
            if iii == jjj:
                _ = 1
            else:  
                if dist[iii] == dist[jjj] and distances[iii][jjj] > k:
                    return False

    return res


def md_tree(Graph, k):
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



def set_resolved( G, nodes_in_subset ):
    """Given a graph and the matrix with all shortest paths, 
    output the set of nodes that are resolved by the given subset

    Args:
        G (Graph): A graph
        nodes_in_subset (set): A set of nodes
        length (dict): Dictionary with all shortest path

    Returns:
        bool: true if the set of nodes resolves the graph and false otherwise
    """
    length = dict(nx.all_pairs_shortest_path_length(G))
    dist = {}

    # For each node in G, compute the shortest path lengths to the nodes in nodes_in_subset
    for node in G.nodes( ):
        distances_subset = { }
        for n in nodes_in_subset:
            distances_subset[n] = length[node][n]
        dist[node] = tuple(distances_subset.values())  

    return get_unique_keys(dist)

def get_unique_keys(dictionary):
    value_counter = Counter(dictionary.values())
    unique_keys = [key for key, value in dictionary.items() if value_counter[value] == 1]
    return unique_keys