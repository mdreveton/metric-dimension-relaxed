#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:03:27 2024
"""


import numpy as np
import networkx as nx
import igraph as ig
import scipy as sp
import time
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt 


import relaxedMetricDimension as rmd
import utils as utils
#from experiments import plotFigure
import experiments as experiments


synthetic_graph_implemented = [ 'BA', 'GW', 'CM', 'RGG', 'kNN' ]
real_graph_implemented = [ 'authors', 'roads', 'powergrid', 'copenhagen-calls', 'copenhagen-friends', 'yeast' ]


"""

# =============================================================================
# TWO-STEP LOCALIZATION ON SYNTHETIC NETWORKS
# =============================================================================


graph_type = 'RGG'

n_range = [ 200, 600, 1000 ]
nAverage = 2
#k_range = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16 ]
k_range = [ 0, 1, 2, 3, 4, 5, 6 ]
#k_range = [ 0, 2, 4, 6, 8 ]

if graph_type == 'RGG':
    k_range = [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28 ]
    k_range = [ i for i in range( 27 ) ]
elif graph_type == 'kNN':
    k_range = [ i for i in range( 30 ) ]
elif graph_type == 'BA':
    k_range = [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28 ]
elif graph_type == 'CM':
    k_range = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ]


average_number_sensors, std_number_sensors = syntheticGraphsSequentialGame( graph_type , n_range = n_range, k_range = k_range, nAverage = nAverage )

if graph_type == 'RGG':
    fileName = 'RGG_1,5'
elif graph_type == 'kNN':
    fileName = 'kNN_2'
else:
    fileName = graph_type
savefig = False
experiments.plotFigure( k_range, average_number_sensors, std_number_sensors, ylabel = "$q^*$", methods = n_range, savefig = savefig, fileName = 'twoStepGame_' + fileName + '_nAverage_' + str(nAverage) + '.pdf' )



# =============================================================================
# TWO-STEP LOCALIZATION ON REAL NETWORKS
# =============================================================================


graph_name = 'authors'
if graph_name == 'authors':
    k_range = [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 ]
elif graph_name == 'copenhagen-calls':
    k_range = [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 ]
elif graph_name == 'copenhagen-friends':
    k_range = [ 0, 1, 2, 3, 4, 5, 6, 7 ]

G = experiments.getRealGraph( graph_name )
g = ig.Graph.from_networkx( G )
number_sensors, numberFixedCameras, numberExtraCameras = sequentialGame( g, k_range, print_progress = True )

savefig = False
#experiments.plotFigure( k_range, number_sensors, ylabel = "$q^*$", savefig = savefig, fileName = 'twoStepGame_' + graph_name + '.pdf' )
curves = dict( )
curves[ '$q^*_k$' ] = number_sensors
curves[ '$MD_k$' ] = numberFixedCameras
methods = [ '$q^*_k$', '$MD_k$' ]
experiments.plotFigure( k_range, curves, ylabel = "Number sensors", methods = legend, savefig = savefig, fileName = 'twoStepGame_' + graph_name + '.pdf' )


plt.plot( k_range, number_sensors )
plt.plot( k_range, numberFixedCameras )
plt.show()

graph_names = [ 'authors', 'copenhagen-calls', 'copenhagen-friends' ]
number_sensors = dict( )
for graph_name in graph_names:
    print( graph_name )
    G = experiments.getRealGraph( graph_name )
    g = ig.Graph.from_networkx( G )
    numberCameras, numberFixedCameras, numberExtraCameras = sequentialGame( g, k_range, print_progress = True )
    number_sensors[ graph_name ] = numberCameras

savefig = False
experiments.plotFigure( k_range, number_sensors, ylabel = "$q^*$", methods = graph_names, savefig = savefig, fileName = 'twoStepGame_realGraphs.pdf' )


"""


def syntheticGraphsSequentialGame( graph_type , n_range = [200,400,600], k_range = [0,1,2,3,4], nAverage = 20 ):
    
    if graph_type not in synthetic_graph_implemented:
        raise TypeError( 'The type of graph should belong to ', str( graph_type ) )
    
    average_numberSensors = dict( )
    std_numberSensors = dict( )
    
    for n in n_range:
        average_numberSensors[ n ] = [ ]
        std_numberSensors[ n ] = [ ]

    for dummy in tqdm( range( len(n_range) ) ):
        n = n_range[ dummy ]
        numberSensors = dict( )
        
        for k in k_range:
            #print( k )
            numberSensors[ k ] = np.zeros( nAverage )
        
        for run in range( nAverage ):
            g = experiments.generateSyntheticGraph( n, graph_type )
            numberCameras, numberFixedCameras, numberExtraCameras = sequentialGame( g, k_range )
            for dummy in range( len( k_range ) ):
                k = k_range[ dummy ]
                numberSensors[ k ][ run ] = numberCameras[ dummy ]
        
        for k in k_range:
            average_numberSensors[ n ].append( np.mean( numberSensors[ k ] ) )
            std_numberSensors[ n ].append( np.std( numberSensors[ k ] ) / np.sqrt( nAverage ) )
        
    return average_numberSensors, std_numberSensors



def sequentialGame( g, relaxation_values, print_progress = False ):
    
    numberCameras = [ ] 
    numberFixedCameras = [ ]
    numberExtraCameras = [ ]
    
    distances = g.distances( )
        
    if print_progress:
        loop = tqdm( range( len( relaxation_values ) ) )
    else:
        loop = range( len( relaxation_values ) )
        
    for k_ in loop:
        k = relaxation_values[ k_ ]
        #for k in relaxation_values:
        fixed_sensors = rmd.relaxedResolvingSet( g, k, print_detailed_running_time = False )
        
        identification_vectors = utils.getIdentificationVectors( fixed_sensors, g = g, distances = distances )
        equivalent_classes = utils.getEquivalentClasses( identification_vectors )
        #nonResolvedSetsOfVertices = utils.getNonResolvedVertices( equivalent_classes )
        
        number_additional_sensors_for_worst_equiv_class = 0
        for equivalent_class in equivalent_classes.values( ):
            additional_sensors = resolveaSetOfVertices( equivalent_class, g = g, distances = distances, print_detailed_running_time=False )
            if len( additional_sensors ) > number_additional_sensors_for_worst_equiv_class:
                number_additional_sensors_for_worst_equiv_class = len( additional_sensors )
        
        numberCameras.append( number_additional_sensors_for_worst_equiv_class + len( fixed_sensors ) )
        numberFixedCameras.append( len( fixed_sensors ) )
        numberExtraCameras.append( number_additional_sensors_for_worst_equiv_class )

    return numberCameras, numberFixedCameras, numberExtraCameras



def resolveaSetOfVertices( target_vertices, g = None, distances = None, fixed_sensors = [ ], print_detailed_running_time = False ):
    
    if g == None and distances == None:
        raise TypeError( 'You need to provide either the distances or the graph' )
    
    if distances == None:
        distances = g.distances( )
        n = g.vcount( )  
    else:
        n = len( distances )

    start = time.time( )
    #all_vertex_pairs_to_resolve = set( [ (min(pair), max(pair) ) for pair in combinations( target_vertices, 2 ) ] ) 
    
    #all_vertex_pairs_to_resolve = set( (u,v) for u in target_vertices for v in target_vertices if u!= v )
    all_vertex_pairs_to_resolve = set( [ (pair[0], pair[1] ) for pair in combinations( target_vertices, 2 ) ] ) 

    non_resolved_pairs = dict( )
    for vertex in range( n ):
        S_vertex = [ ]
        vertices_to_resolve = [ u for u in target_vertices if u != vertex ]
        for dummy in set( distances[ vertex ] ):
            nodes_at_distance_dummy = [ u for u in vertices_to_resolve if distances[ vertex ][ u ] == dummy ]
            for pair in combinations( nodes_at_distance_dummy, 2 ): #loop over all pair of vertices at sane distance of u
                S_vertex.append( pair )
                #S_vertex.append( ( min(pair), max(pair) ) ) #To ensure list of edges is always the one with smallest value first.
        non_resolved_pairs[ vertex ] = set( S_vertex )
    end = time.time()
    if print_detailed_running_time:
        print( 'Subset generation took : ' + str( end - start ) + ' seconds' )

    start = time.time( )
    sensors = set( fixed_sensors )
    additional_sensors = set( )
    while len( all_vertex_pairs_to_resolve ) > 0:
        new_resolving_node = np.argmin( [ len( non_resolved_pairs[ vertex ].intersection( all_vertex_pairs_to_resolve ) ) for vertex in range( n ) ] )
        #NOTE: minimizing the number of non resolved pairs is equivalent to maximizing the number of resolved pairs
        sensors.add( new_resolving_node )
        additional_sensors.add( new_resolving_node )
        resolved_pairs = all_vertex_pairs_to_resolve - non_resolved_pairs[ new_resolving_node ] 
        all_vertex_pairs_to_resolve = all_vertex_pairs_to_resolve.difference( resolved_pairs )
    end = time.time()
    if print_detailed_running_time:
        print( 'Greedy search of resolving vertices took : ' + str( end - start ) + ' seconds' )
    
    return additional_sensors



def sequentialMetricDimention( g = None, distances = None, print_detailed_running_time = False ):
    
    if g == None and distances == None:
        raise TypeError( 'You need to provide either the distances or the graph' )
    
    if distances == None:
        distances = g.distances( )
        n = g.vcount( )  
    else:
        n = len( distances )

    number_of_iterative_placements = [ ]
    for target_vertex in range( n ):
        number_of_iterative_placements.append( len( sequentialLocalisation( target_vertex, distances=distances ) ) )
    
    return number_of_iterative_placements
    


def sequentialLocalisation( target_vertex, g = None, distances = None, fixed_sensors = [ ], print_detailed_running_time = False ):
    
    if g == None and distances == None:
        raise TypeError( 'You need to provide either the distances or the graph' )
    
    if distances == None:
        distances = g.distances( )
        n = g.vcount( )  
    else:
        n = len( distances )
    
    identification_vectors = utils.getIdentificationVectors( fixed_sensors, g = g, distances = distances )
    equivalent_classes = utils.getEquivalentClasses( identification_vectors )
    
    additional_sensors = set( )
    equivalent_class_of_target = equivalent_classes[ str( identification_vectors[target_vertex] ) ]
    possible_locations = equivalent_class_of_target
    while len( possible_locations ) >= 2:
        all_vertex_pairs_to_resolve = set( [ (pair[0], pair[1] ) for pair in combinations( possible_locations, 2 ) ] ) 
        
        non_resolved_pairs = dict( )
        for vertex in range( n ):
            S_vertex = [ ]
            vertices_to_resolve = [ u for u in range( n ) if u != vertex ]
            for dummy in set( distances[ vertex ] ):
                nodes_at_distance_dummy = [ u for u in vertices_to_resolve if distances[ vertex ][ u ] == dummy ]
                for pair in combinations( nodes_at_distance_dummy, 2 ): #loop over all pair of vertices at sane distance of u
                    S_vertex.append( pair )
            non_resolved_pairs[ vertex ] = set( S_vertex )
        
        new_resolving_node = np.argmin( [ len( non_resolved_pairs[ vertex ].intersection( all_vertex_pairs_to_resolve ) ) for vertex in range( n ) ] )
        additional_sensors.add( new_resolving_node )
        resolved_pairs = all_vertex_pairs_to_resolve - non_resolved_pairs[ new_resolving_node ] 
        all_vertex_pairs_to_resolve = all_vertex_pairs_to_resolve.difference( resolved_pairs )
        possible_locations = getPossibleLocations( target_vertex, all_vertex_pairs_to_resolve )

    return additional_sensors



def getPossibleLocations( target_vertex, non_resolved_pairs ):
    
    possible_locations = set( )
    for pair in non_resolved_pairs:
        if target_vertex == pair[ 0 ]:
            possible_locations.add( pair[1] )
        elif target_vertex == pair[ 1 ]:
            possible_locations.add( pair[0] )
    
    return possible_locations    


# =============================================================================
# ALL BELOW SHOULD BE OLD CODE SAFE TO DELETE (AS OF JANUARY 2ND 2025)
# =============================================================================


def sequentialGameOLD( g, relaxation_values, samplingSize = None, print_progress = False ):
    
    if samplingSize == None:
        samplingSize = g.vcount( )
    
    numberCameras = dict( )
    numberFixedCameras = dict( )
    numberExtraCameras = dict( )
    
    for k in relaxation_values:
        numberCameras[ k ] = np.zeros( samplingSize )
        numberFixedCameras[ k ] = np.zeros( samplingSize )
        numberExtraCameras[ k ] = np.zeros( samplingSize )
    
    
    if print_progress:
        loop = tqdm( range( len( relaxation_values ) ) )
    else:
        loop = range( len( relaxation_values ) )
        
    for k_ in loop:
        k = relaxation_values[ k_ ]
        #for k in relaxation_values:
        fixed_cameras = rmd.relaxedResolvingSet( g, k, print_detailed_running_time = False )
        nonResolvedSetsOfVertices = rmd.nonResolvedSets( fixed_cameras, k, g = g )
            
        nonResolvedVertices = [ ]
        for elt in nonResolvedSetsOfVertices:
            nonResolvedVertices += elt
            
        robber_positions = list( range( g.vcount() ) )
        np.random.shuffle( robber_positions )
        robber_positions = robber_positions[ : samplingSize ]
        
        for sampling in range( samplingSize ):
        #for sampling_robber_position in robber_positions:
            
            robber = robber_positions[ sampling ]
                
            if robber not in nonResolvedVertices:
                extra_cameras = [ ]
            else:
                for elt in nonResolvedSetsOfVertices:
                    if robber in elt:
                        setOfRobber = elt
                        break
                extra_cameras = partialResolvingSet( g, setOfRobber, print_detailed_running_time = print_progress )
            
            numberCameras[ k ][ sampling ] = len( extra_cameras ) + len( fixed_cameras )
            numberFixedCameras[ k ][ sampling ] = len( fixed_cameras )
            numberExtraCameras[ k ][ sampling ] = len( extra_cameras )

    return numberCameras, numberFixedCameras, numberExtraCameras


def partialResolvingSet( setToResolve, g = None, distances = None, print_detailed_running_time = True ):
    """    
    Parameters
    ----------
    graph : TYPE
        igraph Graph.
    setToResolve : set
        set of vertices to resolve

    Returns
    -------
    resolving_set : set
        set of vertices to be a k-relaxed resolving set.
    """
    resolving_set = set ( )
    
    if g == None and distances == None:
        raise TypeError( 'You need to provide either the graph or the distances' )
    
    elif distances == None:
        n = g.vcount( )
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
    start = time.time( )
    U = set( (u,v) for u in setToResolve for v in setToResolve if u!= v )
    #U = set( ( u,v ) for u in setToResolve for v in setToResolve if u < v )

    S = { }
    for node in range( n ):
        S_node = [ ]
        #distances_to_nodes = set( distances[ node ].values() )
        distances_to_nodes = set( distances[ node ] )
        nodes_other_than_node = [ u for u in range( n ) if u != node ] #TODO: it seems this line is wrong, need to replace by the set of vertices to be resolved and not all vertices !!!
        for dummy in distances_to_nodes:
            nodes_at_distances_dummy = [ u for u in nodes_other_than_node if distances[node][u] == dummy ]
            
            for pair in combinations( nodes_at_distances_dummy, 2 ):
                S_node.append( pair )
                #S_node.append( ( min(pair), max(pair) ) ) #To ensure list of edges is always the one with smallest value first.
        S[ node ] = set( S_node )
    end = time.time()
    if print_detailed_running_time:
        print( 'Subset generation took : ' + str( end - start ) + ' seconds' )
    
    #print( "Start greedy search of resolving vertices" )
    start = time.time()
    while len( U ) > 0:
        new_resolving_node = np.argmin( [ len( S[ i ].intersection( U ) ) for i in range( n ) ] )
        resolving_set.add( new_resolving_node )
        resolved_pairs = U - S[ new_resolving_node ] 
        U = U.difference( resolved_pairs )
    end = time.time( )
    if print_detailed_running_time:
        print( 'Greedy search of resolving vertices took : ' + str( end - start ) + ' seconds' )

    return resolving_set

