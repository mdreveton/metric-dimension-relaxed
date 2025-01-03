#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:03:27 2024

@author: dreveton
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

graph_type = 'RGG'

n_range = [ 200, 600, 1000 ]
nAverage = 2
#k_range = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16 ]
k_range = [ 0, 1, 2, 3, 4, 5, 6 ]
#k_range = [ 0, 2, 4, 6, 8 ]


average_number_sensors, std_number_sensors = syntheticGraphsSequentialGame( graph_type , n_range = n_range, k_range = k_range, nAverage = nAverage )

if graph_type == 'RGG':
    fileName = 'RGG_1,5'
else:
    fileName = graph_type
savefig = False
experiments.plotFigure( relaxation_values, average_number_sensors, std_number_sensors, ylabel = "Sensors", methods = n_range, savefig = savefig, fileName = fileName + '_TotalSensors_nAverage_' + str(nAverage) + '.pdf' )



#############################

    
fixed_cameras = rmd.relaxedResolvingSet( g, k )
nonResolvedSetsOfVertices = rmd.nonResolvedSets( fixed_cameras, k, g = g )

nonResolvedVertices = [ ]
for elt in nonResolvedSetsOfVertices:
    nonResolvedVertices += elt

number_cameras = [ ]

for robber in tqdm( range( n ) ):
    if robber not in nonResolvedVertices:
        extra_cameras = []
    else:
        for elt in nonResolvedSetsOfVertices:
            if robber in elt:
                setOfRobber = elt
                break
        extra_cameras = partialResolvingSet( g, setOfRobber, print_detailed_running_time = False )
    number_cameras.append( len( fixed_cameras ) + len( extra_cameras ) )

plt.plot( relaxation_values, average_number_cameras[200] )
plt.show() 
"""





"""
# =============================================================================
# SEQUENTIAL GAME ON REAL NETWORKS
# =============================================================================

graph_name ='copenhagen-calls'

G = experiments.getRealGraph( graph_name )
g = ig.Graph.from_networkx( G )

relaxation_values = [ 0, 2, 4, 6, 8 ]

numberCameras, numberFixedCameras, numberExtraCameras = sequentialGame( g, relaxation_values, print_progress = True )

average_number_cameras = [ ]
std_number_cameras = [ ]
average_fixedCameras = [ ]
std_fixedCameras = [ ]
average_ExtraCameras = [ ]
std_ExtraCameras = [ ] 
worst_case_number_cameras = [ ]

for k in relaxation_values:
    average_number_cameras.append( np.mean( numberCameras[ k ] ) )
    worst_case_number_cameras.append( np.max( numberCameras[k ] ) )
    average_fixedCameras.append( np.mean( numberFixedCameras[ k ] ) )
    average_ExtraCameras.append( np.mean( numberExtraCameras[ k ] ) )


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
            numberSensors[ k ] = np.zeros( nAverage )
        
        for run in range( nAverage ):
            g = experiments.generateSyntheticGraph( n, graph_type )
            numberCameras, numberFixedCameras, numberExtraCameras = sequentialGame( g, k_range )
            for k in k_range:
                numberSensors[ k ][ run ] = numberCameras[ k ]
        
        for k in k_range:
            average_numberSensors[ n ].append( np.mean( numberSensors[ k ] ) )
            std_numberSensors[ n ].append( np.std( numberSensors[ k ] ) )
        
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



def sequentialMetricDimention( g = None, distances = None, fixed_sensors = [ ], print_detailed_running_time = False ):
    
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




"""
BELOW: SOME OLD CODE SHOULD NOT BE NEEDED ANYMORE (AS OF JANUARY 3RD 2025)

average_number_cameras = dict( )
std_number_cameras = dict( )
average_fixedCameras = dict( )
std_fixedCameras = dict( )
average_ExtraCameras = dict( )
std_ExtraCameras = dict( )

average_worst_case_number_cameras = dict( )
std_worst_case_number_cameras = dict( )

for n in n_range:
    average_number_cameras[ n ] = [ ]
    std_number_cameras[ n ] = [ ]
    
    average_fixedCameras[ n ] = [ ]
    std_fixedCameras[ n ] = [ ]

    average_ExtraCameras[ n ] = [ ]
    std_ExtraCameras[ n ] = [ ]

    average_worst_case_number_cameras[ n ] = [ ]
    std_worst_case_number_cameras[ n ] = [ ]


for dummy in tqdm( range( len( n_range ) ) ):
    n = n_range[ dummy ]
    
    numberCameras = dict( )
    numberFixedCameras = dict( )
    numberExtraCameras = dict( )
    
    if samplingSize == None:
        samplingSize = n
    elif samplingSize > n:
        samplingSize = n
        
    for k in relaxation_values:
        numberCameras[ k ] = np.zeros( ( nAverage, samplingSize ) )
        numberFixedCameras[ k ] = np.zeros( ( nAverage, samplingSize ) )
        numberExtraCameras[ k ] = np.zeros( ( nAverage, samplingSize ) )
        
    for run in range( nAverage ):
        
        g = experiments.generateSyntheticGraph( n, graph_type )
        
        for k_ in tqdm( range( len(relaxation_values) ) ):
            k = relaxation_values[ k_ ]
        #for k in relaxation_values:
            fixed_cameras = rmd.relaxedResolvingSet( g, k, print_detailed_running_time = False )
            nonResolvedSetsOfVertices = rmd.nonResolvedSets( fixed_cameras, k, g = g )
            
            nonResolvedVertices = [ ]
            for elt in nonResolvedSetsOfVertices:
                nonResolvedVertices += elt
            
            robber_positions = list( range(n) )
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
                    extra_cameras = partialResolvingSet( g, setOfRobber, print_detailed_running_time = False )
                
                numberCameras[ k ][ run, sampling ] = len( extra_cameras ) + len( fixed_cameras )
                numberFixedCameras[ k ][ run, sampling ] = len( fixed_cameras )
                numberExtraCameras[ k ][ run, sampling ] = len( extra_cameras )
                
    for k in relaxation_values:
        average_number_cameras[ n ].append( np.mean( numberCameras[k] ) )
        std_number_cameras[ n ].append( np.std( numberCameras[k] ) )
        
        average_fixedCameras[ n ].append( np.mean( numberFixedCameras[k] ) )
        std_fixedCameras[ n ].append( np.std( numberFixedCameras[k] ) )

        average_ExtraCameras[ n ].append( np.mean( numberExtraCameras[k] ) )
        std_ExtraCameras[ n ].append( np.std( numberExtraCameras[k] ) )

        average_worst_case_number_cameras[ n ].append( np.mean( [ np.max( numberCameras[k][run,:]) for run in range(nAverage) ] ) ) 
        std_worst_case_number_cameras[ n ].append( np.std( [ np.max( numberExtraCameras[k][run,:]) for run in range( nAverage ) ] ) ) 

if graph_type == 'RGG':
    fileName = 'RGG_1,5'
else:
    fileName = graph_type
savefig = False
experiments.plotFigure( relaxation_values, average_number_cameras, std_number_cameras, ylabel = "Sensors", methods = n_range, savefig = savefig, fileName = fileName + '_TotalCameras_nAverage_' + str(nAverage) + '_samplingSize_' + str(samplingSize) + '.pdf' )
experiments.plotFigure( relaxation_values, average_worst_case_number_cameras, std_worst_case_number_cameras, ylabel = "Sensors", methods = n_range, savefig = savefig, fileName = fileName + '_TotalCameras_worstCase_nAverage_' + str(nAverage) + '_samplingSize_' + str(samplingSize) + '.pdf' )
experiments.plotFigure( relaxation_values, average_fixedCameras, std_fixedCameras, ylabel = "Fixed cameras", methods = n_range, savefig = savefig, fileName = fileName + '_FixedCameras_nAverage_' + str(nAverage) + '_samplingSize_' + str(samplingSize) + '.pdf' )
experiments.plotFigure( relaxation_values, average_ExtraCameras, std_ExtraCameras, ylabel = "Extra cameras", methods = n_range, savefig = savefig, fileName = fileName + '_ExtraCameras_nAverage_' + str(nAverage) + '_samplingSize_' + str(samplingSize) + '.pdf' )


"""