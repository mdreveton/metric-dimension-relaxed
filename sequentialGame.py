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
#from experiments import plotFigure
import experiments as experiments


synthetic_graph_implemented = [ 'BA', 'GW', 'CM', 'RGG', 'kNN' ]


"""

graph_type = 'RGG'

n_range = [ 100, 200, 400 ]
nAverage = 2
samplingSize = 100
#relaxation_values = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16 ]
relaxation_values = [ 0, 1, 2, 3, 4, 5, 6 ]
#relaxation_values = [ 0, 2, 4, 6, 8 ]

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


#############################""

    
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




real_graph_implemented = [ 'authors', 'roads', 'powergrid', 'copenhagen-calls', 'copenhagen-friends', 'yeast' ]

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



def partialResolvingSet( g, setToResolve, print_detailed_running_time = True ):
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
    U = set( (u,v) for u in setToResolve for v in setToResolve if u!= v )
    S = { }
    for node in range( n ):
        S_node = [ ]
        #distances_to_nodes = set( distances[ node ].values() )
        distances_to_nodes = set( distances[ node ] )
        nodes_other_than_node = [ u for u in range( n ) if u != node ]
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
    end = time.time()
    if print_detailed_running_time:
        print( 'Greedy search of resolving vertices took : ' + str( end - start ) + ' seconds' )

    return resolving_set


def getEquivalentClasses( S, u, g = None, distances = None ):
    """
    
    """
    if distances == None:
        distances = g.distances( )
        n = g.vcount()
    if g == None:
        n = len( distances )
    
    equivalentClasses = dict( )    
    equivalentClasses = dict( )
    unique_distance_profiles = [ ]
    for u in range( n ):
        vector_profile_u = [ distances[u][source] for source in S ] 
        if vector_profile_u not in unique_distance_profiles:
            unique_distance_profiles.append( vector_profile_u )
            equivalentClasses[ str(vector_profile_u) ] = [ u ]
        else:
            equivalentClasses[ str(vector_profile_u) ].append( u )
    
    return equivalentClasses


def sequentialGameNew( g, relaxation_values, print_progress = False ):
        
    numberCameras = dict( )
    numberFixedCameras = dict( )
    numberExtraCameras = dict( )
    n = g.vcount( )
    
    for k in relaxation_values:
        numberCameras[ k ] = np.zeros( n )
        numberFixedCameras[ k ] = np.zeros( n )
        numberExtraCameras[ k ] = np.zeros( n )
    
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
        
        equivalent_classes = getEquivalentClasses( g, fixed_cameras )
        
        for equivalent_class in equivalent_classes:
            
        
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

    


def sequentialGame( g, relaxation_values, samplingSize = None, print_progress = False ):
    
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
