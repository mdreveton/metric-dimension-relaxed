#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:48:59 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import scipy as sp
from tqdm import tqdm

import relaxedMetricDimension as rmd
import galtonWatsonTree as gw
import utils as utils


#from networkx.drawing.nx_pydot import graphviz_layout
#from networkx.nx_agraph.graphviz_layout import graphviz_layout

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


synthetic_graph_implemented = [ 'BA', 'GW', 'CM', 'RGG', 'kNN' ]
real_graph_implemented = [ 'authors', 'roads', 'powergrid', 'copenhagen-calls', 'copenhagen-friends', 'yeast' ]
#Note: computations of metric dimension on the powergrid network takes time.


"""

# =============================================================================
# SYNTHETIC GRAPHS: EVOLUTION OF MD AS FUNCTION OF THE RELAXATION PARAMETER k 
# =============================================================================

graph_type = 'GW'
n_range = [ 200, 600, 1000 ]
nAverage = 20
#k_range = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9 ]
k_range = [ 0, 1, 2, 3, 4, 5, 6 ]

if graph_type == 'RGG' or graph_type == 'kNN':
    k_range = [ 0, 1, 2, 3, 4, 5 ]
elif graph_type == 'BA' or graph_type == 'GW':
    k_range = [ 0, 1, 2, 3, 4, 5, 6, 7 ]


average_sizeResolvingSet, std_sizeResolvingSet, average_sizeLargestNonResolvedSet, std_sizeLargestNonResolvedSet, average_ratioNonResolvedVertices, std_ratioNonResolvedVertices = syntheticGraphsEvolutionMetricDimension( graph_type , n_range = n_range, k_range = k_range, nAverage = nAverage )
    
if graph_type == 'RGG':
    fileName = 'RGG_1,5'
elif graph_type == 'kNN':
    fileName = 'kNN_2'
else:
    fileName = graph_type

average_ratioVerticesInResolvingSet = dict( )
std_ratioVerticesInResolvingSet = dict( )
for n in n_range:
    average_ratioVerticesInResolvingSet[ n ] = list( np.asarray( average_sizeResolvingSet[ n ] ) / n )
    std_ratioVerticesInResolvingSet[ n ] = list( np.asarray( std_sizeResolvingSet[ n ] ) / n )

average_alpha_over_n = dict( )
std_alpha_over_n = dict( )
for n in n_range:
    average_alpha_over_n[ n ] = list( np.asarray( average_sizeLargestNonResolvedSet[ n ] ) / n )
    std_alpha_over_n[ n ] = list( np.asarray( std_sizeLargestNonResolvedSet[ n ] ) / n )


savefig = False
#plotFigure( k_range, average_sizeResolvingSet, accuracy_err = std_sizeResolvingSet, ylabel = "$MD_k$", methods = n_range, savefig = savefig, fileName = fileName + '_MDk_nAverage_' + str(nAverage) + '.pdf' )
plotFigure( k_range, average_ratioVerticesInResolvingSet, accuracy_err = std_ratioVerticesInResolvingSet, ylabel = "$MD_k \, / \, n$", methods = n_range, savefig = savefig, fileName = fileName + '_MDk_over_n_nAverage_' + str(nAverage) + '.pdf' )
#plotFigure( k_range, average_relaxation_ratio, accuracy_err = std_relaxation_ratio, ylabel = "Relaxation ratio", methods = n_range, savefig = savefig, fileName = fileName + '_relaxationRatio_nAverage_' + str(nAverage) + '.pdf' )
plotFigure( k_range, average_sizeLargestNonResolvedSet, accuracy_err = std_sizeLargestNonResolvedSet, ylabel = r"$\alpha$", methods = n_range, savefig = savefig, fileName = fileName + '_sizeLargestNonResolved_nAverage_' + str(nAverage) + '.pdf' )
plotFigure( k_range, average_ratioNonResolvedVertices, accuracy_err = std_ratioNonResolvedVertices, ylabel = 'Ratio non-resolved', methods = n_range, savefig = savefig, fileName = fileName + '_ratioNonResolvedVertices_nAverage_' + str(nAverage) + '.pdf' )
plotFigure( k_range, average_alpha_over_n, accuracy_err = std_alpha_over_n, ylabel = r"$\alpha \, / \, n$", methods = n_range, savefig = savefig, fileName = fileName + '_relativeSizeLargestNonResolved_nAverage_' + str(nAverage) + '.pdf' )

# =============================================================================
# SYNTHETIC GRAPHS: VIZUALISATION OF RESOLVING SET IN SMALL GRAPHS
# =============================================================================

n = 100
graph_type = 'BA'
seed = 896803 #Note that the seed for GW trees does not work (offspring generation does not care about the seed)

if graph_type == 'BA':
    G = nx.barabasi_albert_graph( n, 1, seed = seed )
    my_pos = nx.nx_agraph.graphviz_layout( G, prog = "twopi" )
    g = ig.Graph.from_networkx( G )

elif graph_type == 'GW':
    g = gw.GaltonWatsonRandomTree( n, distribution_name = 'poisson', mean = 3 )
    G = nx.from_edgelist( g.get_edgelist() )
    my_pos = nx.nx_agraph.graphviz_layout( G, prog="twopi" )
    
elif graph_type == 'RGG':
    G = nx.random_geometric_graph( n, 1.5 * np.sqrt( np.log(n) / ( n * np.pi ) ), seed = seed )
    my_pos = nx.get_node_attributes( G, "pos" )
    g = ig.Graph.from_networkx( G )

elif graph_type == 'kNN':
    X = np.random.multivariate_normal( np.zeros(2), np.eye(2), n )    
    g = kNN_graph( X )
    my_pos = dict( )
    for i in range( n ):
        my_pos[ i ] = X[ i, : ]
    G = nx.from_edgelist( g.get_edgelist() )

k_range = [ ]
if graph_type in [ 'BA', 'GW' ]:
    k_range = [ 0, 2, 4 ]
else:
    k_range = [ 0, 1, 2, 3, 4 ]

singleGraphVisualization( g, G, my_pos = my_pos, filename = graph_type, savefig = False )



# =============================================================================
# SYNTHETIC GRAPHS: DISTRIBUTION OF THE SIZES OF THE EQUIVALENT CLASSES 
# =============================================================================

n = 1000
graph_type = 'BA'
savefig = False

g = generateSyntheticGraph( n, graph_type )

k = 2
resolving_set = rmd.relaxedResolvingSet( g, k )
                    
identification_vectors = utils.getIdentificationVectors( resolving_set, g = g )
equivalent_classes = utils.getEquivalentClasses( identification_vectors )
    
sizes_equivalent_classes = [ ]
sizes_non_resolved_equivalent_classes = [ ]
for equivalent_class in equivalent_classes.values():
    sizes_equivalent_classes.append( len( equivalent_class ) )
    if len( equivalent_class ) > 1:
        sizes_non_resolved_equivalent_classes.append( len( equivalent_class ) )
    
plt.hist( sizes_non_resolved_equivalent_classes )
plt.xlabel( 'Number of vertices', fontsize = SIZE_LABELS )
plt.ylabel( 'Frequency', fontsize = SIZE_LABELS )
plt.xticks( fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
if(savefig):
    plt.savefig( graph_type + '_distribution_non_resolved_classes_k_' + str(k) + '.pdf' , bbox_inches='tight' )
plt.show( )    


degrees = g.degree( [i for i in range(n)] )
degrees_non_leaves = [ ]
for i in range( n ):
    if degrees[i] != 1:
        degrees_non_leaves.append( degrees[ i ] )
plt.hist( degrees_non_leaves )
plt.show( )


# =============================================================================
# SYNTHETIC GRAPHS: PLOT WITH NUMBER NON-RESOLVED VERTICES AND LARGEST EQUIVALENT CLASS 
# =============================================================================

n = 1000
graph_type = 'BA'

g = generateSyntheticGraph( n, graph_type )

k_range = [ i for i in range( g.diameter( ) + 1 ) ]

number_non_resolved_vertices = [ ]
sizes_equivalent_classes = [ ]

for k in tqdm( k_range ):
    resolving_set = rmd.relaxedResolvingSet( g, k )
    
    identification_vectors = utils.getIdentificationVectors( resolving_set, g = g )
    equivalent_classes = utils.getEquivalentClasses( identification_vectors )
    non_resolved_vertices = utils.getNonResolvedVertices( equivalent_classes )
    
    number_non_resolved_vertices.append( len( non_resolved_vertices ) ) 
    sizes_equivalent_classes.append( np.max( [ len( subset ) for subset in equivalent_classes.values() ] ) )


savefig = True
plt.plot( k_range, number_non_resolved_vertices, linestyle = '-.', marker = '.', label = 'Non-resolved' )
plt.plot( k_range, sizes_equivalent_classes, linestyle = '-.', marker = '.', label = r'$\alpha$' )
legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )
plt.xlabel( 'k', fontsize = SIZE_LABELS )
plt.ylabel( 'Number of vertices', fontsize = SIZE_LABELS )
plt.xticks( fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
if(savefig):
    plt.savefig( graph_type + '_non_resolved_and_alpha.pdf' , bbox_inches='tight' )
plt.show( )



# =============================================================================
# SYNTHETIC GRAPHS: SBM
# =============================================================================

n_per_community = 300
k = 2

n = n_per_community * k
block_sizes = [ n_per_community for i in range(k) ]

a = 4
b = 2

rate_matrix = b * np.ones( k ) + (a-b) * np.eye( k )
pref_matrix = rate_matrix * np.log( n ) / n

g = ig.GraphBase.SBM( n, pref_matrix, block_sizes, directed=False, loops=False )


largest_cc = getLargestConnectedComponent( g )


difficulty = (np.sqrt(a) - np.sqrt(b))**2




# =============================================================================
# REAL GRAPHS: STATISTICS AND EVOLUTION OF METRIC DIMENSION
# =============================================================================

k_range = [ 0, 1, 2, 3, 4 ]
graph_names = [ 'authors', 'copenhagen-calls', 'copenhagen-friends', 'yeast' ]
sizeResolvingSet, relaxationRatio, largestNonResolvedSet, ratioNonResolvedVertices = realGraphsEvolutionMetricDimension( graph_names = graph_names, k_range = k_range )

savefig = False
plotFigure( k_range, sizeResolvingSet, ylabel = "$MD_k$", methods = graph_names, savefig = savefig, fileName = 'realGraphs_MDk.pdf', xticks = relaxation_values )
plotFigure( k_range, relaxationRatio, ylabel = "Relaxation ratio", methods = graph_names, savefig = savefig, fileName =  'realGraphs_relaxationRatio.pdf', xticks = relaxation_values )
plotFigure( k_range, largestNonResolvedSet, ylabel = r"$\alpha$", methods = graph_names, savefig = savefig, fileName = 'realGraphs_sizeLargestNonResolved.pdf', xticks = relaxation_values )
plotFigure( k_range, ratioNonResolvedVertices, ylabel = 'Ratio non-resolved', methods = graph_names, savefig = savefig, fileName = 'realGraph_ratioNonResolvedVertices.pdf', xticks = relaxation_values )


for graph_name in graph_names:
    for k in k_range:
        relaxationRatio[ graph_name ].append( sizeResolvingSet[ graph_name ][ k ] / sizeResolvingSet[ graph_name ][ 0 ] )

relativeSizeResolvingSet = dict ( )
average_alpha_over_n = dict( )
for graph_name in graph_names:
    relativeSizeResolvingSet[ graph_name ] = [ ]
    average_alpha_over_n[ graph_name ] = [ ]
    G = getRealGraph( graph_name )
    for k in k_range:
        relativeSizeResolvingSet[ graph_name ].append( sizeResolvingSet[ graph_name ][ k ] / nx.number_of_nodes( G ) )
        average_alpha_over_n[ graph_name ].append( largestNonResolvedSet[ graph_name ][ k ] / nx.number_of_nodes( G ) )

plotFigure( k_range, relativeSizeResolvingSet, ylabel = "$MD_k \, / \, n$", methods = graph_names, savefig = savefig, fileName = 'realGraphs_relativeSizeResolvingSet.pdf', xticks = relaxation_values )
plotFigure( k_range, average_alpha_over_n, ylabel = r"$\alpha \, / \, n$", methods = graph_names, savefig = savefig, fileName = 'realGraphs_relativeSizeLargestNonResolved.pdf', xticks = relaxation_values )


# =============================================================================
# REAL GRAPHS: VISUALIZATION
# =============================================================================

statistics = getGraphsStatistics( graph_names = real_graph_implemented )

graph_name = 'copenhagen-calls'
G = getRealGraph( graph_name )
g = ig.Graph.from_networkx( G )
my_pos = getRealGraphsPositions( graph_name, G = G, seed = 1 )
#my_pos = nx.spring_layout( G, seed = 1 )

singleGraphVisualization(g, G, my_pos = my_pos, filename = graph_name, savefig = False )


"""



"""
OLD CODE COULD BE SAFELY DELETED (25 DECEMBER 2024)

g = ig.GraphBase.Tree_Game( 100 )
g = ig.GraphBase.Barabasi( 75, 1 )

g = ig.Graph.GRG(n, 1.5 * np.sqrt( np.log(n) / ( n * np.pi ) ) )
G = nx.Graph( g.get_edgelist() )


n = 75
G = nx.random_geometric_graph( n, 1.5 * np.sqrt( np.log(n) / ( n * np.pi ) ), seed=896803)
my_pos = nx.get_node_attributes(G, "pos")
#g = ig.Graph.TupleList( G.edges() )
g = ig.Graph.from_networkx( G )
#my_pos = nx.spring_layout( G, seed = 1 )
#my_pos = graphviz_layout( G, prog="twopi" )

"""


# =============================================================================
# EVOLUTION METRIC DIMENSION
# =============================================================================

def syntheticGraphsEvolutionMetricDimension( graph_type , n_range = [200,400,600], k_range = [0,1,2,3,4], nAverage = 20 ):
    
    if graph_type not in synthetic_graph_implemented:
        raise TypeError( 'The type of graph should belong to ', str( graph_type ) )
    
    average_sizeResolvingSet = dict( )
    std_sizeResolvingSet = dict( )

    average_sizeLargestNonResolvedSet = dict( )
    std_sizeLargestNonResolvedSet = dict( )

    average_ratioNonResolvedVertices = dict( )
    std_ratioNonResolvedVertices = dict( )

    for n in n_range:
        average_sizeResolvingSet[ n ] = [ ]
        std_sizeResolvingSet[ n ] = [ ]
        
        average_sizeLargestNonResolvedSet[ n ] = [ ]
        std_sizeLargestNonResolvedSet[ n ] = [ ]
        
        average_ratioNonResolvedVertices[ n ] = [ ]
        std_ratioNonResolvedVertices[ n ] = [ ]
        

    for dummy in tqdm( range(len(n_range))):
        n = n_range[ dummy ]
        
        sizeLargestNonResolvedSet = dict( )
        ratioNonResolvedVertices = dict( )
        sizeResolvingSet = dict( )
        ratioWithDegree = dict( )
        
        for k in k_range:
            sizeLargestNonResolvedSet[ k ] = np.zeros( nAverage )
            ratioNonResolvedVertices[ k ] = np.zeros( nAverage )
            sizeResolvingSet[ k ] = np.zeros( nAverage )
            ratioWithDegree[ k ] = np.zeros( nAverage )
        
        for run in range( nAverage ):
            
            g = generateSyntheticGraph( n, graph_type )

            for k in k_range:
                resolving_set = rmd.relaxedResolvingSet( g, k )
                
                identification_vectors = utils.getIdentificationVectors( resolving_set, g = g )
                equivalent_classes = utils.getEquivalentClasses( identification_vectors )
                non_resolved_equivalent_classes = utils.getNonResolvedEquivalentClasses( equivalent_classes )
                non_resolved_sets_of_vertices = utils.getNonResolvedVertices( equivalent_classes )
                
                sizeResolvingSet[ k ][ run ] = len( resolving_set )
                
                if non_resolved_sets_of_vertices == [ ]: #all the vertices are resolved
                    sizeLargestNonResolvedSet[ k ][ run ] = 0
                    ratioNonResolvedVertices[ k ][ run ] = 0 
                else:
                    sizeLargestNonResolvedSet[ k ][ run ] = np.max( [ len( subset ) for subset in non_resolved_equivalent_classes ] )
                    ratioNonResolvedVertices[ k ][ run ] = len( non_resolved_sets_of_vertices ) / n
                
                average_degree = 2 * g.ecount( ) / g.vcount( )
                ratioWithDegree[ k ][ run ] = (average_degree)**( k//2 ) * sizeResolvingSet[ k ][ run ] / sizeResolvingSet[ 0 ][ run ]

        for k in k_range:
            average_sizeResolvingSet[ n ].append( np.mean( sizeResolvingSet[k] ) )
            std_sizeResolvingSet[ n ].append( np.std( sizeResolvingSet[k] ) )

            #average_relaxation_ratio[ n ].append( np.mean( sizeResolvingSet[k] / sizeResolvingSet[ 0 ] ) )
            #std_relaxation_ratio[ n ].append( np.std( sizeResolvingSet[k] / sizeResolvingSet[ 0 ] ) )
            
            average_sizeLargestNonResolvedSet[ n ].append( np.mean( sizeLargestNonResolvedSet[k] ) )
            std_sizeLargestNonResolvedSet[ n ].append( np.std( sizeLargestNonResolvedSet[k] ) )
            
            average_ratioNonResolvedVertices[ n ].append( np.mean( ratioNonResolvedVertices[k] ) )
            std_ratioNonResolvedVertices[ n ].append( np.std( ratioNonResolvedVertices[k] ) )
            
            #average_ratioWithDegree[ n ].append( np.mean( ratioWithDegree[k] ) )
            #std_ratioWithDegree[ n ].append( np.std( ratioWithDegree[k] ) )

    return average_sizeResolvingSet, std_sizeResolvingSet, average_sizeLargestNonResolvedSet, std_sizeLargestNonResolvedSet, average_ratioNonResolvedVertices, std_ratioNonResolvedVertices


def realGraphsEvolutionMetricDimension( graph_names = real_graph_implemented, k_range = [0,1,2,3,4] ):
    
    sizeResolvingSet = dict( )
    relaxationRatio = dict( )
    largestNonResolvedSet = dict( )
    ratioNonResolvedVertices = dict( )
    #ratioWithDegree = dict( )
    
    for graph_name in graph_names:
        sizeResolvingSet[ graph_name ] = [ ]
        relaxationRatio[ graph_name ] = [ ]
        largestNonResolvedSet[ graph_name ] = [ ]
        ratioNonResolvedVertices[ graph_name ] = [ ]
        
    for graph_name in graph_names:
        print( graph_name )
        
        G = getRealGraph( graph_name )
        g = ig.Graph.from_networkx( G )
                
        for k in k_range:
            resolving_set = rmd.relaxedResolvingSet( g, k )
            
            #nonResolvedSetsOfVertices = rmd.nonResolvedSets( resolving_set, k, g = g )
            identification_vectors = utils.getIdentificationVectors( resolving_set, g = g )
            equivalent_classes = utils.getEquivalentClasses( identification_vectors )
            non_resolved_equivalent_classes = utils.getNonResolvedEquivalentClasses( equivalent_classes )
            non_resolved_vertices = utils.getNonResolvedVertices( equivalent_classes )

            sizeResolvingSet[ graph_name ].append( len( resolving_set ) )
            
            if len( non_resolved_vertices ) == 0:
                largestNonResolvedSet[ graph_name ].append( 0 )
                ratioNonResolvedVertices[ graph_name ].append( 0 ) 
            else:
                largestNonResolvedSet[ graph_name ].append( np.max( [ len(subset) for subset in non_resolved_equivalent_classes ] ) )
                ratioNonResolvedVertices[ graph_name ].append( len( non_resolved_vertices ) / g.vcount( ) )
                
                #average_degree = 2 * g.ecount( ) / g.vcount( )
                #ratioWithDegree[ graph_name ][ k ] = (average_degree)**( k//2 ) * sizeResolvingSet[ k ][ run ] / sizeResolvingSet[ 0 ][ run ]
    
    return sizeResolvingSet, relaxationRatio, largestNonResolvedSet, ratioNonResolvedVertices



# =============================================================================
# CODE TO GENERATE SYNTHETIC AND REAL GRAPHS
# =============================================================================


def generateSyntheticGraph( n, graph_type ):
    if graph_type not in synthetic_graph_implemented:
        raise TypeError( 'This synthetic graph is not implemented' )
    
    if graph_type == 'BA':
        g = ig.GraphBase.Barabasi( n, 1 )
        
    elif graph_type == 'GW':
        g = gw.GaltonWatsonRandomTree( n, distribution_name = 'poisson', mean = 3 )
            
    elif graph_type == 'CM':
        #degree_sequence = 3 * np.ones(n) +  sp.stats.lognorm.rvs( 2, loc = 0, size = n )
        #degree_sequence = sp.stats.lognorm.rvs( 2, loc = 3, size = n )
        degree_sequence = [0,1]
        while( ig.is_graphical( degree_sequence ) == False ):
            degree_sequence = 2 * np.ones(n) + sp.stats.zipfian.rvs( 2.5, n-3, size = n )
            for i in range(n):
                degree_sequence[ i ] = int( degree_sequence[i] )
                if degree_sequence[ i ] > 150:
                    print( 'The degree of a vertex is too high: we reduced to 150' )
                    degree_sequence[ i ] = np.sqrt( n )
            if np.sum( degree_sequence ) % 2 == 1:
                degree_sequence[ 1 ] += 1 
            
        g = ig.Graph.Degree_Sequence( degree_sequence, method = 'configuration' )
        g = ig.GraphBase.simplify( g )
        
        """
        if enforceConnected:
            max_tries = 100
            number_trials = 0
            while (number_trials < max_tries and g.is_connected() == False ):
                degree_sequence = sp.stats.lognorm.rvs( 2, loc = 2, size = n )
                for i in range(n):
                    degree_sequence[ i ] = int( degree_sequence[i] ) 
                if np.sum( degree_sequence ) % 2 == 1:
                    degree_sequence[ 1 ] += 1 
                G = nx.configuration_model( degree_sequence )
                g = ig.from_networkx( G )
                #g = ig.Graph.Degree_Sequence( degree_sequence, method = 'configuration' )
                g = ig.GraphBase.simplify( g )
                number_trials += 1
        """
        
    elif graph_type == 'RGG':
        g = ig.Graph.GRG(n, 1.5 * np.sqrt( np.log(n) / ( n * np.pi ) ) )
        """
        if enforceConnected:
            max_tries = 100
            number_trials = 0
            while (number_trials < max_tries and g.is_connected() == False ):
                g = ig.Graph.GRG(n, 1.5 * np.sqrt( np.log(n) / ( n * np.pi ) ) )
                number_trials += 1
        """    
    elif graph_type == 'kNN':
        #X = np.random.multivariate_normal( np.zeros(2), np.eye(2), n )
        mu = 2
        X1 = np.random.multivariate_normal( mu * np.ones(2), np.eye(2), n//2 )
        X2 = np.random.multivariate_normal( - mu * np.ones(2), np.eye(2), n//2 )
        X = np.concatenate( (X1, X2) )
        g = kNN_graph( X )

    else:
        raise TypeError( 'graph type is not implemented' )

    if g.is_connected( ) == False:
        print( 'The synthetic random graph generated is not connected' )

    return g



def kNN_graph( X, k = 10 ):
    """
    
    Create the k-nearest neighbor graph from a cloud of points
    
    Parameters
    ----------
    X : n by d array
        Positions of the n data points in a d dimensional space.
    k : TYPE, optional
        Number of nearest neighbors. The default is 10.

    Returns
    -------
    g : igraph graph
        k-NN graph.
    """
    
    #X = np.random.multivariate_normal( [0,0], [ [1,0], [0,1] ], n )
    
    (n,d) = X.shape
    
    Xtree = sp.spatial.cKDTree( X )
    knn_dist, knn_ind = Xtree.query( X, k=k )

    
    #Flatten knn data to get targets of edges
    targets = knn_ind.flatten( )

    #Sources of edges
    sources = np.ones( ( n,k ), dtype = int ) * np.arange( n )[ :,None ]
    sources = sources.flatten( )
    
    #Create graph
    edgelist = list( zip( sources.tolist(), targets.tolist() ) )
    g = ig.Graph( edgelist )
    g_ = g.simplify() #Delete multi-edges and self-loops
    
    return g_


def getRealGraph( graph_name ):
    
    if graph_name not in real_graph_implemented:
        raise TypeError( 'This graph is not implemented' )
        
    if graph_name.lower( ) == 'authors':
        # Read the edge list from the file
        edge_list = [ ]
        with open('datasets/dimacs10-netscience/out.dimacs10-netscience', 'r') as f:
            for line in f:
                parts = line.split( )
                source = int( parts[0] )
                target = int( parts[1] )
                edge_list.append( ( source, target ) )
        G = nx.from_edgelist( edge_list )
        
    elif graph_name.lower( ) == 'roads':
        G = nx.read_edgelist('datasets/subelj_euroroad/out.subelj_euroroad_euroroad', delimiter=' ')
    
    elif graph_name.lower( ) == 'powergrid':
        G = nx.read_edgelist('datasets/opsahl-powergrid/out.opsahl-powergrid', delimiter=' ')
        
    elif graph_name.lower( ) == 'copenhagen-calls':
        G = nx.read_edgelist('datasets/Copenhagen-graphs/calls.csv/edges.csv', delimiter=',', data=(('timestamp', int),('duration', int)))
        
        positions = {}
        i = 0
        with open('datasets/Copenhagen-graphs/calls.csv/nodes.csv', 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                _, pos,_ = line.strip().split('"')
                pos = np.array([float(coord) for coord in pos.strip('array([])').split(',')])
                positions[str(i)] = pos
                i += 1
        
        for node in G.nodes:
            G.nodes[ node ][ 'position' ] = positions[ node ]

    elif graph_name.lower( ) == 'copenhagen-friends':
        G = nx.read_edgelist('datasets/Copenhagen-graphs/fb_friends.csv/edges.csv', delimiter=',')
        
        mapping = dict( )
        for node in G.nodes:
            mapping[ node ] = node.rstrip()
        G = nx.relabel_nodes( G, mapping )

        
        positions = {}
        i = 0
        with open('datasets/Copenhagen-graphs/fb_friends.csv/nodes.csv', 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                _, pos,_ = line.strip().split('"')
                pos = np.array([float(coord) for coord in pos.strip('array([])').split(',')])
                positions[str(i)] = pos
                i += 1
                
        for node in G.nodes:
            G.nodes[ node ][ 'position' ] = positions[ node ]

    elif graph_name.lower( ) == 'yeast' :
        G = nx.read_edgelist('datasets/moreno_propro/out.moreno_propro_propro', delimiter=' ')

    G.remove_edges_from( nx.selfloop_edges( G ) ) 

    if not nx.is_connected( G ):
        S = max( nx.connected_components(G), key=len )
        G = G.subgraph( S ).copy() 
        
    return nx.convert_node_labels_to_integers( G )


def getRealGraphsPositions( graph_name, G = None, seed = 1 ):
    
    if graph_name not in real_graph_implemented:
        raise TypeError( 'This graph is not implemented' )
    
    if G is None:
        G = getRealGraph( graph_name )

    if graph_name.lower( ) in [ 'copenhagen-calls', 'copenhagen-friends' ]:
        positions = nx.get_node_attributes( G, "position" )
        
    else:
        positions = nx.spring_layout( G, seed = seed )
        
    return positions



# =============================================================================
# CODE TO PLOT FIGURES AND VISUALIZE GRAPHS
# =============================================================================

def plotFigure( x, accuracy_mean, accuracy_err = None, methods = None, 
               xticks = None, yticks = None,
               xlabel = "k", ylabel = "Relative size",
               savefig = False, fileName = "fig.pdf" ):
    
    if methods is None and accuracy_err is None:
        plt.plot( x, accuracy_mean, linestyle = '-.', marker = '.' )
        
    elif methods is None and accuracy_err is not None:
        plt.errorbar( x, accuracy_mean, yerr = accuracy_err, linestyle = '-.' )
        
    elif methods is not None and accuracy_err is None:
        for method in methods:
            plt.plot( x, accuracy_mean[ method ], linestyle = '-.', marker = '.', label = method )
            legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
            plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )
    
    elif methods is not None and accuracy_err is not None:
        for method in methods:
            plt.errorbar( x, accuracy_mean[ method ], yerr = accuracy_err[ method ], linestyle = '-.', label = method )
            legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
            plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )

    plt.xlabel( xlabel, fontsize = SIZE_LABELS )
    plt.ylabel( ylabel, fontsize = SIZE_LABELS )
    
    if xticks != None:
        plt.xticks( xticks, fontsize = SIZE_TICKS )
    else:
        plt.xticks( fontsize = SIZE_TICKS )
    
    if yticks != None:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    else:
        plt.yticks( fontsize = SIZE_TICKS )

    if(savefig):
        plt.savefig( fileName, bbox_inches='tight' )
    plt.show( )    
    

    
def singleGraphVisualization( g = None, G = None, my_pos = None, k_range = [0,1,2,3,4], 
                             filename = 'mygraph', savefig = False,
                             node_size = 10 ):
    
    if g is None and G is None:
        raise TypeError( 'You should provide at least g (is igraph format) or G (in networkx format)' )
        
    if g is None:
        G = nx.Graph( g.get_edgelist() )
        
    if G is None:
        g = ig.Graph.from_networkx( G )
    
    if my_pos is None:
        my_pos = nx.spring_layout( G )
        
    resolvingSet = dict( )
    nonResolvedSetsOfVertices = dict( )
    largestUnresolvedSet = dict( )
    resolvedVertices = dict( )

    for k in k_range:
        
        resolvingSet[k] = rmd.relaxedResolvingSet( g, k )
        identification_vector = utils.getIdentificationVectors( resolvingSet[k], g = g )
        equivalent_classes = utils.getEquivalentClasses( identification_vector )
        nonResolvedSetsOfVertices = utils.getNonResolvedEquivalentClasses( equivalent_classes )
        
        if len( nonResolvedSetsOfVertices ) > 0:
            largestUnresolvedSet[ k ] = nonResolvedSetsOfVertices[ np.argmax( [ len( subset) for subset in nonResolvedSetsOfVertices ] ) ]
        else:
            largestUnresolvedSet[ k ] = [ ]
            
        nonResolvedVertices = [ ]
        for nonResolved_set in nonResolvedSetsOfVertices:
            nonResolvedVertices += nonResolved_set
            
        resolvedVertices[ k ] = [ i for i in range( g.vcount( ) ) if i not in nonResolvedVertices and i not in resolvingSet[ k ] ]
        
    for k in k_range:    
        plt.figure( figsize=(12, 8) )
        nx.draw( G, node_size = node_size, pos = my_pos, with_labels = False)
        nx.draw_networkx_nodes( G, label = True, pos = my_pos, linewidths = 2, edgecolors = 'black', node_color = 'white')
        nx.draw_networkx_nodes( G, pos = my_pos,label = True, nodelist = resolvingSet[ k ], linewidths = 2, edgecolors = 'black', node_color = 'red')
        nx.draw_networkx_nodes( G, pos = my_pos, nodelist = largestUnresolvedSet[ k ], linewidths = 2, edgecolors = 'black', node_color = 'orange')
        nx.draw_networkx_nodes( G, pos = my_pos, nodelist = resolvedVertices[ k ], linewidths = 2, edgecolors = 'black', node_color = 'lightgreen' )
        nx.draw_networkx_edges( G, my_pos, width = 1 )
        if savefig:
            plt.savefig( filename + '_md_' + str(k) + '.pdf')
        plt.show()


# =============================================================================
# STATISTICS OF THE REAL AND SYNTHETIC GRAPHS
# =============================================================================

def getGraphsStatistics( graph_names = real_graph_implemented ):
    
    statistics = dict( )
    for graph in graph_names:
        print( graph )
        G = getRealGraph( graph )
        g = ig.Graph.from_networkx(G)
        
        graph_stats = dict ( )
        graph_stats[ 'n' ] = g.vcount( )
        graph_stats[ 'm' ] = g.ecount( )
        graph_stats[ 'average degree' ] = 2 * g.ecount( ) / g.vcount( )
        graph_stats[ 'diameter' ] = g.diameter( )
        graph_stats[ 'average shortest path length' ] = nx.average_shortest_path_length( G )
        graph_stats[ '1-shell' ] = nx.k_shell(G, k=1).number_of_nodes( )
        statistics[ graph ] = graph_stats
    
    return statistics


def getRandomGraphsStatistics ( n, graph_type, nAverage = 100 ):
        
    graph_stats = dict ( )
    graph_stats[ 'n' ] = np.zeros( nAverage )
    graph_stats[ 'm' ] = np.zeros( nAverage )
    graph_stats[ 'average degree' ] = np.zeros( nAverage )
    graph_stats[ 'diameter' ] = np.zeros( nAverage )
    graph_stats[ 'average shortest path length' ] = np.zeros( nAverage )
    graph_stats[ '1-shell' ] = np.zeros( nAverage )
    
    for run in tqdm( range( nAverage ) ):
        g = generateSyntheticGraph( n, graph_type ) 
        G = nx.Graph( g.get_edgelist( ) )
                
        graph_stats[ 'n' ][ run ] = g.vcount( )
        graph_stats[ 'm' ][ run ] = g.ecount( )
        graph_stats[ 'average degree' ][ run ] = 2 * g.ecount( ) / g.vcount( )
        graph_stats[ 'diameter' ][ run ] = g.diameter( )
        graph_stats[ 'average shortest path length' ][ run ] = nx.average_shortest_path_length( G )
        graph_stats[ '1-shell' ][ run ] = nx.k_shell(G, k=1).number_of_nodes( )

    statistics_average = dict( )
    statistics_std = dict( )
    
    for stat in [ 'n', 'm', 'average degree', 'diameter', 'average shortest path length', '1-shell' ]:
        statistics_average[ stat ] = np.mean( graph_stats[ stat ] )
        statistics_std[ stat ] = np.std( graph_stats[ stat ] )
        
    return statistics_average, statistics_std
    
    