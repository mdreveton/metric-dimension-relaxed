#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:09:26 2024
"""

import networkx as nx
import igraph as ig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import relaxedMetricDimension as rmd
from sequentialGame import partialResolvingSet

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


"""
G_calls = nx.read_edgelist('datasets/Copenhagen-calls.csv', delimiter=',', data=(('timestamp', int),('duration', int)))
G_fb = nx.read_edgelist('datasets/edges.csv', delimiter=',')


G = getLargestConnectedComponent( G_calls )
nx.is_connected(G_calls)

g = ig.Graph.from_networkx( G )

#######################""


relaxation_values = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16 ]
samplingSize = g.vcount()

fixedCameras = [ ]
extraCameras = [ ]
ratioNonResolvedVertices = [ ]
largestUnresolvedSet = [ ]

for dummy in tqdm( range( len( relaxation_values ) ) ):
    k = relaxation_values[ dummy ]
    
    resolvingSet = rmd.relaxedResolvingSet( g, k )
    fixedCameras.append( len( resolvingSet ) )
    
    nonResolvedSetsOfVertices = rmd.nonResolvedSets( resolvingSet, k, g = g )

    nonResolvedVertices = [ ]
    for elt in nonResolvedSetsOfVertices:
        nonResolvedVertices += elt
       
    ratioNonResolvedVertices.append( len( nonResolvedVertices ) / g.vcount() )
    if len( nonResolvedSetsOfVertices ) == 0:
        largestUnresolvedSet.append( 0 )
    else:
        largestUnresolvedSet.append( np.max( [ len(subset) for subset in nonResolvedSetsOfVertices ] ) )
    
    
    number_ExtraCameras = [ ]
    
    for dummy2 in range( samplingSize ): 
        robber = np.random.randint( 0, g.vcount() )
        if robber not in nonResolvedVertices:
            extra_cameras = [ ]
        else:
            for elt in nonResolvedSetsOfVertices:
                if robber in elt:
                    setOfRobber = elt
                    break
            extra_cameras = partialResolvingSet( g, setOfRobber, print_detailed_running_time = False )
        number_ExtraCameras.append( len( extra_cameras ) )
    extraCameras.append( np.mean( number_ExtraCameras ) )


fileName = 'calls' 
savefig = False
plotFigure( relaxation_values, fixedCameras, ylabel = "$MD_k$", savefig = savefig, fileName = fileName + '_MDk' + '.pdf' )
plotFigure( relaxation_values, np.asarray( fixedCameras ) + np.asarray( extraCameras ), ylabel = "Total sensors", savefig = savefig, fileName = fileName + '_totalNumberSensors' + '.pdf' )
plotFigure( relaxation_values, largestUnresolvedSet, ylabel = "Non-resolved set", savefig = savefig, fileName = fileName + '_sizeLargestNonResolvedSet.pdf' )
plotFigure( relaxation_values, ratioNonResolvedVertices, ylabel = 'Ratio non-resolved', savefig = savefig, fileName = fileName + '_ratioNonResolvedVertices.pdf' )



"""

def getLargestConnectedComponent(G):
    # Remove the small components such that the graph becomes connected
    connected_components = nx.connected_components(G)
    biggest = max(connected_components, key=len)
    largestComponent = G.subgraph(biggest).copy()
    largestComponent.remove_edges_from(nx.selfloop_edges( largestComponent ) )
    return largestComponent


def plotFigure( x, accuracy_mean, methods = None, 
               xticks = None, yticks = None,
               xlabel = "k", ylabel = "Relative size",
               savefig = False, fileName = "fig.pdf" ):
    
    plt.plot( x, accuracy_mean, linestyle = '-.' )
    
    plt.xlabel( xlabel, fontsize = SIZE_LABELS)
    plt.ylabel( ylabel, fontsize = SIZE_LABELS)
    
    if xticks != None:
        plt.xticks( xticks, fontsize = SIZE_TICKS)
    else:
        plt.xticks( fontsize = SIZE_TICKS)
    
    if yticks != None:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    else:
        plt.yticks( fontsize = SIZE_TICKS )


    if(savefig):
        plt.savefig( fileName, bbox_inches='tight' )
    plt.show( )

