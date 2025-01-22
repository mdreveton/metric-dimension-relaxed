#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:58:05 2024
"""


import scipy as sp
import igraph as ig

import bigtree as tree


def GaltonWatsonRandomTree( n, distribution_name = 'poisson', mean = 1, max_iter = 1000 ):
    
    iteration = 0
    edge_list = [ ]
    while iteration < max_iter and len( edge_list ) != n-1:
        edge_list = growth_GW( n, distribution_name = 'poisson', mean = mean )
        iteration += 1
    
    if len( edge_list ) != n-1:
        print( 'The branching process has failed to produce a tree with n vertices' )
    
    return ig.Graph( edges = edge_list )


def growth_GW( n, distribution_name = 'poisson', mean = 1 ):
    
    edge_list = [ ]
    vertex_list = [ 0 ]
    aliveVertices = [ 0 ]
    
    while len( aliveVertices ) > 0 and len( vertex_list ) < n:
        
        for aliveVertex in aliveVertices:        
            
            nb_children = discrete_distribution( distribution_name, mean )
            
            for child in range( min( nb_children, n - len( vertex_list ) ) ):
                uniqueId = len( vertex_list ) #This is ok because we start enumerating the vertices from 0
                vertex_list.append( uniqueId )
                aliveVertices.append( uniqueId )
                edge_list.append( ( aliveVertex, uniqueId ) )
                
            aliveVertices.remove( aliveVertex )
            
    return edge_list



    
def discrete_distribution( distribution_name, mean ):
    if distribution_name.lower() == 'geometric':
        return sp.stats.geometric.rvs( 1/ (mean +1), loc = -1 )

    elif distribution_name.lower() == 'poisson':
        return sp.stats.poisson.rvs( mean )
    
    else:
        raise TypeError( 'This distribution is not implemented' )