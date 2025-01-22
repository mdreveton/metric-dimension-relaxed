#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:37:46 2024

@author: dreveton
"""

import itertools
import bigtree as tree
import scipy as sp




def GW( max_generation = 100, name_distribution = 'Poisson', mean = 1 ):
    
    t = tree.Node( "0", generation = 0, alive = 1 )
    generation = 0
    subtrees = 
    while check_if_process_alive( t ):
        generation += 1 
        print( 'Generation : ', generation )
        for leaf in t.leaves:
            if leaf.get_attr( 'alive' ) == 1:
                nb_children = discrete_distribution( name_distribution, mean )
                for child in range(nb_children):
                    subtree = tree.Node( leaf.name + str(child), generation = leaf.get_attr('generation')+1, alive = 1, parent = leaf )
                leaf.set_attrs( {"alive": 0} )
                
    return t


def check_if_process_alive( t ):
    for leaf in t.leaves:
        if leaf.get_attr( 'alive' ) == 1:
            return True
    return False
        



def discrete_distribution( distribution_name, mean ):
    
    if distribution_name.lower( ) == 'geometric':
        return sp.stats.geom.rvs( 1/(1+mean), loc = -1 )

    elif distribution_name.lower( ) == 'poisson':
        return sp.stats.geom.rvs( mean )

    raise TypeError( 'Distribution is not implemented' )