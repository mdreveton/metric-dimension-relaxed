#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:56:11 2025
"""



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
    else:
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


def getNonResolvedVerticesFromSetOfSensors( S, g = None, distances = None ):
    identification_vectors = getIdentificationVectors( S, g = g, distances = distances )
    equivalent_classes = getEquivalentClasses( identification_vectors )
    return getNonResolvedVertices( equivalent_classes )
