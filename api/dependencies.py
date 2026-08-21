# creates and manages shared resources that FastAPI endpoints need

# a dependency is a function that FastAPI calls automatically before
# running an endpoint - the endpoint receives the dependency's return
# value as a parameter

# dependency injection (Depends()) is necessary when many endpoints need
# a common / shared resource (e.g. database connection)

# why using @lru_cache?
# lru_cache => Least Recently Used Cache
# it makes a function return the same object every time it is called
# instead of creating a new one each time

# lru_cache decorator applied to a function, it caches the result of the first call
# and returns that cached result on all subsequent calls to the function
# perfect for expensive-to-create objects like database connection pools

from functools import lru_cache
from graph.hitl import HITL
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore
from graph.graph_builder import graph
# importing the pre-compiled graph from graph_builder.py
# graph was built once at module import time
# we expose it through a dependency so endpoints can access it cleanly

@lru_cache(maxsize=1)
def get_hitl() -> HITL:
    """
    returns the shared HITL instance
    maxsize=1 means cache the result of the first call
    every subsequent call returns the same HITL object
    the HITL holds a database connection pool - we want
    exactly one pool, not new request everytime
    """

    return HITL()

@lru_cache(maxsize=1)
def get_episodic_store() -> EpisodicStore:
    """
    returns the shared episodic store instance
    """

    return EpisodicStore()

@lru_cache(maxsize=1)
def get_procedural_store() -> ProceduralStore:
    """
    returns the shared procedural store instance
    """

    return ProceduralStore()

@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    """
    returns the shared semantic store instance
    """

    return SemanticStore()

def get_graph():
    """
    return the compiled LangGraph `graph`

    NOT cached with lru_cache since `graph` is
    already a module-level singleton - it was
    compiled once in graph_builder.py and importing
    it here gives the same object everytime
    """

    return graph