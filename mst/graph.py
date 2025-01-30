import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        For more information on Prim's algorithm, see the following link: 
        https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/

        """
        ## pseudocode from class slide
        # PRIM(V, E, c)
        # _______________________________________________________________
        # S ← ∅, T ← ∅.
        # s ← any node in V.
        # FOREACH v ≠ s : π [v] ← ∞ , pred[v] ← null; π [s] ← 0.
        # Create an empty priority queue pq.
        # FOREACH v ∈ V : INSERT(pq, v, π[v]).
        # WHILE (IS-NOT-EMPTY(pq))
        #   u ← DEL-MIN(pq).
        #   S ← S ∪ { u }, T ← T ∪ { pred[u] }.
        #   FOREACH edge e = (u, v) ∈ E with v ∉ S :
        #       IF (ce < π [v])
        #           DECREASE-KEY(pq, v, ce).
        #           π [v] ← ce; pred[v] ← e.
        
        ## Extran note from class
        # ・[note] π[v] = cost of cheapest known edge between v and S

        ## Step by step explanation
        # - We transform the adjacency matrix into adjacency list using list of list in Python
        # - Then we create a Pair class to store the vertex and its weight .
        # - We sort the list on the basis of lowest weight.
        # - We create priority queue and push the first vertex and its weight in the queue
        # - Then we just traverse through its edges and store the least weight in a variable called ans.
        # - At last after all the vertex we return the ans.

        ## Code implementation
        V, E = self.adj_mat.shape
        edges = self.adj_mat
        adj = [[] for _ in range(V)]
        for i in range(E):
            for j in range(V):        
                if edges[i][j] != 0:
                    v = int(edges[i][j])
                    adj[i].append((j, v))
        
        pq = []
        visited = [False] * V
        res = 0 # Sum of the edge weights
        parent = [None] * V  # Array to store constructed MST
        heapq.heappush(pq, (0, 0))

        parent[0] = -1

        # Perform Prim's algorithm to find the Minimum Spanning Tree
        while pq:
            wt, u = heapq.heappop(pq)
            if visited[u]:
                continue
            res += wt
            visited[u] = True
            for v, w in adj[u]:
                if not visited[v]:
                    heapq.heappush(pq, (w, v)) # Add the adjacent edge to the priority queue
                    parent[v] = u # Store the parent of the vertex

        mst = np.zeros((V, V))
        for i in range(1, V):
            mst[i][parent[i]] = edges[i][parent[i]]
            mst[parent[i]][i] = edges[parent[i]][i]
        
        self.mst = mst
