# Circuit Representation Learning and Its Applications to VLSI Testing 
This repo contains the initial codes and docs for circuit representation learning.

So far, the idea is to transform CNF into AIG, then using circuit synthesis tools (e.g., abc) to simplify AIG into another AIG. 
Then our graph neural networks are constructed based on extracted AIG structure. 
In this way, we have two kinds of nodes: AND node and NOT (negative) node. Attention machinism and heterogeneous graph embedding may be considered further.

For the first phase, let's just try whether building a AIG graph and considering in the circuit structure will help solving SAT or not!
