# Circuit Representation Learning and Its Applications to VLSI Testing 
This repo contains the initial codes and docs for circuit representation learning.

So far, the idea is to transform Conjunctive Normal Form (CNF) into And-Inverter Graph (AIG), then using circuit synthesis tools (e.g., abc) to simplify AIG into another AIG. 
Then our graph neural networks are constructed based on extracted AIG structure. 
In this way, we have two kinds of nodes: AND node and NOT (negative) node. Attention mechanism and heterogeneous graph embedding may be considered further.

For the first phase, let's just try whether building an AIG graph and considering in the circuit structure will help solving SAT or not!

## CNF->AIG->AIG
### Libraries
The libraries we need:
1. [abc](https://github.com/berkeley-abc/abc): System for sequential logic system and formal verification;
2. [AIGER](http://fmv.jku.at/aiger/): A format, library and sets of utilities for And-Inverter Graphs (AIGs);
3. [CNFtoAIG](http://fmv.jku.at/cnf2aig/): A converter extracts an AIG in AIGER format from a CNF in DIMACS format;
4. [PyMiniSolvers](https://github.com/liffiton/PyMiniSolvers): a Python API for the MiniSat and MiniCard constraint solvers.

### Workflow
* **Step1**: *PyMiniSovlers* to generate SAT and UNSAT pairs in *dimacs* format, which representing the propositional equations as CNF;
* **Step2**: *CNFtoAIG* to convert the CNF circuits into AIG circuits;
* **Step3** (Optional): *aigtoaig* (utilities in *AIGER*) to convert ASCII AIGER format (\*.agg) in to binary AIGER (\*.aig)format.
* **Step4**: *ABC* to optimize AIG and output optimized AIG, which is usually be done for synthesis.
* **Step5** (TO DO): Parse and construct graph representation in PyTorch using generate AIG file.