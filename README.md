# Circuit Representation Learning and Its Applications to VLSI Testing 
This repo contains the initial codes and docs for circuit representation learning.

So far, the idea is to transform Conjunctive Normal Form (CNF) into And-Inverter Graph (AIG), then using circuit synthesis tools (e.g., abc) to simplify AIG into the optimized AIG. 
Then our graph neural networks are constructed based on extracted AIG structure. 
In this way, we have two kinds of nodes: AND node and NOT (negative) node. Attention mechanism and heterogeneous graph embedding may be considered further.

For the first phase, let's just try whether building an AIG graph and considering in the circuit structure will help solving SAT or not!

## CNF->AIG->Optimzed AIG
### Libraries
The libraries we need:
1. [abc](https://github.com/berkeley-abc/abc): System for sequential logic system and formal verification;
2. [AIGER](http://fmv.jku.at/aiger/): A format, library and sets of utilities for And-Inverter Graphs (AIGs);
3. [CNFtoAIG](http://fmv.jku.at/cnf2aig/): A converter extracts an AIG in AIGER format from a CNF in DIMACS format;
4. [PyMiniSolvers](https://github.com/liffiton/PyMiniSolvers): a Python API for the MiniSat and MiniCard constraint solvers.

The downloading and installation of these libraries are packed in [setup.sh](setup.sh).

### Workflow
* **Step1**: *PyMiniSovlers* to generate SAT and UNSAT pairs in *dimacs* format, which representing the propositional equations as CNF;
* **Step2**: *CNFtoAIG* to convert the CNF circuits into AIG circuits;
* **Step3**: *ABC* to optimize AIG and output optimized AIG, which is usually be done for synthesis. The optimization process follows the [demo example](https://github.com/berkeley-abc/abc/blob/master/src/demo.c): 1, (Blancing) `balance`; 2, (Synethesis) `balance; rewrite -l; rewrite -lz; balance; rewrite -lz; balance`; 3, (Verification) `ces`; 4, Save AIG `write *.aig`.
* **Step4** (Optional): *aigtoaig* (utilities in *AIGER*) to convert binary AIGER format (\*.aig) into ASCII AIGER (\*.aag) format.
* **Step5** (TO DO): Parse and construct graph representation in PyTorch using generate AIG file.

### Motivation
If AIG representation works, the motivation behind it is quite similar to the one described in [Applying Logic Synthesis for Speeding Up SAT](https://www.researchgate.net/profile/Niklas_Een/publication/220944461_Applying_Logic_Synthesis_for_Speeding_Up_SAT/links/00b7d537cde06c8184000000.pdf). Also, the creator of *abc* also published a paper [Circuit-Based Intrinsic Methods to Detect Overfitting](http://proceedings.mlr.press/v119/chatterjee20a.html), which might be useful later.

## Graph Neural Networks
The exact structure/representation of GNNs for AIG graphs is still to be determined.
I might first consider the intrinsic properties of [AIG](https://en.wikipedia.org/wiki/And-inverter_graph), and their AND nodes/Inverter edges.

Also, C-VAE, [D-VAE](https://github.com/muhanzhang/D-VAE), Attention Mechanism and Heterogeneous Graph Embedding can be referred.

I think using an adjacency matrix to represent the ordering of the nodes and the connections between nodes would be a feasible solution.

### Problem
1. Are Permutation invariance and negation invariance (mentioned in *NeuronSAT*) existing in AIG representation?
2. How to solve SAT problem in a sensible way?

### TO DO
- [ ] Generate SR3to10 optimized AIG dataset.
- [ ] Try small-size problems, *i.e.*, r3to10 problems.




