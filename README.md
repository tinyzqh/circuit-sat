# Learning to Solve Circuit-SAT: an Unsupervised Differentiable Approach
This is the re-implementation of [Circuit-SAT](https://openreview.net/forum?id=BJxgz2R9t7). The implementation is not exactly the one from the original paper, with respect to graph specification and deep-set function. More details will be supplemented later.

## Pre-Processing: CNF->AIG->Optimzed AIG
### Libraries
The libraries we need:
1. [abc](https://github.com/berkeley-abc/abc): System for sequential logic system and formal verification;
2. [AIGER](http://fmv.jku.at/aiger/): A format, library and sets of utilities for And-Inverter Graphs (AIGs);
3. [CNFtoAIG](http://fmv.jku.at/cnf2aig/): A converter extracts an AIG in AIGER format from a CNF in DIMACS format;
4. [PyMiniSolvers](https://github.com/liffiton/PyMiniSolvers): a Python API for the MiniSat and MiniCard constraint solvers.

The downloading and installation of these libraries are packed in [setup.sh](setup.sh).

### Workflow
* [**Step1**](scripts/gen_dimacs.sh): *PyMiniSovlers* to generate SAT and UNSAT pairs in *dimacs* format, which representing the propositional equations as CNF;
* [**Step2**](scripts/dimacs2aig.sh): *CNFtoAIG* to convert the CNF circuits into AIG circuits;
* [**Step3**](scripts/aig2aigabc.sh): *ABC* to optimize AIG and output optimized AIG, which is usually be done for synthesis. The optimization process follows the [demo example](https://github.com/berkeley-abc/abc/blob/master/src/demo.c): 1, (Balancing) `balance`; 2, (Synthesis) `balance; rewrite -l; rewrite -lz; balance; rewrite -lz; balance`; 3, (Verification) `ces`; 4, Save AIG `write *.aig`. I assume the networks before and after synthesis are equivalent.
* [**Step4**](scripts/aig2aigabc.sh): *aigtoaig* (utilities in *AIGER*) to convert binary AIGER format (\*.aig) into ASCII AIGER (\*.aag) format.
* [**Step5**](scripts/aigabc2pyG.sh): Parse and construct graphs in  [PyGeometric](https://github.com/rusty1s/pytorch_geometric) format with generated AIG circuits.

All steps can be done using [bash scripts](scripts/data_gen.sh).


## Graph Neural Networks

### Graph Specification
For AIG, the nodes can be categorized as the literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
* Literal input node: 1 (input nodes, have a common predecessor Input node);
* Internal AND nodes: 2;
* Internal NOT nodes; 3;


### GNN Design
I try to implement to GNN functional as the same as the DG-DARGNN as possible, but there's some implementation difference.

1. Graph nodes: There's no OR gate type for now, cause I use AIG to represent the circuits.
2. Solver Network:
   * GCN: 1 (q-100D) forward GRU layer followed by 1 (q-100) reversed GRU layer. 
   * The embedding computation is iteratively run for T Times. A FC mapping function to do it (100D-to-3D). May be modified later.
   * Aggregation function: gated sum and average function. Not Deep-Set function so far.
   * Classifer: 100D-30D-1-Sigmoid function. Read off the final outputs embedding from the Literal nodes.
3. Evaluator Network:
   *  AND -> smooth min function
   *  NOT -> 1-z functionm
4. Loss Function:
   *  SmoothStep function.





