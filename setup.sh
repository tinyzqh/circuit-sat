cd src
# PyMiniSolvers
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
cd ../
# abc
git clone https://github.com/berkeley-abc/abc.git
cd abc
make
make libabc.a
cd ../
# Aiger
mkdir aiger
cd aiger
wget http://fmv.jku.at/aiger/aiger-1.9.9.tar.gz
tar -xf aiger-1.9.9.tar.gz
mv aiger-1.9.9.tar.gz aiger
rm aiger-1.9.9.tar.gz
cd aiger
./configure.sh && make
cd ../
# cnf2aig
wget http://fmv.jku.at/cnf2aig/cnf2aig-1.tar.gz
tar -xf cnf2aig-1.tar.gz
rm cnf2aig-1.tar.gz
cd cnf2aig
./configure && make

cd ../../..


mkdir data model log