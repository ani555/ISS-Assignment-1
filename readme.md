# Instructions to compile

### Normal compilation
g++ Matmul.cpp

### With optimization level 2
g++ Matmul.cpp -O2

### With vectorization
g++ Matmul.cpp -O2 -ftree-vectorize -fopt-info-vec