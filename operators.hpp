#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
 
using namespace arma;

// formulation is $\sum_{AB}$
mat covariance(const mat A, const mat B, float (*Kernel)(mat A, mat B));

//formulation is $\sum_{AA|B}$
mat conditional_covariance(const mat A, const mat B, float (*Kernel)(mat A, mat B));

mat parseCsvFile(std::string inputFileName, int rows);