#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <string>
#include <vector>
#include <sstream>  //istringstream
#include <iostream> // cout
#include <fstream>  // ifstream
 
using namespace arma;

float* matToArray(mat m);

mat parseCsvFile(std::string inputFileName, int rows);
