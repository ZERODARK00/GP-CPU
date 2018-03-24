// define the basic calculation of vector and matrix
#include "operators.hpp"
using namespace arma;

// formulation is $\sum_{AB}$
mat covariance(const mat A, const mat B, float (*Kernel)(mat A, mat B)){
    // the inputs all have the shape of (samples, features)
    int A_samples = A.n_rows;
    int B_samples = B.n_rows;
    mat C = zeros<mat>(A_samples, B_samples);
    for(int i=0; i<A_samples; i++){
        for(int j=0; j<B_samples; j++){
            C(i, j) = Kernel(A.row(i), B.row(j));
        }
    }
    return(C);
}

// //formulation is $\sum_{AA|B}$
mat conditional_covariance(const mat A, const mat B, float (*Kernel)(mat A, mat B)){
    // the inputs all have the shape of (samples, features)
    mat AA = covariance(A, A, Kernel);
    mat AB = covariance(A, B, Kernel);
    mat BB = covariance(B, B, Kernel);
    mat BA = covariance(B, A, Kernel);
    mat C = AA - AB * inv(BB) * BA;
    return(C);
}

mat parseCsvFile(std::string path, int rows) {
 
    mat data;
    char inputFileName[20];
    strcpy(inputFileName, path.c_str());
    std::ifstream inputFile(inputFileName, std::ifstream::in);
    int l = 0;
 
    while (l < rows) {
        l++;
        std::string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            std::istringstream ss(s);
             mat record;
 
            while (ss) {
                std::string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    mat col = ones<mat>(1,1)* stof(line);
                    record.insert_cols(0, col);
                }
                catch (const std::invalid_argument e) {
                    mat col = ones<mat>(1,1)* 0;
                    record.insert_cols(0, col);
                    // cout << "NaN found in file " << inputFileName << " line " << l
                    //      << endl;
                    // e.what();
                }
            }
 
            data.insert_rows(0, record);
        }
    }
 
    // if (!inputFile.eof()) {
    //     cerr << "Could not read file " << inputFileName << "\n";
    //     std::__throw_invalid_argument("File not found.");
    // }
 
    return(data);
}