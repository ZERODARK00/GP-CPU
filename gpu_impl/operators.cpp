// define the basic calculation of vector and matrix
#include "operators.hpp"

using namespace arma;

float* matToArray(mat m){
    int row = m.n_rows;
    int col = m.n_cols;
    float *arr = new float[row*col];

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            arr[i*row+j] = m(i,j);
        }
    }

    return arr;
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
                }
            }
 
            data.insert_rows(0, record);
        }
    }
 
    return(data);
}
