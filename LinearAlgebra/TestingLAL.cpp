#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <tuple>
#include "matrices.hpp"
//#include <LinearAlgebraLibrary.hpp>

int main() {
    //Creating a dataframe (like those seen in Pandas)
    std::vector<std::vector<double> > normal_Square = {{-6, 0, 4, 3, 3}, {0, 0, 0, 0, 0}, {3, 0, 5, 6, 7}, {8, 0, 2, 9, 1}, {4, 0, 9, 5, 2}};
    std::vector<std::vector<double> > identityMatrix = {{1, 0, 0, 0 }, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    std::vector<std::vector<double> > moreRows_Rec = {{7, 9, 7, 1, 2}, {-2, 6, -2, -2, 10}};
    std::vector<std::vector<double> > moreCols_Rec = {{-3, -5}, {4, -3}, {0, 6}, {0, -1}};
    std::vector<std::vector<double> > scalar_multiple = {{1, 2}, {0, 0}, {1, 2}};
    std::vector<std::vector<double> > linear_combinations = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    std::vector<std::vector<double> > oneCol_All_Zero = {{8, 7, 2, 4}, {0, 0, 0, 0}, {1, 2, -4, -3}, {9, 5, 6, -3}};
    std::vector<std::vector<double> > oneRow_All_Zero = {{3, 0, 2, 4}, {0, 0, 5, 3}, {6, 0, 9, 7}, {6, 0, -1, 7}};
    std::vector<std::vector<double> > mostlyZero = {{0, 2, 0, 0, 0}, {0, 4, 0, 6, 0}, {0, 2, 10, 0, 0}, {0, 0, 0, 8, 0}};
    std::vector<std::vector<double> > mostlyReal = {{-2, 0, 8, 8, 10}, {2, -3, 6, 10, 0}, {-3, 5, 3, 10, 4}, {8, 4, 0, -3, 1}};
    std::vector<std::vector<double> > mostlyReal_square = {{-2, 0, 8, 8}, {2, -3, 6, 10}, {-3, 5, 3, 10}, {8, 4, 0, -3}};
    std::vector<std::vector<double> > symetric_ColsRows = {{-3, -2, 0, -4}, {-3, -2, 0, -4}, {-3, -2, 0, -4}, {-3, -2, 0, -4}};
    std::vector<std::vector<double> > lower_Triangular = {{3, 2, 5, 2}, {0, -4, 2, -3}, {0, 0, 6, 10}, {0, 0, 0, 7}};
    std::vector<std::vector<double> > upper_Triangular = {{8, 0, 0, 0}, {4, 4, 0, 0}, {3, -2, 5, 0}, {-1, 9, 10, 6}};
    std::vector<std::vector<double> > toeplitz = {{1, 2, 3, 4}, {5, 1, 2, 3}, {9, 5, 1, 2}};
    std::vector<std::vector<double> > vandermonde = {{1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}, {0, 1, 4, 9, 16}, {0, 1, 8, 27, 64}, {0, 1, 16, 81, 256}};
    std::vector<std::vector<double> > hilbert = {{1, 0.5, 0.33333, 0.25, 0.20}, {0.5, 0.33, 0.25, 0.20, 0.16667}, {0.33, 0.25, 0.20, 0.16667, 0.142857}, {0.25, 0.20, 0.16667, 0.142857, 0.125}, {0.20, 0.16667, 0.142857, 0.125, 0.11111}};
    std::vector<std::vector<std::vector<double>>> allMatrixHolder = {normal_Square, identityMatrix, moreRows_Rec, moreCols_Rec, scalar_multiple, linear_combinations, oneCol_All_Zero, oneRow_All_Zero, mostlyZero, mostlyReal, mostlyReal_square,symetric_ColsRows, lower_Triangular, upper_Triangular, toeplitz, vandermonde, hilbert};
    std::vector<std::string> allNameHolder = {"normal_Square", "identityMatrix", "moreRows_Rec", "moreCols_Rec", "scalar_multiple", "linear_combinations", "oneCol_All_Zero", "oneRow_All_Zero", "mostlyZero", "mostlyReal", "mostlyReal_square", "symetric_ColsRows", "lower_Triangular", "upper_Triangular", "toeplitz", "vandermonde", "hilbert"};
    
    //Following along with this example: http://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
    std::vector<std::vector<double> > Q_Vector = {{0, 0.80, 0.60}, {-0.80, -0.36, 0.48}, {-0.60, 0.48, -0.64}};
    std::vector<std::vector<double> > D_Vector = {{9, 0, 0}, {0, 4, 0}, {0, 0, 1}};

    matrix Q (Q_Vector); matrix D (D_Vector);
    //matrix A = (Q*D)*Q.inverse();
    std::vector<std::vector<double> > A_Vector = {{2.92, 0.86, -1.15}, {0.86, 6.51, 3.32}, {-1.15, 3.32, 4.57}};
    //std::vector<std::vector<double> > A_Vector = {{-6,4}, {3,5}};
    
    matrix A (A_Vector);
    //std::cout << A.findDeterminant();
    A.eigenvalues();

    //This means A is symmetric and has the eigenvalues 9, 4, and 1
    

    //Doesn't equal 1 in all pivots because of floating point error
    /*
    matrix m1 {df1}; matrix m2 {df2}; matrix m3 {df3};
    matrix m1Inverse = m1.inverse();
    m1.print();
    std::cout << std::endl;
    m1Inverse.print();
    matrix Identitym1 = m1.multiply(m1Inverse);
    std::cout << std::endl;
    Identitym1.print();
    */

    //https://www.youtube.com/watch?app=desktop&v=ShonVncOAB4
    //https://www.youtube.com/watch?v=TQvxWaQnrqI

    //https://www.mathsisfun.com/algebra/eigenvalue.html

    return 0;
}