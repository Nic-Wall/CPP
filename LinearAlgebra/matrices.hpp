#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <tuple>

class matrix {
//https://www.learncpp.com/cpp-tutorial/introduction-to-constructors/
    private:
    //Private variables and functions
        int numCols;
        //Count of columns (not including index)
        double numRows = -1;
        //Count of rows (not including column headers)
        std::vector<std::vector<double> > matrice;
        //The actual matrix, storing all the row and column data (including index and column headers)
        double determinant;
        //Double that holds the determinant of the matrix (if =0 the matrix has no inverse)
        bool determinantSolved = false;
        //Keeps track of if the determinant has been solved, must turned to true and false accordingly when the matrix is changed
        int rows_swapped = 0;
        //Rows swapped
        std::vector<double> eigenValue_Vector;

    //Ensures the columns and rows are of equal proportions (for addition and subtraction)
        void rowColComparing(matrix& M, bool add_and_sub) {
            const std::vector<std::string> addSub_or_multDiv = {"must be equal for addition and subtraction", "must be inversed for multiplication and division"};
            if(((numRows != M.rowCount() || numCols != M.colCount()) && add_and_sub) || ((numRows != M.colCount() || numCols != M.rowCount()) && !add_and_sub)) {
            //If the number of rows are not equal or the number of columns are not equal the program ends here 
            std::cout << "Column count and row count of all matrices " << addSub_or_multDiv[add_and_sub] << ".\nThe first provided matrix has a row size of " << numRows << " and a column size of " 
                << numCols << ", while the matrix to be added has a row size of " << M.rowCount() << " and a column size of " << M.colCount() << "." << std::endl; throw std::exception();
            //Warns the user and exits the function
            }
        }
    //Simply returns a column vector based on the index notation entered
        std::vector<double> getColumn(const int& num) {
            return matrice[num];
        }
    //Simply returns a vector of the row requested based on the index notation provided
        std::vector<double> getRow(const int& num) {
            std::vector<double> row;
            for(int i= 0; i < numCols; i++) {
                row.push_back(matrice[i][num]);
            }
            return row;
        }
    //
        void accumulate(matrix M) {
                this->rowColComparing(M, true);
                //Checks to ensure the number of rows and columns are equal between the matrices
                for(int i= 0; i < numCols; i++) {
                //Itterates through columns
                    for(int j= 0; j < numRows; j++) {
                    //Itterates through rows
                        matrice[i][j] += M.locateValue(j, i);
                        //Adds the coordinated colummn, row cell found in the supplied matrix and adds it to the column, row cell located in the dataframe that which add was called on
                    }
                }
            }
    //GramScmidt process performed on vectors
        matrix GramSchmidt() {
        //https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
        //https://www.statlect.com/matrix-algebra/Gram-Schmidt-process
            //Create new vectors to hold the soon to be returned Q and R
            std::vector<std::vector<double> > returningQ;
            //Normalize the vector with (1/||s1||) * s1 this returns u1 (double vector)
            returningQ.push_back(vector_multiplication(matrice[0],(1/vector_norm(matrice[0])))); //Acquires u1 or the first orhtonormal vector
            //u1 is used in all subsequent vector calculations
            for(int i= 1; i < numRows; i++) {
                //Project si based on vector_inner_product(si, u1) * u1 OR ^si = <si, u1> * u1
                std::vector<double> si_residual (numCols); //Generating a 0 filled vector the same size as all other rows

                for(int j= i-1; j >= 0; j--) {  //Itterating through every previously normalized residual already existing in returningQ
                    std::vector<double> to_add_to_si = vector_multiplication(returningQ[j], vector_inner_product(matrice[i], returningQ[j]));
                    si_residual = vector_subtraction(si_residual, vector_multiplication(to_add_to_si, double(-1)));
                }
                si_residual = vector_subtraction(matrice[i], si_residual);
                returningQ.push_back(vector_multiplication(si_residual, (1/(vector_norm(si_residual)))));
            }
                return matrix (returningQ);
        }
    //Vector normalization
        double vector_norm(const std::vector<double>& V) {
            double sum_of_vector_multiples = 0;
            for(int i= 0; i < V.size(); i++) {
                sum_of_vector_multiples += (V[i] * V[i]);
            }
            return sqrt(sum_of_vector_multiples);
        }
    //Vector inner product
        double vector_inner_product(const std::vector<double>& V1, const std::vector<double>& V2) {
            if(V1.size() != V2.size() || V1.size() <= 0 || V2.size() <= 0) {
                std::cout << "Input vector sizes must be equal and greater than zero to calculate the inner product." << std::endl; throw std::exception();
            }
            double inner_product = 0;
            for(int i= 0; i < V1.size(); i++) {
                inner_product += (V1[i] * V2[i]);
            }
            return inner_product;
        }
    //Vector multiplication
        std::vector<double> vector_multiplication(const std::vector<double>& V, const double& N= 1) {
            if(V.size() <= 0) {
                return std::vector<double> {};
            }

            std::vector<double> returningV;
            for(int i= 0; i < V.size(); i++) {
                returningV.push_back(V[i] * N);
            }
            return returningV;
        }
    //Vector subtraction
        std::vector<double> vector_subtraction(const std::vector<double>& V1, const std::vector<double>& V2) {
            if(V1.size() <= 0 || V2.size() <= 0) {
                std::cout << "Input vector sizes must be greater than zero to subtract." << std::endl; throw std::exception();
            }
            std::vector<double> returningV;
            for(int i= 0; i < V1.size(); i++) {
                returningV.push_back(V1[i]-V2[i]);
            }
            return returningV;
        }
    //Determing the relative distance of two rows compared to a supplied tolerance (std::vector<double>)
        bool toleranceMeasure(const std::vector<double>& AQ, const std::vector<double>& Prev, const double& tolerance = 0.0001) {
            for(int i= 0; i < AQ.size(); i++) {
                if(fabs((AQ[i] - Prev[i])) > tolerance) {
                    return false;
                }
            }
            return true;    //If all elements in the vectors are close enough together given the tolerance
        }
    //Resetting the determinant and rows changed when portions of the matrix are added, removed, or changed
        void reset_determinant_calc() {
            determinantSolved = false;
            rows_swapped = 0;
            eigenValue_Vector = {};
        }

    //Transposes a vector<vector<double> >  (swaps the rows and columns)
        std::vector<std::vector<double> > transposeVector(const std::vector<std::vector<double> >& matriceToTranspose) {
            std::vector<std::vector<double> > returningVector (matriceToTranspose[0].size());
            for(int i= 0; i < numCols; i++) {
                for(int j= 0; j < numRows; j++) {
                    returningVector[j].push_back(matriceToTranspose[i][j]);
                }
            }
            return returningVector;
        }
    //Returns either upper or lower triangular matrices based on the boolean provided
        matrix upper_or_lower_triangular_matrices(bool upper = true) {
            /*
            MUST meet two conditions:   https://www.youtube.com/watch?app=desktop&v=ShonVncOAB4
            1. Any row of zeros MUST be at the bottom
            2. The leading entry in a row MUST be to the right of the leading in the row above it
            */
            std::vector<std::vector<double>> I_Copy = (this->identityMatrix()).vectorCopy();    //Only used if upper == false
            std::vector<std::vector<double> > matrixCopy = this->vectorCopy();
            if(numRows == 1 && numCols == 1) {
            //Arbitrary number, so it just returns itself
            }
            else {
            //2x2 matrices to nxn matrices
                std::tuple<int, int> pivot = {-1, 0};   //Making a variable to keep track of the current pivot
                for(int i= 0; i < numCols; i++) {
                //Iterating through the columns of the matrix
                    bool foundPivot = false;    //Marking the pivot finder variable as false, so the while loop to find said pivot starts in the next loop
                    for(int j= std::get<0>(pivot)+1; j < numRows; j++) {
                    //Iterating through the rows of a matrix starting on the row directly after the last pivot (hence why pivot is initialized as -1,0 instead of 0,0)
                        while((i < numCols) && !foundPivot) {
                        //So long as i isn't greater than the number of columns AND a pivot hasn't already been determined...
                        //NOTE: This will run every time the i loop iterates
                            pivot = {j, i}; //Set the pivot to it's default location (usually i,i (0,0 , 1,1 , etc.))
                            if(matrixCopy[std::get<1>(pivot)][std::get<0>(pivot)] == 0) {   //If the selected pivot is equal to 0
                                int rowIter = j+1;  //Create a tracker to watch the rows below
                                bool swappableRow = false;  //Create a tracker to determine if a row is found
                                while((swappableRow == false) && (rowIter < numRows)) {
                                //While no swappable row has been found (i.e. the row underneath is also 0) and rowIter hasn't surpassed the number of rows...
                                    if(matrixCopy[std::get<1>(pivot)][rowIter] != 0) {
                                    //If a row under the first elected cell for a pivot isn't equal to 0
                                        swappableRow = true;    //it can be swapped and the while loop should break
                                    }
                                    else {
                                    //... else, check the next row by iterating the while loop
                                        rowIter += 1;
                                    }
                                }
                                if(swappableRow == true) {
                                //A swappable row has been found so use rowIter to swap the rows
                                    for(int k= i; k < numCols; k++) {
                                    //Assuming all columns before have already been made to equal 0 the loop can start at i instead of at the beginning
                                        std::swap(matrixCopy[k][j], matrixCopy[k][rowIter]);
                                        //Swapping the row with the 0 pivot to the row with the non-zero in the pivot position
                                    }
                                    foundPivot = true;  //Breaking the while loop by confirming a new pivot has been found
                                    rows_swapped += 1;  //Keeping track of the rows swapped to ensure an accurate determinant
                                }
                                else {
                                //Must move pivot over one column (if possible) because all possible options in the previous columns were 0's 
                                    if(i == numCols-1) {
                                    //If this is the last column to be evaluated
                                        i = numCols;
                                        j = numRows;
                                        //Setting i to numCols and j to numRows so the main loops complete
                                    }
                                    else {
                                        i += 1; //Moving the selected column over one to check for the next pivot point
                                    }
                                }
                            }
                            else {
                            //If the first checked pivot of the while loop isn't == 0, mark foundPivot as true and break the while loop
                                foundPivot = true;
                            }

                            if(foundPivot == true) {
                            //If a pivot has been found ensure the coordinates are makred appropriately
                                pivot = {j, i};
                            }
                        }

                        //Pivot is establish above
                        //Transforming the rows below the pivot
                        if((j != std::get<0>(pivot) ) && (i < numCols) && (j < numRows)) {
                        //So long as the last column of the matrix hasn't been checked by the pivot finder above...
                        //and j is not equal to 0, because the first row stays as it is when evaluating
                            for(int k= numCols-1; k >= std::get<1>(pivot); k--) {
                                if((k == std::get<1>(pivot)) || ((matrixCopy[k][j] == matrixCopy[std::get<1>(pivot)][j]) && (matrixCopy[k][std::get<0>(pivot)] == matrixCopy[std::get<1>(pivot)][std::get<0>(pivot)]))) {
                                //If this row is under the pivot OR the cell in the same row as the pivot is equal to the pivot AND the cell being transformed is the same as the one directly under the pivot...
                                    matrixCopy[k][j] = 0;
                                    //Ensuring those underneath the pivot (and those that mimic the pivot equation) are 0 instead of an underflow floating point
                                    //Also saves time/ processing power, since no calculation is necessary
                                }
                                else {
                                //Perform the calculation to determine the value of the transformed row
                                matrixCopy[k][j] = matrixCopy[k][j] - 
                                                   (matrixCopy[k][std::get<0>(pivot)] * 
                                                   (matrixCopy[std::get<1>(pivot)][j] / matrixCopy[std::get<1>(pivot)][std::get<0>(pivot)]));
                                }
                                if(upper == false) {
                                        I_Copy[k][j] = matrixCopy[k][j] / matrixCopy[std::get<1>(pivot)][std::get<0>(pivot)];
                                }
                            }
                        }
                    }
                }
            }
            
            if(upper == false) {return matrix(I_Copy);} //Returning the 

            return matrix(matrixCopy);  //Returning the matrix in row echelon form
        }

    public:
    //Constructor: Building the matrix class
        matrix(std::vector<std::vector<double> > V = {{}}, bool columns_in_vector = true) {
        //Creating a matrix that will be used in all future linear algebra functions
            matrice = V;
            //V = {{col}, {col}, {col}, etc.}
            numCols = matrice.size();
            for(int i= 0; i < numCols; i++) {
                while(matrice[i].size() < numRows) {
                    matrice[i].push_back(double(0));
                }
                if(matrice[i].size() > numRows) {
                    numRows = matrice[i].size();
                    i = 0;
                }
            }
            
//Test this
            if(!columns_in_vector) {
            //V = {{row}, {row}, {row}, etc.}
                matrice = transposeVector(matrice);
                numCols = numCols + numRows;
                numRows = numCols - numRows;
                numCols = numCols - numRows;
            }
        }

    //Simply prints the number of rows and columns in the matrix, not including the column headers or index. Formatted as "(numRows, numCols)"
        std::string shape() const {
        //Printing (count of rows, count of columns columns)
            return std::string ("(" + std::to_string(int(numRows)) + "," + std::to_string(numCols) + ")");
        }

    //Getting numRows and numCols
        int rowCount() {
            return numRows; //Simply returning the numRows variable
        }
        int colCount() {
            return numCols; //Simply returning the numCols variable
        }

    //Print
        void print(int rowsToPrint = 0) {
        //Printing the matrix
            using std::left;
            if((rowsToPrint <= 0) || (numRows < rowsToPrint)) {
            //
                rowsToPrint = numRows;
                //
            }

            for(int i= 0; i < rowsToPrint; i++) {
            //Loops as many times as the number of rows (i.e. the number of vectors present as values in the map dataFrame)
                for(int j= 0; j < numCols; j++) {
                //Loops as many times as the number of columns (i.e. the length of each vector designated as a value in the map dataFrame). Will loop one less time if withIndexAndCol is marked as False (skipping the index column)
                    std::cout << std::setw(5) << matrice[j][i];
                    std::cout << " ";
                    //Prints the singular output of matrice[row][column]
                }
                std::cout << std::endl;
                //Breaks the line
            }
            using std::right;
            std::cout << std::endl;
        }

    //Transposition: Turns the columns into rows and rows into columns
        matrix transpose() {
            //By this point all the rows and columns should be the same size, as is set during the matrix creation (filled in blanks with 0's)
            std::vector<std::vector<double> > returningMatrix; //Setting the number of columns equal to the number of rows
            returningMatrix = transposeVector(matrice);
            return matrix(returningMatrix);            
        }

    //Add a row (axis = 0 = false)
    //Add a column (axis = 1 = true)
        void addAxis(std::vector<double> rowOrCol, int position = -1, bool axis = true) {
        //Function for adding single row (called on by the overloaded version of addAxis to add the individual rows in the vector parameter provided)
            if(!axis) {
                matrice = transposeVector(matrice);
                numCols = numCols + numRows;
                numRows = numCols - numRows;
                numCols = numCols - numRows;
            }
            
            if(rowOrCol.size() > numRows) {
            //If the added column/ row is larger than the existing columns/ rows
                for(int i= 0; i < numCols; i++) {
                    while(matrice[i].size() < rowOrCol.size()) {
                    //
                        matrice[i].push_back(double(0));
                    }
                }
            }
            if(rowOrCol.size() < numRows) {
            //If the added column/ row is smaller than the existing columns/ rows
                while(rowOrCol.size() < numRows) {
                //
                    rowOrCol.push_back(double(0));
                }
            }
                
            matrice.insert(matrice.begin() + position, rowOrCol);
            numCols += 1;
            
            if(!axis) {
                matrice = transposeVector(matrice);
                numCols = numCols + numRows;
                numRows = numCols - numRows;
                numCols = numCols - numRows;
            }
            reset_determinant_calc();
        }
        void addAxis(std::vector<std::vector<double> > listRowOrCol, int position = -1, int axis = true) { //Overloaded
        //Overloaded function allowing for vectors of vectors in the parameter (i.e. a list of rows/ columns the user wants added)
            for(int i= 0; i < listRowOrCol.size(); i++) {
                addAxis(listRowOrCol[i], position+i,axis);
            }
        }
    
    //Addition (allow for the entry of multiple matrices (more than two))
        matrix add(matrix M) {
        //Adds two matrices together
            this->rowColComparing(M, true); //Checks to ensure the column and row sizes are the same

            std::vector<std::vector<double> > addedM (numCols);  //Map making up the new matrix, created with col size of current matrix
            for(int i= 0; i < numCols; i++) {
            //Iterates through number of columns
                for(int j= 0; j < numRows; j++) {
                //Itterates through number of rows
                    addedM[i].push_back(matrice[i][j] + M.locateValue(j, i));
                    //Adds the two matrices positions together and pushes it into the vector associated with the column (based on column name returned from the vector and index j)
                }
            }
            return matrix(addedM);
            //Returns the matrix
        }
        matrix add(std::vector<matrix> V) { //Overloaded
            if(V.empty() || matrice.empty()) {
            //Checks to ensure the supplied vector isn't empty
                std::cout << "The vector entered into add must contain matrices, it's current size is zero."; throw std::exception();
            }
            matrix resultingMatrix = this -> add(V[0]);
            //Creating a new matrix based on the addition of whichever matrix was used to call this method and the first matrix in the vector supplied
            for(int i= 1; i < V.size(); i++) {
            //Itterating for every matrix in the vector supplied
                resultingMatrix.accumulate(V[i]);
                //resultingMatrix = resultingMatrix.add(V[i]);
                //Adding the Matrix calculated with the original matrix and V[0] with every other matrix in the vector supplied
            }
            return resultingMatrix;
            //Returns the added matrix
        }
        matrix operator +(const matrix& M) { //Overloaded
            matrix calledMatrix (*this);
            return calledMatrix.add(M);
        }
        matrix operator +(const std::vector<matrix>& V) {//Overloaded
            matrix calledMatrix (*this);
            return calledMatrix.add(V);
        }

    //Pull the value of a cell from a specific location
        double locateValue(int R, int C) {  //Overloaded
        //Finds the column header (map key) associated with the index in colOrder
            if(C > numCols+1 || C < (-1*(numCols))-1) {
                std::cout << "The column number " << C << "does not exist in the matrix. Please use a column between 0 and" << numCols+1 << "."; throw std::exception();
            }
            return matrice[C][R];
        }
        
    //Multiplication (occurs only if the num of columns in the left matrix is the same as the num rows of the right) (allow for multiplication with single digits)
        matrix multiply(matrix M) {
            //ROW1xCOL1 * ROW2xCOL2 = ROW1xCOL2
            //MxN * NxP = MxP
            //So N's must be equal

            if(numCols != M.rowCount()) {
                std::cout << "Matrice multiplication must match the following: MxN * NxP = MxP. If the first matrix's row size does not match the second matrix's column size multiplication is not possible. " <<
                    "\nPlease use matrix.transpose() to switch row size to column size and vice versa before attempting to multiply again." << std::endl; throw std::exception();
            }

            std::vector<std::vector<double> > returningVV(M.colCount());

            for(int i= 0; i < numRows; i++) {
            //Iterattes through as many times as there are rows in dataFrame (the this matrix)
                for(int j= 0; j < numCols; j++) {
                //Iterattes through as many times as there are columns in the dataFrame
                    for(int k= 0; k < M.colCount(); k++) {
                    //Iterattes through as many times as there are columns in the M matrix
                        if(returningVV[k].size() < numRows) {
                        //If the returning matrix already has enough rows (as determined by the column size of this (dataFrame)), no more rows are created
                            returningVV[k].push_back(0);
                        }
                        returningVV[k][i] += matrice[j][i] * M.locateValue(j,k);
                        //Adds the values of dataFrame[col j][row i] * M[row j][col k] to the cell returningVV[col k][row i]
                    }
                }
            }

            return matrix(returningVV);
        }
        matrix multiply(double num) {
            std::vector<std::vector<double> > returningVV (numCols);
            //Creating a vector within a vector for the eventual creation of a new matrix
            for(int i= 0; i < numRows; i++) {
            //Itterates through each row
                for(int j= 0; j < numCols; j++) {
                //Itterates through each column (starts at 1, so the dataFrame (already existing matrix) has their first column referenced rather than their index)
                    returningVV[j].push_back(matrice[j][i] * num);
                    //Adds the multiplied value to the empty (but growing) returningVV variable
                }
            }
            matrix returningM {returningVV};
            //Turns returningVV into a matrix
            return returningM;
            //Returns matrix
        }
        matrix multiply(int num) {
            return this->multiply(static_cast<double>(num));
        }
        std::vector<double> multiply(std::vector<double> V) {
            std::vector<double> returningV (numCols);
            for(int i= 0; i < numRows; i++) {
                returningV[i] = 0;
                for(int j= 0; j < numCols; j++) {
                    returningV[i] += matrice[j][i] * V[i];
                }
            }
            return returningV;
        }
        matrix operator *(const matrix& M) {
            matrix calledMatrix (*this);
            return calledMatrix.multiply(M);
        }
        matrix operator *(const double N) {
            matrix calledMatrix (*this);
            return calledMatrix.multiply(N);
        }
        std::vector<double> operator *(const std::vector<double> V) {
            matrix calledMatrix (*this);
            return calledMatrix.multiply(V);
        }
    
//Inversion (only square matrices have inverses, though rectangular matrices can have left/right inverses)
        matrix inverse() {
        /*
            1. Can't have a determinant of 0
            2. 2x2
            3. 3x3 and larger using minors, cofactors, and adjugate
                3.a. Calculate the matrix of minors (determinant of every possible sub-matrix in the matrix)
                3.b. Matrix of cofactors (multiply the determinent by 1 or -1 depending on it's position in the matrix)
                3.c. Adjugate (flip it along the right to left diagonal line)
                3.d. Multiply the matrix by 1/OriginalDeterminant
                https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
            
        */
            double originalDeterminant = this->findDeterminant();
            if(originalDeterminant == 0) {
            //A matrix with a determinant of 0 has no inverse
                return matrix(std::vector<std::vector<double> > {});
                //Returns an empty matrix indicating there is no inverse of the submitted matrix
            }
            else {
            //Any other square matrix with a determinant other than 0 passes through
                std::vector<std::vector<double> > thisMatrix;
                std::vector<std::vector<double> > determinantsOfMiniMatrices (numCols);
                for(int i= 0; i < numCols; i++) {
                    determinantsOfMiniMatrices[i] = std::vector<double> (numRows);
                    thisMatrix.push_back(matrice[i]);
                }

                //Matrix of minors and cofactors (done in one to limit the number of nested loops)
                if(numCols == 2 && numRows == 2) {
                    determinantsOfMiniMatrices = thisMatrix;
                    determinantsOfMiniMatrices[0][1] *= -1;
                    determinantsOfMiniMatrices[1][0] *= -1;
                }
                else {
                    bool PosCofactorCol = true;
                    for(int i= 0; i < numCols; i++) {
                        bool PosCofactorRow = PosCofactorCol;
                        double cofactor;
                        for(int j= 0; j < numRows; j++) {
                            std::vector<std::vector<double> > adjustableVecCopy = thisMatrix;
                            adjustableVecCopy.erase(adjustableVecCopy.begin()+i);   //Erases the column
                            for(int k= 0; k < numCols-1; k++) {
                                adjustableVecCopy[k].erase(adjustableVecCopy[k].begin()+j); //Erases the row
                            }
                            !PosCofactorRow ? cofactor = -1 : cofactor = 1;
                            determinantsOfMiniMatrices[i][j] = (matrix(adjustableVecCopy)).findDeterminant() * cofactor;
                            PosCofactorRow = !PosCofactorRow;
                        }
                        PosCofactorCol = !PosCofactorCol;
                    }
                }

                //Adjugate (Adjoint)
                if(numCols == 2 && numRows == 2) {
                    std::swap(determinantsOfMiniMatrices[0][0], determinantsOfMiniMatrices[1][1]);
                }
                else { 
                    for(int i= 0; i < numCols; i++) {
                        int rowNum = 0;
                        while(rowNum < i) {
                            std::swap(determinantsOfMiniMatrices[i][rowNum], determinantsOfMiniMatrices[rowNum][i]);
                            rowNum++;
                        }
                    }
                }

                //Multiply by 1/Determinant
                return matrix(determinantsOfMiniMatrices) * (1/originalDeterminant);
            }
        }

    //Subtraction (is easy because; A - B = A + (-1) B, where A and B are matrices)
        matrix subtract(matrix M) {
            matrix returningMatrix = this->add(M.multiply(-1));
            return returningMatrix;
        }
        matrix subtract(std::vector<matrix> V) {
            if(V.empty()) {
            //Checks to ensure the supplied vector isn't empty
                std::cout << "The vector entered into add must contain matrices, it's current size is zero."; throw std::exception();
            }
            matrix returningMatrix = this-> subtract(V[0]);
            if(V.size() > 1) {
                for(int i= 1; i < V.size(); i++) {
                    returningMatrix.accumulate(V[i].multiply(double(-1)));
                }
            }
            return returningMatrix;
        }
        matrix operator -(const matrix& M) {
            matrix returningMatrix = this->subtract(M);
            return returningMatrix;
        }
        matrix operator -(const std::vector<matrix>& V) {
            matrix returningMatrix = this->subtract(V);
            return returningMatrix;
        }
    
    //Making a vector copy (not pointer) of the called on matrix
        std::vector<std::vector<double> > vectorCopy() {
            std::vector<std::vector<double> > thisCopy (numCols);
            for(int i= 0; i < numCols; i++) {
                for(int j= 0; j < numRows; j++) {
                    thisCopy[i].push_back(matrice[i][j]);
                }
            }
            return thisCopy;
        }

    //Returning the Echelon Form of a matrix
        matrix rowEchelonForm() {
            /*
            MUST meet two conditions:   https://www.youtube.com/watch?app=desktop&v=ShonVncOAB4
            1. Any row of zeros MUST be at the bottom
            2. The leading entry in a row MUST be to the right of the leading in the row above it
            */
            return (this->upper_or_lower_triangular_matrices(true));
        }

    //Returning the Reduced Row Echelon Form
        matrix reduced_RowEchelonForm() {
            /*https://www.emathhelp.net/en/calculators/linear-algebra/reduced-row-echelon-form-rref-calculator/?i=%5B%5B-6%2C3%2C8%2C4%5D%2C%5B4%2C5%2C2%2C9%5D%2C%5B3%2C6%2C9%2C5%5D%2C%5B3%2C7%2C1%2C2%5D%5D&reduced=on
            MUST meet three conditions:
            1. The matrix MUST be in row echelon form
            2. The pivots MUST be one
            3. Every column with a pivot MUST have every other value in the column equal to 0

            So...
            1. If the pivot isn't one make it one (don't forget to divide the rest of the row)
            2. Work bottom pivot up to transform everything in a pivot's column to 0 (columns with empty pivots are skipped)
            */
            //https://www.youtube.com/watch?v=l69YjkuUym0
            std::vector<std::vector<double> > matrixCopy = (this->rowEchelonForm()).vectorCopy();
            //Acquiring a vector copy of thisMatrix to change

            std::vector<std::tuple<int,int> > pivotVector;   //Vector that holds all pivots for the second loop
            for(int i= 0; i < numRows; i++) {
                for(int j= 0; j < numCols; j++) {
                    if(i >= numRows) {
                        break;
                        //Break out of the j loop otherwise a row that doesn't exist could be referenced
                    }
                    else {
                        if(matrixCopy[j][i] != 0) {
                        //If the cell is not equal to 0 it must be a pivot
                            pivotVector.push_back(std::tuple<int,int> (i, j));  //Add the pivot to the pivot list
                            for(int k= numCols-1; k >= j; k--) {
                            //Iterate through the pivot's row (excluding the 0's to the left of the pivot)
                                matrixCopy[k][i] = matrixCopy[k][i] / matrixCopy[j][i]; //Dividing by the pivot (to turn the pivot into 1)
                            }
                            i += 1;
                            //Iterate i by one so the next row can be check for pivots (only one per row and in sequential order column-wise)
                        }
                    }
                }
            }

            for(int i= pivotVector.size()-1; i > 0; i--) {
            //Iterating through the pivots found in the loop above
            //The first pivot does not need to be included because the rest of the row is only affected by those below it. There should be no change beyond making the pivot equal to 1
                for(int j= numCols-1; j >= std::get<1>(pivotVector[i]); j--) {
                //Starting from the far right side of the matrix, on the first pivot, to the pivot column (and no further, as it's not necessary)
                    for(int k= std::get<0>(pivotVector[i])-1; k >= 0; k--) {
                    //Starting on the row just above the pivot row to the top of the matrix
                        matrixCopy[j][k] = matrixCopy[j][k] - (matrixCopy[j][std::get<0>(pivotVector[i])] * matrixCopy[std::get<1>(pivotVector[i])][k]);
                        //whateverAbovePivotRow - (PivotRow * whateverAbovePivot)
                    }
                }
            }
            return matrix(matrixCopy);  //Returning the modified matrix
        }

    //Upper Triangular Matrix
        matrix upper_triangular() {
            return (this->upper_or_lower_triangular_matrices(true));
        }

    //Lower Triangular Matrix
        matrix lower_triangular() {
            return(this->upper_or_lower_triangular_matrices(false));
        }

    //Returning the determinant
        double findDeterminant() {
            //Returns the determinant of a matrix
        //https://www.mathsisfun.com/algebra/matrix-determinant.html
            if(determinantSolved) {
                return determinant;
            }
            else if((numRows <= 0) || (numCols <= 0) || (numRows != numCols)) {
            //No determinent can be found if the matrix isn't a square or if the row/ column size is less than 1
                return 0;
                //Returning 0 to make the function recursive as well as satisfying the inverse function logic statement (can't be inversed if the determinant is 0)
            }
            else if((numRows == 1) && (numCols == 1)) {
            //If a 1x1 matrix is submitted its' determinant (the only value present inside) is returned
                return matrice[0][0];
                //Returning the only value present in the matrix column 1 (column 0 is the index) row 0
            }
            else if((numCols == 2) && (numRows == 2)) {
            //If the matrix is 2x2
                determinant = (matrice[0][0] * matrice[1][1]) - (matrice[1][0] * matrice[0][1]);
                //The determinant of a 2x2 matrix is |A| = ad - bc
            }
            else {
            //3x3 or greater
                determinant = 1;
                matrix thisREF = this->rowEchelonForm();
                for(int i= 0; i < numCols; i++) {
                //Since only square can have determinants no double loop is necessary
                    determinant = determinant * thisREF.locateValue(i, i);
                }
                determinant = determinant * std::pow(-1, rows_swapped); //Multiplying the determinant by the number of rows swapped to ensure an accurate measure determinant * -1^rows_swapped
            }

            determinantSolved = true;   //Marking as true so computational power isn't wasted finding it again
            return determinant;
            //Returns the determinant total of the matrix
        }

    //Returning the eigenvalue(s)
        std::vector<double> eigenvalues() {
        //R goes unused in the else branch, find a way to delete it
            std::vector<double> b_k;
            if((numCols != numRows) || ((numCols == 1) && (numRows == 1))) {
            //Non-square matrice, so no eigenvalue/vector
                std::cout << "Only square matrices have eigenvalues. Please ensure the matrix entered is square (MxN: M=N) and not a number in a matrix class (MxN = 1x1)."; throw std::exception();
            }
            else if(this->findDeterminant() == 0) {
            //Special case to find eigenvalues

            }
            else {
            //All other matrices

            //for(int i= 0; i < numRows; i++) {do the below} keep Q and R stored in a vector, then multiply all of them together to get the eigenvalues

                //Using Von Mises iteration to find eigenvectors
                double tolerance = 0.0001;  //Used to measure the relative distance of two rows
                bool AX_eq_UpTrX = false;    //Used to determine the the relative distance between two rows is within tolerance
                std::tuple<matrix, matrix> X_QR = this->qr_decomposition();
                int looper_watcher = 0;
                while(AX_eq_UpTrX == false && looper_watcher <= 22) {
                    matrix X = (std::get<1>(X_QR).multiply(std::get<0>(X_QR)));     // X = R * Q
                    X_QR = X.qr_decomposition();
                    matrix X_UpperTri = X.upper_triangular();
                    for(int i= 0; i < X.rowCount(); i++) {
                        //Compare AQ and previous within a given tolerance (create a function so copies don't need to be made)
                        if(toleranceMeasure(X.getRow(i), X_UpperTri.getRow(i), tolerance) == false) {
                            AX_eq_UpTrX = false;
                            break;
                        }
                        else {AX_eq_UpTrX = true;}
                        //If one of the rows don't meet tolerance the loop is broken and convergence and computed again with adjustments made again
                    }
                    looper_watcher++;
                }
                matrix eigenValues_in_diagonal = (std::get<1>(X_QR).multiply(std::get<0>(X_QR)));
                for(int i= 0; i < numCols; i++) {
                    eigenValue_Vector.push_back(eigenValues_in_diagonal.locateValue(i,i));
                }
            }
            return eigenValue_Vector;
        }

    //Using VonMises' Iteration to find eigenvectors (which can then be used to find eigen values)
    //You'll need this later: https://math.stackexchange.com/questions/662036/is-there-a-general-way-to-solve-a-multivariate-polynomial
    //https://en.wikipedia.org/wiki/Power_iteration
        std::tuple<matrix,matrix> qr_decomposition() {

            //R is not simply the upper triangular, see... https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf

            //https://en.wikipedia.org/wiki/Matrix_decomposition | Might add the rest of these at some point too
            //http://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
            
            matrix Q = this->GramSchmidt();
            matrix R = (Q.transpose()).multiply(*this);

            return std::tuple<matrix,matrix> (Q, R);
        }
    
    //Returning an identity matrix
        matrix identityMatrix() {
            int largestVector = (numCols >= numRows) ? numCols : numRows;
            std::vector<std::vector<double> > I;
            for(int k= 0; k < largestVector; k++) {
                I.push_back(std::vector<double> (largestVector));
                I[k][k] = 1;
            }
            return matrix(I);
        }
    
    //Returning the eigenvector(s)

    //Returning a vector of double values representing eigenvalues
    //To Add:
        //Multiplication, addition, and subtraction with just a double vector (one column) WITHOUT making the "one column" (vector) a matrix first
        //Delete rows and delete columns (will need to check for the largest value again and set determinant to false)
        //Change the value of a cell (basically done with locateValue)
        //Read CSV (makes your life easier when testing larger matrices)
            //Add an option for including indeces and column headers or not (in the csv) when importing into the script and exporting out

    //Changes:
        //Might need to re-write triangular or row echelon to meet the following requirements: https://math.stackexchange.com/questions/1720647/is-there-no-difference-between-upper-triangular-matrix-and-echelon-matrixrow-ec
        //Determine when a matrix can have an inverse
        //Change determinantSolved to calculatedPertubations as well as adding a private value of the identity matrix, so it doesn't need to be recalculated
        //For print(): want left alignned and might include pipes (|) on the edges to better distinguish columns in place on an extra space between each
        //Maybe add something so 0's are filled in so all columns are of equal length if they aren't inserted as so
};