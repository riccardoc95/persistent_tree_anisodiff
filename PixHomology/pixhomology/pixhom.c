#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define MODULE_API_EXPORTS
#include "pixhom.h"


// Function to find Argmin and Argmax of array
MinMaxIndices findArgminArgmax(const double *arr, int size) {
    MinMaxIndices result;

    if (size == 0) {
        // Handle the case when the array is empty
        result.argmin = -1;  // Convention for indicating no minimum
        result.argmax = -1;  // Convention for indicating no maximum
        return result;
    }

    double min_val = arr[0];
    double max_val = arr[0];
    int min_index = 0;
    int max_index = 0;

    for (int i = 1; i < size; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
            min_index = i;
        }else if (arr[i] == min_val){
            if (i < min_index){
                min_index = i;
            }
        }

        if (arr[i] > max_val) {
            max_val = arr[i];
            max_index = i;
        }else if (arr[i] == max_val){
            if (i > max_index){
                max_index = i;
            }
        }

    }

    result.argmin = min_index;
    result.argmax = max_index;

    return result;
}

// Function to compare UPoints for qsort
int compareUPoints(const void *a, const void *b) { 
    UPoint *p1 = (UPoint *) a;
    UPoint *p2 = (UPoint *) b;
    if (p2->u_val > p1->u_val){
        return 1;
    }else if (p2->u_val < p1->u_val){
        return -1;
    }else{
        if (p2->c_val > p1->c_val){
            return 1;
        }else if (p2->c_val < p1->c_val){
            return -1;
        }else{
            return 0;    
        }
    }
}


// Persistent Homology dimension 0 function
MODULE_API Result computePH(double *inputArray, int numRows, int numCols) {
    // Calculate Argmin and Argmax
    MinMaxIndices argMinMax = findArgminArgmax(inputArray, numRows * numCols);
    
    // Allocate memory for the mpatch array
    int *mpatch = malloc(numRows * numCols * sizeof(int));

    // Allocate memory for the array of UPoints
    UPoint *u_points = malloc(1 * sizeof(UPoint));
    int num_u_points = 0;

    // Set up mpatch array
    for (int i = 0; i < numRows * numCols; i++) {
        mpatch[i] = i;
    }

    // Create padded array to handle boundaries with -inf
    double* paddedArray = (double*)malloc((numRows + 2) * (numCols + 2) * sizeof(double));

    // Initialize padded array to -inf
    for (int i = 0; i < (numRows + 2) * (numCols + 2); i++) {
        paddedArray[i] = -DBL_MAX;
    }

    // Copy inputArray into the paddedArray (shifted by 1)
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            paddedArray[(i + 1) * (numCols + 2) + (j + 1)] = inputArray[i * numCols + j];
        }
    }

    // Iterate over the original image dimensions
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int c_point = i * numCols + j;

            // Find the maximum value in the 3x3 neighborhood
            int maxIdx = -1;
            double maxValue = -DBL_MAX;

            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int ni = i + di; // padded index
                    int nj = j + dj;
                    double val = paddedArray[ni * (numCols + 2) + nj];

                    if (val > maxValue) {
                        maxValue = val;
                        maxIdx = di * 3 + dj; // Store index relative to 3x3 neighborhood
                    }
                }
            }

            // Convert local maxIdx to the original (flattened) index
            int ni = i + (maxIdx / 3) - 1; // Adjust back to non-padded index
            int nj = j + (maxIdx % 3) - 1;
            mpatch[c_point] = ni * numCols + nj;
        }
    }

    // Free allocated memory for the padded array
    free(paddedArray);

    // Second pass to update the mpatch array
    while (1) {
        int changed = 0;
        for (int i = 0; i < numRows * numCols; i++) {
            if (mpatch[i] != mpatch[mpatch[i]]) {
                mpatch[i] = mpatch[mpatch[i]];
                changed = 1;
            }
        }
        if (!changed) {
            break;
        }
    }

    // Find u_points
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int x_start = (i > 0) ? (i - 1) : 0;
            int y_start = (j > 0) ? (j - 1) : 0;
            int x_end = (i < (numRows - 1)) ? (i + 1) : (numRows - 1);
            int y_end = (j < (numCols - 1)) ? (j + 1) : (numCols - 1);

            int c_point = i * numCols + j;

            int u_point = c_point;
            double u_val = inputArray[u_point];

            for (int h = x_start; h <= x_end; h++) {
                for (int k = y_start; k <= y_end; k++) {
                    int t_point = h * numCols + k;

                    int c_obj = mpatch[c_point];
                    int t_obj = mpatch[t_point];

                    double c_val = inputArray[c_point];
                    double t_val = inputArray[t_point];

                    if (c_point != t_point && c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                        u_points = realloc(u_points, (1 + num_u_points) * sizeof(UPoint));
                        
                        u_point = t_point;
                        u_val = inputArray[u_point];

                        // Store information about u_point in the u_points array
                        u_points[num_u_points].u_val = u_val;
                        u_points[num_u_points].c_val = c_val;
                        u_points[num_u_points].c_point = c_point;
                        u_points[num_u_points].u_point = u_point;

                        num_u_points++;           
                        
                    }
                }
            }
        }
    }

    // Sort u_points in descending order
    qsort(u_points, num_u_points, sizeof(*u_points), compareUPoints);

    // Create an array to store information about dgm
    double *dgm = (double *)malloc(2 * sizeof(double));
    int num_dgm = 0;
   
    // Find dgm
    for (int i = 0; i < num_u_points; i++) {
        int c_point = u_points[i].c_point;
        int u_point = u_points[i].u_point;

        int c_obj = c_point;
        int u_obj = u_point;

        while (c_obj != mpatch[c_obj]) {
            c_obj = mpatch[c_obj];
        }
        while (u_obj != mpatch[u_obj]) {
            u_obj = mpatch[u_obj];
        }
    
        if (c_obj != u_obj) {
            if (inputArray[c_obj] > inputArray[u_obj]) {
                mpatch[u_obj] = c_obj;
                if (fabs(inputArray[u_obj] - inputArray[u_point]) > 0) {
                    dgm[num_dgm] = inputArray[u_obj];
                    dgm[(num_dgm + 1)] = inputArray[u_point];
                    num_dgm = num_dgm + 2;
                    dgm = realloc(dgm, (num_dgm + 2) * sizeof(double));
                }
            } else if (inputArray[c_obj] < inputArray[u_obj]) {
                mpatch[c_obj] = u_obj;
                if (fabs(inputArray[c_obj] - inputArray[u_point]) > 0) {
                    dgm[num_dgm]  = inputArray[c_obj];
                    dgm[(num_dgm + 1)] = inputArray[u_point];
                    num_dgm = num_dgm + 2;
                    dgm = realloc(dgm, (num_dgm + 2) * sizeof(double));
                }
            } else{
                if (c_obj > u_obj){
                    mpatch[u_obj] = c_obj;
                    if (fabs(inputArray[u_obj] - inputArray[u_point]) > 0) {
                        dgm[num_dgm] = inputArray[u_obj];
                        dgm[(num_dgm + 1)] = inputArray[u_point];
                        num_dgm = num_dgm + 2;
                        dgm = realloc(dgm, (num_dgm + 2) * sizeof(double));
                    }
                }else{
                    mpatch[c_obj] = u_obj;
                    if (fabs(inputArray[c_obj] - inputArray[u_point]) > 0) {
                        dgm[num_dgm]  = inputArray[c_obj];
                        dgm[(num_dgm + 1)] = inputArray[u_point];
                        num_dgm = num_dgm + 2;
                        dgm = realloc(dgm, (num_dgm + 2) * sizeof(double));
                    }
                }
                
            }
        }
    }

    // Append the maximum and minimum values to dgm
    dgm[num_dgm] = inputArray[argMinMax.argmax];  // Maximum value
    dgm[(num_dgm + 1)] = inputArray[argMinMax.argmin];  // Minimum value
    num_dgm = num_dgm + 2;

    //Clean
    free(mpatch);
    free(u_points);

    Result res = { dgm, num_dgm };
    return res; 
    
}
