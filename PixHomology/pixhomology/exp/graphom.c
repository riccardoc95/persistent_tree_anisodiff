#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define MODULE_API_EXPORTS
#include "graphom.h"

// Function to generate random normal dist from uniform with Marsaglia and Bray method
double randn(double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }
    
    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (double) X1);
}


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
MODULE_API Graph computeGraph(double *inputArray, int numRows, int numCols) {

    // Calculate Argmin and Argmax
    MinMaxIndices argMinMax = findArgminArgmax(inputArray, numRows * numCols);

    
    // Allocate memory for the edges and weights arrays
    int *edges = malloc(numRows * numCols * sizeof(int));
    double *weights = malloc(numRows * numCols * sizeof(double));

    // Allocate memory for the array of UPoints
    UPoint *u_points = malloc(1 * sizeof(UPoint));
    int num_u_points = 0;

    // Set up edges array
    for (int i = 0; i < numRows * numCols; i++) {
        edges[i] = i;
        weights[i] = 0;
    }

    // First pass to find local maxima
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
            edges[c_point] = ni * numCols + nj;
        }
    }

    // Free allocated memory for the padded array
    free(paddedArray);

    // Second pass to update the edges array
    while (1) {
        int changed = 0;
        for (int i = 0; i < numRows * numCols; i++) {
            if (edges[i] != edges[edges[i]]) {
                edges[i] = edges[edges[i]];
                weights[i] = (inputArray[edges[edges[i]]] - inputArray[i]);
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

                    int c_obj = edges[c_point];
                    int t_obj = edges[t_point];

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

    // Find dgm
    for (int i = 0; i < num_u_points; i++) {
        int c_point = u_points[i].c_point;
        int u_point = u_points[i].u_point;

        int c_obj = c_point;
        int u_obj = u_point;

        while (c_obj != edges[c_obj]) {
            c_obj = edges[c_obj];
        }
        while (u_obj != edges[u_obj]) {
            u_obj = edges[u_obj];
        }

        if (c_obj != u_obj) {
            if (inputArray[c_obj] > inputArray[u_obj]) {
                edges[u_obj] = c_point;
                weights[u_obj] = (inputArray[c_obj] - inputArray[u_obj]);
            } else if (inputArray[c_obj] < inputArray[u_obj]) {
                edges[c_obj] = u_point;
                weights[c_obj] = (inputArray[u_obj] - inputArray[c_obj]);
            } else{
                if (c_obj > u_obj){
                    edges[u_obj] = c_point;
                    weights[u_obj] = (inputArray[c_obj] - inputArray[u_obj]);
                }else{
                    edges[c_obj] = u_point;
                    weights[c_obj] = (inputArray[u_obj] - inputArray[c_obj]);
                }

            }
        }
    }

    //Clean
    free(u_points);

    Graph res = { edges, weights };
    return res; 
    
}
