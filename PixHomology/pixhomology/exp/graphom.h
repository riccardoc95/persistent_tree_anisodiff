#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef MODULE_API_EXPORTS
#    define MODULE_API __declspec(dllexport)
#  else
#    define MODULE_API __declspec(dllimport)
#  endif
#else
#  define MODULE_API
#endif


// Define a struct to store the result
typedef struct {
    int *edges;
    double *weights;
} Graph;

// Define a struct to store information about u_points
typedef struct {
    double u_val;
    double c_val;
    int c_point;
    int u_point;
} UPoint;

// Define a struct to store information about argmin and argmax
typedef struct {
    int argmin;
    int argmax;
} MinMaxIndices;

// Function to generate random normal dist from uniform with Marsaglia and Bray method
double randn(double mu, double sigma);

// Function to find Argmin and Argmax of array
MinMaxIndices findArgminArgmax(const double *arr, int size);

// Function to compare UPoints for qsort
int compareUPoints(const void *a, const void *b);



// Persistent Homology dimension 0 function
MODULE_API Graph computePH(double *inputArray, int numRows, int numCols);

#ifdef __cplusplus
}
#endif
