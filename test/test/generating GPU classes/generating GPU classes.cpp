/**
* Create the major classes in DECX, assign values and print them
*/

#define TYPE float

#define GPU_MATRIX 1
#define GPU_VECTOR 0
#define GPU_TENSOR 0
#define GPU_MATRIXARRAY 0
#define GPU_TENSORARRAY 0

#if GPU_MATRIX
#include "creating_GPU_Matrix.h"

int main()
{
    generate_GPU_matrix<TYPE>(10, 10, de::Page_Locked);

    return 0;
}

#endif


#if GPU_VECTOR
#include "creating_GPU_Vector.h"

int main()
{
    generate_GPU_Vector<TYPE>(1920, de::Page_Locked);

    return 0;
}

#endif


#if GPU_TENSOR
#include "creating_GPU_Tensor.h"

int main()
{
    generate_GPU_Tensor<TYPE>(10, 10, 10, de::Page_Locked);

    return 0;
}

#endif


#if GPU_MATRIXARRAY
#include "creating_GPU_MatrixArray.h"

int main()
{
    generate_GPU_MatrixArray<TYPE>(10, 10, 10, de::Page_Locked);

    return 0;
}
#endif



#if GPU_TENSORARRAY
#include "creating_GPU_TensorArray.h"

int main()
{
    generate_TensorArray<TYPE>(10, 10, 10, 2, de::Page_Locked);

    return 0;
}

#endif