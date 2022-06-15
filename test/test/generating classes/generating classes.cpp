/**
* Create the major classes in DECX, assign values and print them
*/

#define TYPE float

#define MATRIX 0
#define VECTOR 0
#define TENSOR 0
#define MATRIXARRAY 0
#define TENSORARRAY 1

#if MATRIX
#include "creating_Matrix.h"

int main()
{
    generate_matrix<TYPE>(10, 10, de::Page_Locked);

    return 0;
}

#endif


#if VECTOR
#include "creating_Vector.h"

int main()
{
    generate_Vector<TYPE>(1920, de::Page_Locked);

    return 0;
}

#endif


#if TENSOR
#include "creating_Tensor.h"

int main()
{
    generate_Tensor<TYPE>(10, 10, 10, de::Page_Locked);

    return 0;
}

#endif


#if MATRIXARRAY
#include "creating_MatrixArray.h"

int main()
{
    generate_MatrixArray<TYPE>(10, 10, 10, de::Page_Locked);

    return 0;
}
#endif



#if TENSORARRAY
#include "creating_TensorArray.h"

int main()
{
    generate_TensorArray<TYPE>(10, 10, 10, 2, de::Page_Locked);

    return 0;
}

#endif