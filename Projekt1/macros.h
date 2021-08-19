#pragma once
//Macro obliczajace id watku w calym gridzie
#define GetID() (blockIdx.x * blockDim.x + threadIdx.x)