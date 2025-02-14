__global__
void productoExternoKernel(float* U, float* V, float* M, int n) {
    // Calculamos el índice global
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Verificamos que el índice esté dentro del rango
    if (idx < n * n) {
        int i = idx / n; // Índice de la fila
        int j = idx % n; // Índice de la columna
        M[idx] = U[i] * V[j];
    }
}



extern "C"
void productoExterno(float* d_U, float* d_V, float* d_M, int n) {
    int numHilos = 1024;
    int numBloques = (n * n + numHilos - 1) / numHilos;
    productoExternoKernel<<<numBloques, numHilos>>>(d_U, d_V, d_M, n);
}
