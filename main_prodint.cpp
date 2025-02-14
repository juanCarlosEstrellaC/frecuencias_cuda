#include <iostream>
#include <cuda_runtime.h>

extern "C"
void productoExterno(float* d_U, float* d_V, float* d_M, int n);

void imprimirMatriz(float* M, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << M[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  int n = 4; // Puedes cambiar el tamaño de la matriz
  float U[] = {1.0, 2.0, 3.0, 4.0}; // Vector U
  float V[] = {5.0, 6.0, 7.0, 8.0}; // Vector V

  float* d_U, * d_V, * d_M;
  float M[n * n]; // Matriz resultado

  // Reservamos memoria en el dispositivo
  cudaMalloc(&d_U, n * sizeof(float));
  cudaMalloc(&d_V, n * sizeof(float));
  cudaMalloc(&d_M, n * n * sizeof(float));

  // Copiamos los datos de los vectores U y V a la memoria del dispositivo
  cudaMemcpy(d_U, U, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice);

  // Invocamos el kernel
  productoExterno(d_U, d_V, d_M, n);

  // Copiamos el resultado de vuelta a la memoria del host
  cudaMemcpy(M, d_M, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Imprimimos el resultado
  imprimirMatriz(M, n);

  // Liberamos la memoria del dispositivo
  cudaFree(d_U);
  cudaFree(d_V);
  cudaFree(d_M);

  return 0;
}
