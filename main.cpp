#include <iostream>
#include <cuda_runtime.h>

extern "C"
void tabla_frecuencias(int* d_vector, int* d_frecuencia, int tamanio_vector);

int main()
{
  int num_mayor = 44 + 1;  // Incluye el 44
  int h_vector[] = {1, 21, 35, 44, 5, 6, 7, 8, 8, 21};
  int tamanio_vector = sizeof(h_vector) / sizeof(h_vector[0]);

  int h_frecuencia[num_mayor] = {0};

  int *d_vector, *d_frecuencia;

  // 1. Reservar memoria en el device
  cudaMalloc(&d_vector, tamanio_vector * sizeof(int));
  cudaMalloc(&d_frecuencia, num_mayor * sizeof(int));

  // 2. Copiar de host a device
  cudaMemcpy(d_vector, h_vector, tamanio_vector * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_frecuencia, h_frecuencia, num_mayor * sizeof(int), cudaMemcpyHostToDevice);

  // 3. Invocar el kernel
  tabla_frecuencias(d_vector, d_frecuencia, tamanio_vector);

  // 4. Copiar de device a host
  cudaMemcpy(h_frecuencia, d_frecuencia, num_mayor * sizeof(int), cudaMemcpyDeviceToHost);

  // 5. Imprimir resultados
  printf("Valor | Frecuencia\n");
  printf("------------------\n");
  for (int i = 0; i < num_mayor; i++) {
    if (h_frecuencia[i] != 0) {
      printf("%4d  | %d\n", i, h_frecuencia[i]);
    }
  }

  // 6. Liberar memoria en device
  cudaFree(d_vector);
  cudaFree(d_frecuencia);

  return 0;
}
