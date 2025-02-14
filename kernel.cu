__global__
void kernel_frecuencias(int* d_vector, int* d_frecuencia, int tamanio_vector){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < tamanio_vector) {  // Evitar hilos fuera del rango
        int valor = d_vector[index];

        if (valor < tamanio_vector) {  // Evita accesos fuera de los límites
            atomicAdd(&d_frecuencia[valor], 1);
        }
    }
}


extern "C"
void tabla_frecuencias(int* d_vector, int* d_frecuencia, int tamanio_vector){
      int numHilos = 1024;
      int numBloques = (tamanio_vector + numHilos - 1) / numHilos;
      kernel_frecuencias<<<numBloques, numHilos>>>(d_vector, d_frecuencia, tamanio_vector);
  }
