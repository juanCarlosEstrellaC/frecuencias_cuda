__global__
void kernel_histograma(int* d_vector, int* d_histograma, int tamanio_vector, int min_val, int max_val, int num_bins){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < tamanio_vector) {  // Evitar hilos fuera del rango
        int valor = d_vector[index];

        // Calcular el bin correspondiente
        int bin = (valor - min_val) * num_bins / (max_val - min_val);

        // Evitar que los valores fuera del rango caigan en bins incorrectos
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&d_histograma[bin], 1);  // Incrementar el contador del bin correspondiente
        }
    }
}

extern "C"
void tabla_histograma(int* d_vector, int* d_histograma, int tamanio_vector, int num_bins){
    int min_val = INT_MAX;
    int max_val = INT_MIN;

    // Encontrar el valor mínimo y máximo en el vector
    for (int i = 0; i < tamanio_vector; i++) {
        if (d_vector[i] < min_val) min_val = d_vector[i];
        if (d_vector[i] > max_val) max_val = d_vector[i];
    }

    int numHilos = 1024;
    int numBloques = (tamanio_vector + numHilos - 1) / numHilos;

    // Llamar al kernel para calcular el histograma
    kernel_histograma<<<numBloques, numHilos>>>(d_vector, d_histograma, tamanio_vector, min_val, max_val, num_bins);
}
