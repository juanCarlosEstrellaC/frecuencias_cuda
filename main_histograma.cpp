#include <iostream>
#include <vector>
#include <cuda_runtime.h>

extern "C"
void generar_histograma(int* d_vector, int* d_histograma, int tamanio_vector, int num_bins);

int main() {
    std::vector<int> v_vector = {10, 25, 35, 40, 50, 60, 15, 70, 90, 100};
    int tamanio_vector = v_vector.size();
    int num_bins = 10;  // Número de bins para el histograma

    // Crear vector para el histograma
    std::vector<int> h_histograma(num_bins, 0);

    int *d_vector, *d_histograma;

    // 1. Reservar memoria en el device
    cudaMalloc(&d_vector, tamanio_vector * sizeof(int));
    cudaMalloc(&d_histograma, num_bins * sizeof(int));

    // 2. Copiar de host a device
    cudaMemcpy(d_vector, v_vector.data(), tamanio_vector * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histograma, h_histograma.data(), num_bins * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Invocar el kernel
    generar_histograma(d_vector, d_histograma, tamanio_vector, num_bins);

    // 4. Copiar de device a host
    cudaMemcpy(h_histograma.data(), d_histograma, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Imprimir resultados
    std::cout << "Bin | Frecuencia\n";
    std::cout << "----------------\n";
    for (int i = 0; i < num_bins; i++) {
        std::cout << i << "   | " << h_histograma[i] << "\n";
    }

    // 6. Liberar memoria en device
    cudaFree(d_vector);
    cudaFree(d_histograma);

    return 0;
}
