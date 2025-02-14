#include <cuda_runtime.h>
#include <stdio.h>

__global__
void kernel_difuminar(const unsigned char* d_entrada, unsigned char* d_salida, int ancho, int alto, int tamanioFiltro) {
    int indiceX = blockIdx.x * blockDim.x + threadIdx.x;
    int indiceY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (indiceX < ancho && indiceY < alto) {
        int mitadFiltro = tamanioFiltro / 2;
        float rojo = 0, verde = 0, azul = 0, alfa = 0;
        int contador = 0;
        
        // Calcular promedio de pixeles vecinos
        for (int fy = -mitadFiltro; fy <= mitadFiltro; fy++) {
            for (int fx = -mitadFiltro; fx <= mitadFiltro; fx++) {
                int pixelX = indiceX + fx;
                int pixelY = indiceY + fy;
                
                if (pixelX >= 0 && pixelX < ancho && pixelY >= 0 && pixelY < alto) {
                    int pos = (pixelY * ancho + pixelX) * 4;
                    rojo += d_entrada[pos];
                    verde += d_entrada[pos + 1];
                    azul += d_entrada[pos + 2];
                    alfa += d_entrada[pos + 3];
                    contador++;
                }
            }
        }
        
        // Escribir valores promediados
        int pos = (indiceY * ancho + indiceX) * 4;
        d_salida[pos] = (unsigned char)(rojo / contador);
        d_salida[pos + 1] = (unsigned char)(verde / contador);
        d_salida[pos + 2] = (unsigned char)(azul / contador);
        d_salida[pos + 3] = (unsigned char)(alfa / contador);
    }
}

extern "C"
void aplicar_difuminado(const unsigned char* h_entrada, unsigned char* h_salida, int ancho, int alto, int tamanioFiltro) {
    unsigned char *d_entrada, *d_salida;
    size_t tamanioDatos = ancho * alto * 4 * sizeof(unsigned char);
    
    // Reservar memoria en dispositivo
    cudaMalloc(&d_entrada, tamanioDatos);
    cudaMalloc(&d_salida, tamanioDatos);
    
    // Copiar entrada al dispositivo
    cudaMemcpy(d_entrada, h_entrada, tamanioDatos, cudaMemcpyHostToDevice);
    
    // Configurar grid y bloques
    dim3 tamanioBloque(16, 16);
    dim3 tamanioGrid((ancho + tamanioBloque.x - 1) / tamanioBloque.x,
                     (alto + tamanioBloque.y - 1) / tamanioBloque.y);
    
    // Ejecutar kernel
    kernel_difuminar<<<tamanioGrid, tamanioBloque>>>(d_entrada, d_salida, ancho, alto, tamanioFiltro);
    
    // Copiar resultado de vuelta al host
    cudaMemcpy(h_salida, d_salida, tamanioDatos, cudaMemcpyDeviceToHost);
    
    // Liberar memoria
    cudaFree(d_entrada);
    cudaFree(d_salida);
}