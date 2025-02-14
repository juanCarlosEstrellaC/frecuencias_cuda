#include <SFML/Graphics.hpp>
#include <omp.h>
#include <iostream>
#include <vector>

// Declaración de función CUDA
extern "C" void aplicar_difuminado(const unsigned char* entrada, unsigned char* salida, int ancho, int alto, int tamanioFiltro);

// Implementación OpenMP del difuminado
void aplicar_difuminado_omp(const unsigned char* entrada, unsigned char* salida, int ancho, int alto, int tamanioFiltro) {
    int mitadFiltro = tamanioFiltro / 2;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float rojo = 0, verde = 0, azul = 0, alfa = 0;
            int contador = 0;

            // Calcular promedio de pixeles vecinos
            for (int fy = -mitadFiltro; fy <= mitadFiltro; fy++) {
                for (int fx = -mitadFiltro; fx <= mitadFiltro; fx++) {
                    int pixelX = x + fx;
                    int pixelY = y + fy;

                    if (pixelX >= 0 && pixelX < ancho && pixelY >= 0 && pixelY < alto) {
                        int pos = (pixelY * ancho + pixelX) * 4;
                        rojo += entrada[pos];
                        verde += entrada[pos + 1];
                        azul += entrada[pos + 2];
                        alfa += entrada[pos + 3];
                        contador++;
                    }
                }
            }

            // Escribir valores promediados
            int pos = (y * ancho + x) * 4;
            salida[pos] = (unsigned char)(rojo / contador);
            salida[pos + 1] = (unsigned char)(verde / contador);
            salida[pos + 2] = (unsigned char)(azul / contador);
            salida[pos + 3] = (unsigned char)(alfa / contador);
        }
    }
}

int main() {
    // Cargar imagen
    sf::Image imagen;
    if (!imagen.loadFromFile("entrada.jpg")) {
        std::cout << "Error al cargar la imagen" << std::endl;
        return -1;
    }

    // Crear ventana
    sf::Vector2u tamanio = imagen.getSize();
    sf::RenderWindow ventana(sf::VideoMode(tamanio.x, tamanio.y), "Difuminado de Imagen");

    // Crear texturas y sprites para imágenes original y difuminada
    sf::Texture texturaOriginal, texturaDifuminada;
    texturaOriginal.loadFromImage(imagen);
    texturaDifuminada.create(tamanio.x, tamanio.y);

    sf::Sprite spriteOriginal(texturaOriginal);
    sf::Sprite spriteDifuminado(texturaDifuminada);

    // Obtener datos de la imagen
    const sf::Uint8* pixelesEntrada = imagen.getPixelsPtr();
    std::vector<sf::Uint8> pixelesSalida(tamanio.x * tamanio.y * 4);

    // Parámetros
    const int tamanioFiltro = 3; // grid 3x3
    bool mostrarDifuminado = false;
    bool usarGPU = false;

    // Crear versión difuminada
    if (usarGPU) {
        aplicar_difuminado(pixelesEntrada, pixelesSalida.data(), tamanio.x, tamanio.y, tamanioFiltro);
    } else {
        aplicar_difuminado_omp(pixelesEntrada, pixelesSalida.data(), tamanio.x, tamanio.y, tamanioFiltro);
    }

    texturaDifuminada.update(pixelesSalida.data());

    // Bucle principal
    while (ventana.isOpen()) {
        sf::Event evento;
        while (ventana.pollEvent(evento)) {
            if (evento.type == sf::Event::Closed)
                ventana.close();

            if (evento.type == sf::Event::KeyPressed) {
                if (evento.key.code == sf::Keyboard::B)
                    mostrarDifuminado = true;
                else if (evento.key.code == sf::Keyboard::R)
                    mostrarDifuminado = false;
                else if (evento.key.code == sf::Keyboard::G)
                    usarGPU = !usarGPU;
            }
        }

        ventana.clear();
        ventana.draw(mostrarDifuminado ? spriteDifuminado : spriteOriginal);
        ventana.display();
    }

    return 0;
}