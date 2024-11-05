import cv2
import numpy as np
import matplotlib.pyplot as plt

#Função Ruído Gaussiano
def ruido_Gaussiano(imagem):
    #Copia da imagem, para nao alterar a original
    imagem_ruido = np.copy(imagem)

    #media e desvio do ruido
    media = 0
    desvio = 25

    #gerando o ruido
    ruido = np.random.normal(media, desvio, imagem_ruido.shape).astype(np.uint8)
    #adicionando o ruido a imagem
    imagem_ruidosa = cv2.add(imagem, ruido)

    return imagem_ruidosa

#Função Ruido Sal e Pimenta
def sal_Pimenta(imagem, prob):
    #Copia da imagem, para nao alterar a original
    imagem_ruido = np.copy(imagem)

    #Define numero de pixels que receberao o ruido
    num_sal = np.ceil (prob * imagem.size * 0.5).astype(int)
    num_pimenta = np.ceil (prob * imagem.size * 0.5).astype(int)

    #Coordenadas para o ruido "Sal" (Branco)
    coords = [np.random.randint(0, i-1, num_sal) for i in imagem.shape]
    imagem_ruido[tuple(coords)] = 255

    #Coordenadas para o ruido "Pimenta" (Preto)
    coords = [np.random.randint(0, i-1, num_pimenta) for i in imagem.shape]
    imagem_ruido[tuple(coords)] = 0

    return imagem_ruido

def filtro_Gaussiano(imagem, escala):
    #Copia da imagem, para nao alterar a original
    imagem_filtro_gauss = np.copy(imagem)

    #Adicionando o Filtro na imagem
    filtro_gauss = cv2.blur(imagem_filtro_gauss, (escala, escala))

    return filtro_gauss

def filtro_Mediana(imagem, escala):
    #Copia da imagem, para nao alterar a original
    imagem_filtro_mediana = np.copy(imagem)

    #Adicionando o Filtro na imagem
    filtro_mediana = cv2.GaussianBlur(imagem, (escala, escala), 0)

    return filtro_mediana

def filtro_Bilateral(imagem):
    #Copia da imagem, para nao alterar a original
    imagem_filtro_bilateral = np.copy(imagem)

    #Aplicando o filtro
    filtro_bilateral = cv2.bilateralFilter(imagem_filtro_bilateral, d=15, sigmaColor=75, sigmaSpace=75)

    return filtro_bilateral

#Carregando a imagem
imagem = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

#Adicionando o Ruído Gaussiano
imagem_ruidosa_gauss = ruido_Gaussiano(imagem)

#Adicionando o Ruido Sal e Pimenta
imagem_ruidosa_sal_pim = sal_Pimenta(imagem, prob=0.02)

#Adicionando Filtro Gaussiano nas Imagens Ruidosas
imagem_filtro_gauss1 = filtro_Gaussiano(imagem_ruidosa_gauss, 5)
imagem_filtro_gauss2 = filtro_Gaussiano(imagem_ruidosa_sal_pim, 5)

#Adicionando o Filtro de Mediana nas Imagens Ruidosas
imagem_filtro_mediana1 = filtro_Mediana(imagem_ruidosa_gauss, 5)
imagem_filtro_mediana2 = filtro_Mediana(imagem_ruidosa_sal_pim, 5)

#Adicionando o Filtro Bilateral
imagem_filtro_bilat1 = filtro_Bilateral(imagem_ruidosa_gauss)
imagem_filtro_bilat2 = filtro_Bilateral(imagem_ruidosa_sal_pim)

#Exibindo as imagens / Imagem Original
plt.figure(figsize=(10, 5))
plt.subplot(3, 3, 1)
plt.title("Imagem Original")
plt.imshow(imagem, cmap='gray')
plt.axis('off')

#Imagem com Ruido Gaussiano
plt.subplot(3, 3, 2)
plt.title("Ruído Gaussiano")
plt.imshow(imagem_ruidosa_gauss, cmap='gray')
plt.axis('off')

#Imagem com Ruido Sal e Pimenta
plt.subplot(3, 3, 3)
plt.title("Ruído Sal e Pimenta")
plt.imshow(imagem_ruidosa_sal_pim, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Gaussiana / Filtro Gaussiano
plt.subplot(3, 3, 4)
plt.title("Filtro Gaussiano / Ruido Gaussiano")
plt.imshow(imagem_filtro_gauss1, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Sal e Pimenta / Filtro Gaussiano
plt.subplot(3, 3, 7)
plt.title("Filtro Gaussiano / Ruido Sal Pimenta")
plt.imshow(imagem_filtro_gauss2, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Gaussiana / Filtro Mediana
plt.subplot(3, 3, 5)
plt.title("Filtro Mediana / Ruido Gaussiano")
plt.imshow(imagem_filtro_mediana1, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Sal e Pimenta / Filtro Mediana
plt.subplot(3, 3, 8)
plt.title("Filtro Mediana / Ruido Sal Pimenta")
plt.imshow(imagem_filtro_mediana2, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Gaussiana / Filtro Bilateral
plt.subplot(3, 3, 6)
plt.title("Filtro Bilateral / Ruido Gaussiano")
plt.imshow(imagem_filtro_bilat1, cmap='gray')
plt.axis('off')

#Imagem Ruidosa Sal e Pimenta / Filtro Bilateral
plt.subplot(3, 3, 9)
plt.title("Filtro Bilateral / Ruido Sal Pimenta")
plt.imshow(imagem_filtro_bilat2, cmap='gray')
plt.axis('off')

plt.show()