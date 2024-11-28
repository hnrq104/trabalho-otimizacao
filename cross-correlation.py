import cv2
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter

def criar_piramide_gaussiana(imagem, niveis=3):
    """Cria uma pirâmide gaussiana da imagem"""
    piramide = [imagem]
    for i in range(niveis-1):
        imagem = cv2.pyrDown(imagem)
        piramide.append(imagem)
    return piramide

def correlacao_cruzada_normalizada(template, imagem):
    """Calcula a correlação cruzada normalizada"""
    # Remove a média
    template = template - np.mean(template)
    imagem = imagem - np.mean(imagem)
    
    # Calcula a correlação normalizada
    numerador = correlate2d(imagem, template, mode='same')
    denominador = np.sqrt(
        correlate2d(imagem**2, np.ones_like(template), mode='same') *
        np.sum(template**2)
    )
    
    # Evita divisão por zero
    denominador = np.maximum(denominador, np.finfo(float).eps)
    
    return numerador / denominador

def deslocar_imagem(imagem, dx, dy):
    """Desloca a imagem usando interpolação bicúbica"""
    matriz = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(imagem, matriz, (imagem.shape[1], imagem.shape[0]), 
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def encontrar_pico_subpixel(correlacao):
    """Encontra o deslocamento com precisão subpixel usando interpolação parabólica"""
    y_max, x_max = np.unravel_index(np.argmax(correlacao), correlacao.shape)
    
    if 0 < y_max < correlacao.shape[0]-1 and 0 < x_max < correlacao.shape[1]-1:
        # Ajuste parabólico para y
        y_prev = correlacao[y_max-1, x_max]
        y_curr = correlacao[y_max, x_max]
        y_next = correlacao[y_max+1, x_max]
        dy = 0.5 * (y_next - y_prev) / (2*y_curr - y_prev - y_next)
        
        # Ajuste parabólico para x
        x_prev = correlacao[y_max, x_max-1]
        x_curr = correlacao[y_max, x_max]
        x_next = correlacao[y_max, x_max+1]
        dx = 0.5 * (x_next - x_prev) / (2*x_curr - x_prev - x_next)
        
        return y_max + dy - correlacao.shape[0]//2, x_max + dx - correlacao.shape[1]//2
    
    return y_max - correlacao.shape[0]//2, x_max - correlacao.shape[1]//2

def refinar_estimativa(imagem_original, imagem_modificada, dx_inicial, dy_inicial, 
                      tamanho_janela=21, max_iteracoes=10):
    """Refina a estimativa do deslocamento usando uma janela local"""
    dx, dy = dx_inicial, dy_inicial
    altura, largura = imagem_original.shape
    
    for _ in range(max_iteracoes):
        # Aplica o deslocamento atual
        imagem_deslocada = deslocar_imagem(imagem_original, dx, dy)
        
        # Extrai região central para correlação local
        centro_y, centro_x = altura//2, largura//2
        janela_original = imagem_deslocada[
            centro_y-tamanho_janela//2:centro_y+tamanho_janela//2+1,
            centro_x-tamanho_janela//2:centro_x+tamanho_janela//2+1
        ]
        janela_modificada = imagem_modificada[
            centro_y-tamanho_janela//2:centro_y+tamanho_janela//2+1,
            centro_x-tamanho_janela//2:centro_x+tamanho_janela//2+1
        ]
        
        # Calcula correlação local
        correlacao = correlacao_cruzada_normalizada(janela_original, janela_modificada)
        
        # Encontra ajuste fino
        dy_ajuste, dx_ajuste = encontrar_pico_subpixel(correlacao)
        
        # Atualiza estimativas
        dx += dx_ajuste
        dy += dy_ajuste
        
        # Critério de parada
        if abs(dx_ajuste) < 0.01 and abs(dy_ajuste) < 0.01:
            break
    
    return dx, dy

def estimar_deslocamento(imagem_original, imagem_modificada, niveis_piramide=3):
    """Estima o deslocamento usando uma abordagem multiescala"""
    # Criar pirâmides gaussianas
    piramide_original = criar_piramide_gaussiana(imagem_original, niveis_piramide)
    piramide_modificada = criar_piramide_gaussiana(imagem_modificada, niveis_piramide)
    
    # Começa com deslocamento zero
    dx = dy = 0
    escala = 2**(niveis_piramide-1)
    
    # Processa cada nível da pirâmide
    for nivel in range(niveis_piramide-1, -1, -1):
        # Obtém as imagens do nível atual
        img_original = piramide_original[nivel]
        img_modificada = piramide_modificada[nivel]
        
        # Aplica suavização gaussiana
        img_original = gaussian_filter(img_original, sigma=1.0)
        img_modificada = gaussian_filter(img_modificada, sigma=1.0)
        
        # Calcula correlação cruzada normalizada
        correlacao = correlacao_cruzada_normalizada(img_original, img_modificada)
        
        # Encontra o deslocamento com precisão subpixel
        dy_nivel, dx_nivel = encontrar_pico_subpixel(correlacao)
        
        # Atualiza o deslocamento total
        dx = 2 * dx + dx_nivel
        dy = 2 * dy + dy_nivel
        
        # Refina a estimativa no nível atual
        if nivel == 0:  # Refinamento apenas no nível mais fino
            dx, dy = refinar_estimativa(img_original, img_modificada, dx, dy)
    
    return dx, dy


def create_large_image(width, height):
    np.random.seed(42)  # For reproducibility
    image = np.random.rand(height, width) * 255
    return image.astype(np.uint8)

# Shift the image by a small amount
def create_shifted_image(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])  # Affine transformation matrix
    shifted_image = cv2.warpAffine(
        image, 
        matrix, 
        (image.shape[1], image.shape[0]), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REFLECT
    )
    return shifted_image


# Exemplo de uso
if __name__ == "__main__":

    # Parameters
    width, height = 64, 64  # Size of the synthetic image
    shift_x, shift_y = 6, 7  # Known subpixel shifts


    # Create the images
    original_image = create_large_image(width, height)
    cv2.imwrite("original.png",original_image)
    shifted_image = create_shifted_image(original_image, shift_x, shift_y)
    cv2.imwrite("shifted.png",shifted_image)

    # Estima o deslocamento
    dx_estimado, dy_estimado = estimar_deslocamento(original_image, shifted_image)
    print(f"Deslocamento estimado: dx = {dx_estimado:.2f}, dy = {dy_estimado:.2f}")