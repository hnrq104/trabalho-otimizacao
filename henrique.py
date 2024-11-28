import numpy as np
import cv2
from scipy.signal import correlate

def correlacao_cruzada_normalizada(template, imagem):
    """Calcula a correlação cruzada normalizada para detectar deslocamento inteiro."""
    # Remove a média
    template = template - np.mean(template)
    imagem = imagem - np.mean(imagem)
    
    # Calcula a correlação normalizada
    numerador = correlate(imagem, template, mode='same')
    denominador = np.sqrt(
        correlate(imagem**2, np.ones_like(template), mode='same') *
        np.sum(template**2)
    )
    
    # Evita divisão por zero
    denominador = np.maximum(denominador, np.finfo(float).eps)
    
    return numerador / denominador

def encontrar_deslocamento(correlacao):
    """Encontra o deslocamento inteiro a partir do pico da correlação."""
    y_max, x_max = np.unravel_index(np.argmax(correlacao), correlacao.shape)
    dy = y_max - (correlacao.shape[0]) // 2
    dx = x_max - (correlacao.shape[1]) // 2
    return dx, dy

def estimar_deslocamento(imagem_original, imagem_modificada):
    """Estima o deslocamento entre duas imagens pequenas."""
    correlacao = correlacao_cruzada_normalizada(imagem_original, imagem_modificada)
    dx, dy = encontrar_deslocamento(correlacao)
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
        borderMode=cv2.BORDER_WRAP
    )
    return shifted_image


# Exemplo de uso
if __name__ == "__main__":
    # Parameters
    width, height = 128, 128  # Size of the synthetic image
    shift_x, shift_y = 6, 7  # Known subpixel shifts

    # Create the images
    original_image = create_large_image(width, height)
    shifted_image = create_shifted_image(original_image, shift_x, shift_y)

    # Estima o deslocamento
    dx_estimado, dy_estimado = estimar_deslocamento(original_image, shifted_image)
    print(f"Deslocamento estimado: dx = {dx_estimado:.2f}, dy = {dy_estimado:.2f}")