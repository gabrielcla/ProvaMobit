from ultralytics import YOLO
import cv2

# Caminho da imagem que será analisada.
IMAGE_PATH = "person.png"

# Confiança mínima exigida para considerar uma detecção válida.
# Neste caso, apenas detecções com confiança igual ou superior a 50% serão mantidas.
CONFIDENCE_THRESHOLD = 0.50


def contar_pessoas(image_path: str, confidence_threshold: float=0.50):
    """
    Detecta e conta pessoas em uma imagem utilizando o modelo YOLO.

    Parâmetros:
        image_path: caminho da imagem que será analisada.
        confidence_threshold: confiança mínima das detecções.

    Retorno:
        Quantidade de pessoas detectadas na imagem.
    """

    # Lê a imagem informada e armazena seus pixels na variável img.
    img = cv2.imread(image_path)
    # Caso o caminho esteja incorreto ou o arquivo não exista, cv2.imread retorna None.
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    # 1) Carrega o modelo YOLOv8
    # Inicialmente foi utilizado o modelo "n", contudo não apresentou bom desempenho para as pessoas que estão atrás
    # Por isso optou-se pelo modelo "s" que é um pouco mais pesado mas com melhor precisão
    model = YOLO("yolov8s.pt")

    # 2) Faça a inferência utilizando o método model.predict() - Executa a detecção de objetos na imagem.
    results = model.predict(
        source=img,                    # Imagem que será analisada
        conf=confidence_threshold,     # Confiança mínima das detecções
        verbose=False                  # Evita mensagens detalhadas no terminal
    )

    result = results[0] # Como apenas uma imagem foi processada, usamos o primeiro elemento da lista de resultados.

    # Procura qual o ID corresponde à classe "person" dentre os nomes das classes reconhecidas pelo modelo.
    person_class_id = None # Armazenará o número identificador da classe "person".
    for class_id, class_name in result.names.items():
        if class_name == "person":
            person_class_id = class_id
            break
    # Verifica se a classe "person" foi encontrada entre as classes reconhecidas pelo modelo.
    if person_class_id is None:
        raise RuntimeError("A classe 'person' não foi encontrada no modelo.")


    # 3) Conte quantas detecções possuem o rótulo “person”
    # Percorre todas as caixas delimitadoras detectadas pelo YOLO. Cada caixa representa um objeto encontrado na imagem.
    person_count = 0
    for box in result.boxes:
        # Obtém o ID da classe prevista para o objeto.
        predicted_class = int(box.cls[0].item())
        # Verifica se o objeto detectado pertence à classe "person".
        if predicted_class == person_class_id:
            person_count += 1


    # Cria uma cópia da imagem contendo as caixas delimitadoras, os nomes das classes e os valores de confiança das detecções.
    annotated_img = result.plot()

    # Adiciona à imagem um texto mostrando a quantidade total de pessoas detectadas.
    cv2.putText(
        annotated_img,                         # Imagem que receberá o texto
        f"Pessoas detectadas: {person_count}", # Texto exibido
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,    # Posição inicial do texto e fonte utilizada
        0.8, (0,255,0), 2, cv2.LINE_AA         # Tamanho da fonte, cor e espessura das letras e suavização das bordas
    )

    # Mostra a quantidade de pessoas no terminal.
    print(f"Pessoas detectadas: {person_count}")

    # Abre uma janela para exibir a imagem processada.
    cv2.imshow("Deteccao de pessoas", annotated_img)
    cv2.waitKey(0)              # Mantém a janela aberta até que alguma tecla seja pressionada.
    cv2.destroyAllWindows()     # Fecha todas as janelas abertas pelo OpenCV.

    # Retorna a quantidade de pessoas detectadas.
    return person_count

if __name__ == "__main__":
    # Chama a função utilizando o caminho da imagem e o limite de confiança definidos no início do código.
    contar_pessoas(IMAGE_PATH, CONFIDENCE_THRESHOLD)
