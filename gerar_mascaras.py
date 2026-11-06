"""
Geração das máscaras

Converte as anotações dos telhados (polígonos no formato YOLO-seg, exportados do Roboflow) nas máscaras usadas para treinar a U-Net.

Para cada imagem são gerados:
    mascaras/    PNG 8 bits:  0 = fundo, 1 = interior do telhado, 2 = borda do telhado
    instancias/  PNG 16 bits: 0 = fundo, 1..N = número de cada telhado

A borda é uma faixa de 2 pixels por dentro do contorno de cada telhado. Quando dois telhados se encostam, as bordas dos dois formam a divisa entre eles. 
O telhado cortado pelo limite da imagem não recebe borda nesse lado, porque ali não é o fim do telhado.

A divisão em treino, validação e teste é a mesma da pasta dataset.

"""

import shutil
from pathlib import Path
import cv2
import numpy as np

LARGURA_BORDA = 2   # em pixels
CLASSE_FUNDO = 0
CLASSE_INTERIOR = 1
CLASSE_BORDA = 2

PASTA = Path(__file__).resolve().parent
ORIGEM = PASTA / "dataset"
DESTINO = PASTA / "dataset_unet"
DIVISOES = ["train", "valid", "test"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def ler_poligonos(arquivo_rotulo, h, w):
    # lê o .txt do YOLO-seg e converte as coordenadas (0 a 1) para pixels
    poligonos = []
    if not arquivo_rotulo.exists():
        return poligonos
    for linha in arquivo_rotulo.read_text(encoding="utf-8").splitlines():
        valores = linha.split()
        if len(valores) < 7:   # classe + no mínimo 3 pontos (x, y)
            continue
        pontos = np.array(valores[1:], dtype=np.float64).reshape(-1, 2)
        pontos[:, 0] *= w
        pontos[:, 1] *= h
        poligonos.append(np.round(pontos).astype(np.int32))
    return poligonos


def gerar_mascara(poligonos, h, w):
    # mapa de instâncias: cada telhado pintado com o seu número
    instancias = np.zeros((h, w), np.uint16)
    for i, poligono in enumerate(poligonos, 1):
        cv2.fillPoly(instancias, [poligono], color=int(i))

    # máscara de 3 classes: a erosão de cada telhado é o interior, o que sobra é a borda
    mascara = np.full((h, w), CLASSE_FUNDO, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(1, len(poligonos) + 1):
        telhado = (instancias == i).astype(np.uint8)
        if telhado.sum() == 0:   # polígono que ficou todo coberto por outro
            continue
        interior = cv2.erode(telhado, kernel, iterations=LARGURA_BORDA).astype(bool)
        borda = telhado.astype(bool) & ~interior
        mascara[interior] = CLASSE_INTERIOR
        mascara[borda] = CLASSE_BORDA
    return mascara, instancias


def imagem_exemplo(img, mascara):
    # imagem original ao lado da imagem com a máscara por cima
    cor = img.copy()
    cor[mascara == CLASSE_INTERIOR] = (0, 200, 0)
    cor[mascara == CLASSE_BORDA] = (0, 0, 255)
    misturada = cv2.addWeighted(img, 0.5, cor, 0.5, 0)
    return np.hstack([img, misturada])


def texto_proporcao(prop):
    return [f"    fundo:    {prop[0] * 100:6.2f} %",
            f"    interior: {prop[1] * 100:6.2f} %",
            f"    borda:    {prop[2] * 100:6.2f} %"]


def main():
    if DESTINO.exists():
        raise SystemExit(f"A pasta {DESTINO} já existe. Apagar antes de rodar de novo.")
    if not (ORIGEM / "images").exists():
        raise SystemExit(f"Pasta {ORIGEM / 'images'} não encontrada.")

    (DESTINO / "exemplos").mkdir(parents=True)
    estatisticas = [f"LARGURA_BORDA = {LARGURA_BORDA} px", ""]
    total_pixels = np.zeros(3, np.int64)

    for divisao in DIVISOES:
        pasta_img = DESTINO / divisao / "imagens"
        pasta_mascara = DESTINO / divisao / "mascaras"
        pasta_inst = DESTINO / divisao / "instancias"
        for p in (pasta_img, pasta_mascara, pasta_inst):
            p.mkdir(parents=True)

        imagens = sorted(x for x in (ORIGEM / "images" / divisao).iterdir()
                         if x.suffix.lower() in EXTENSOES)
        pixels = np.zeros(3, np.int64)
        n_telhados = 0
        n_vazias = 0
        n_tamanho_diferente = 0
        # salva 3 exemplos por divisão: a primeira, a do meio e a última imagem
        exemplos = {0, len(imagens) // 2, len(imagens) - 1}

        for k, arq in enumerate(imagens):
            img = cv2.imread(str(arq), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Erro ao ler {arq}")
            h, w = img.shape[:2]
            if (h, w) != (512, 512):
                n_tamanho_diferente += 1

            poligonos = ler_poligonos(ORIGEM / "labels" / divisao / (arq.stem + ".txt"), h, w)
            mascara, instancias = gerar_mascara(poligonos, h, w)

            shutil.copy2(arq, pasta_img / arq.name)
            cv2.imwrite(str(pasta_mascara / (arq.stem + ".png")), mascara)
            cv2.imwrite(str(pasta_inst / (arq.stem + ".png")), instancias)

            pixels += np.bincount(mascara.ravel(), minlength=3)[:3]
            n_telhados += len(poligonos)
            if not poligonos:
                n_vazias += 1

            if k in exemplos:
                cv2.imwrite(str(DESTINO / "exemplos" / f"{divisao}_{arq.stem[:40]}.png"),
                            imagem_exemplo(img, mascara))

        total_pixels += pixels
        bloco = [f"[{divisao}] {len(imagens)} imagens / {n_telhados} telhados / {n_vazias} sem telhado"]
        bloco += texto_proporcao(pixels / pixels.sum())
        if n_tamanho_diferente:
            bloco.append(f"    ATENÇÃO: {n_tamanho_diferente} imagens com tamanho diferente de 512x512")
        estatisticas += bloco + [""]
        print("\n".join(bloco))

    final = ["[todas] proporção de pixels"] + texto_proporcao(total_pixels / total_pixels.sum())
    estatisticas += final
    print("\n".join(final))

    (DESTINO / "estatisticas.txt").write_text("\n".join(estatisticas) + "\n", encoding="utf-8")
    print(f"\nMáscaras criadas em {DESTINO}")
    print(f"Exemplos em {DESTINO / 'exemplos'}")


if __name__ == "__main__":
    main()