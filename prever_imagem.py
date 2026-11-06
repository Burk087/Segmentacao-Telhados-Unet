"""
Previsão em imagens novas

Usa o modelo final (U-Net v3) para encontrar os telhados em imagens aéreas novas.

Como rodar (dentro da pasta do projeto):
    python prever_imagem.py --entrada foto.png
    python prever_imagem.py --entrada pasta_com_imagens --saida resultados

As imagens devem ser aéreas, vistas de cima, como os recortes de 1024 x 1024 usados no treinamento (gerados pelo recortar_ortofotos.py). 
A imagem é reduzida para 512 x 512 na previsão e o resultado volta para o tamanho original.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch


PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from treinar_unet import UNet
from avaliar_unet import POS_PADRAO, separar_telhados

EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TAMANHO = 512   # tamanho de entrada da rede


def carregar_modelo(pasta_modelo, dispositivo):
    # carrega a rede, a normalização usada no treino e os parâmetros do pós-processamento
    checkpoint = torch.load(pasta_modelo / "melhor.pt", map_location="cpu", weights_only=False)
    modelo = UNet(checkpoint["config"]).to(dispositivo)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.eval()

    norm = checkpoint["normalizacao"]
    media = torch.tensor(norm["media"], device=dispositivo).view(1, 3, 1, 1)
    desvio = torch.tensor(norm["desvio"], device=dispositivo).view(1, 3, 1, 1)

    arq = pasta_modelo / "posprocessamento.json"
    if arq.exists():
        dados = json.loads(arq.read_text(encoding="utf-8"))
        pp = {k: dados[k] for k in POS_PADRAO}
    else:
        pp = dict(POS_PADRAO)
        print("Aviso: posprocessamento.json não encontrado, usando os valores padrão.")
    return modelo, media, desvio, pp


@torch.no_grad()
def prever(modelo, media, desvio, img, dispositivo):
    # retorna a probabilidade de cada classe (3, 512, 512)
    x = cv2.resize(img, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x).to(dispositivo).permute(2, 0, 1)[None].float() / 255.0
    x = (x - media) / desvio
    with torch.autocast("cuda", enabled=dispositivo.type == "cuda"):
        p = torch.softmax(modelo(x).float(), dim=1)
    return p[0].cpu().numpy()


def pintar_telhados(img, rotulos):
    # pinta cada telhado de uma cor e desenha o contorno em branco
    rng = np.random.default_rng(0)
    cores = rng.integers(60, 255, (int(rotulos.max()) + 1, 3)).astype(np.float32)
    resultado = img.astype(np.float32).copy()
    m = rotulos > 0
    resultado[m] = 0.45 * resultado[m] + 0.55 * cores[rotulos[m]]
    resultado = resultado.astype(np.uint8)
    for i in range(1, int(rotulos.max()) + 1):
        contornos, _ = cv2.findContours((rotulos == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(resultado, contornos, -1, (255, 255, 255), 1)
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", required=True, help="imagem ou pasta com imagens")
    parser.add_argument("--saida", default="previsoes")
    parser.add_argument("--modelo", default=str(PASTA / "experimentos" / "unet_v3"),
                        help="pasta do modelo (com melhor.pt e posprocessamento.json)")
    args = parser.parse_args()

    entrada = Path(args.entrada)
    if entrada.is_file():
        imagens = [entrada]
    else:
        imagens = sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSOES)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada em {entrada}")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, media, desvio, pp = carregar_modelo(Path(args.modelo), dispositivo)
    print(f"Modelo: {args.modelo} / dispositivo: {dispositivo} / pós-processamento: {pp}\n")

    linhas_csv = []
    for caminho in imagens:
        img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  erro ao ler {caminho.name}")
            continue
        altura, largura = img.shape[:2]

        prob = prever(modelo, media, desvio, img, dispositivo)
        rotulos, confiancas = separar_telhados(prob, pp)
        classe = prob.argmax(0).astype(np.uint8)

        # volta o resultado para o tamanho original da imagem
        if (altura, largura) != (TAMANHO, TAMANHO):
            rotulos = cv2.resize(rotulos.astype(np.int32).astype(np.float32), (largura, altura),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
            classe = cv2.resize(classe, (largura, altura), interpolation=cv2.INTER_NEAREST)
        areas = np.bincount(rotulos.ravel(), minlength=len(confiancas) + 1)[1:]

        cv2.imwrite(str(saida / f"{caminho.stem}_telhados.png"), pintar_telhados(img, rotulos))
        cv2.imwrite(str(saida / f"{caminho.stem}_mascara.png"), classe)
        cv2.imwrite(str(saida / f"{caminho.stem}_instancias.png"), rotulos.astype(np.uint16))
        for i, (area, conf) in enumerate(zip(areas, confiancas), 1):
            linhas_csv.append([caminho.name, i, int(area), f"{conf:.3f}".replace(".", ",")])
        print(f"  {caminho.name}: {len(confiancas)} telhados")

    with open(saida / "telhados.csv", "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(["imagem", "telhado", "area_pixels", "confianca"])
        escritor.writerows(linhas_csv)
    print(f"\nResultados em {saida.resolve()}")


if __name__ == "__main__":
    main()