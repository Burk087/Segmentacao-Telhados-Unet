"""
Recorte das ortofotos

Divide cada ortofoto em recortes quadrados de 1024 x 1024 pixels, sem mudar a resolução (a imagem só é dividida, nenhum pixel é alterado)

Quando a ortofoto não divide certinho por 1024, o último recorte de cada linha e de cada coluna é puxado para trás até encostar na borda. Assim nenhuma parte da imagem é perdida, mas esses últimos recortes ficam um pouco sobrepostos aos vizinhos.

"""

import argparse
from pathlib import Path
from PIL import Image

# as ortofotos são muito grandes, ai isso tira o limite de tamanho de imagem do Pillow
Image.MAX_IMAGE_PIXELS = None

TAMANHO = 1024   # lado do recorte em pixels
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def calcular_posicoes(total, tamanho):
    # posições iniciais dos recortes ao longo de um eixo (largura ou altura)
    if total <= tamanho:
        return [0]
    posicoes = list(range(0, total - tamanho + 1, tamanho))
    # se sobrou um pedaço no final, o último recorte encosta na borda
    if posicoes[-1] + tamanho < total:
        posicoes.append(total - tamanho)
    return posicoes


def recortar(caminho, entrada, pasta_saida):
    with Image.open(caminho) as img:
        img.load()
        largura, altura = img.size
        xs = calcular_posicoes(largura, TAMANHO)
        ys = calcular_posicoes(altura, TAMANHO)

        # nome base do recorte
        relativo = caminho.relative_to(entrada)
        prefixo = "_".join(relativo.with_suffix("").parts)

        salvos = 0
        for linha, y in enumerate(ys):
            for coluna, x in enumerate(xs):
                caixa = (x, y, min(x + TAMANHO, largura), min(y + TAMANHO, altura))
                recorte = img.crop(caixa)

                # imagem menor que 1024: completa com preto até o tamanho do recorte
                if recorte.size != (TAMANHO, TAMANHO):
                    fundo = Image.new(img.mode, (TAMANHO, TAMANHO), 0)
                    fundo.paste(recorte, (0, 0))
                    recorte = fundo

                nome = f"{prefixo}_l{linha:03d}_c{coluna:03d}.png"
                recorte.save(pasta_saida / nome, compress_level=6)
                salvos += 1

        print(f"  {caminho.name}  ({largura}x{altura})  ->  "
              f"grade {len(xs)}x{len(ys)}, {salvos} recortes salvos")
        return salvos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", required=True, help="pasta com as ortofotos")
    parser.add_argument("--saida", required=True, help="pasta onde os recortes serão salvos")
    args = parser.parse_args()

    entrada = Path(args.entrada)
    saida = Path(args.saida)
    if not entrada.is_dir():
        raise SystemExit(f"Pasta de entrada não encontrada: {entrada}")
    saida.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(p for p in entrada.rglob("*")
                      if p.is_file() and p.suffix.lower() in EXTENSOES)
    if not arquivos:
        raise SystemExit("Nenhuma imagem encontrada na pasta de entrada.")
    print(f"{len(arquivos)} ortofoto(s) encontrada(s). Recortes de {TAMANHO} x {TAMANHO} px.\n")

    total = 0
    for i, caminho in enumerate(arquivos, 1):
        print(f"[{i}/{len(arquivos)}]")
        try:
            total += recortar(caminho, entrada, saida)
        except Exception as erro:
            print(f"  ERRO em {caminho.name}: {erro}")

    print(f"\nConcluído. {total} recortes gerados em:\n{saida}")


if __name__ == "__main__":
    main()