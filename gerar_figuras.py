"""
Geração das figuras

Gera as figuras do TCC (PNG, 300 dpi) a partir dos resultados do avaliar_unet.py.
Antes, é preciso rodar o avaliar_unet.py do mesmo experimento, com as mesmas opções.

Como rodar (dentro da pasta do projeto):
    python gerar_figuras.py --experimento unet_v3
    python gerar_figuras.py --experimento unet_v3 --conjunto teste
    python gerar_figuras.py --experimento unet_v3 --comparar unet_v1 unet_v2 unet_v3 unet_v4 unet_v5 unet_v6
    python gerar_figuras.py --experimento unet_v3 --tta

"""

import argparse
import csv
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")  # salva as figuras sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))

# aparência das figuras
LARGURA = 6.3   # polegadas
DPI = 300
AZUIS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MAPA_AZUL = LinearSegmentedColormap.from_list("azul", AZUIS)
COR_TEXTO = "#0b0b0b"
COR_TEXTO_2 = "#52514e"
COR_GRADE = "#e4e3df"
CORES_TIPO = {"gaussian": "#2a78d6", "sensor": "#eb6834", "fog": "#1baf7a",
              "illumination": "#eda100", "occlusion": "#e87ba4"}
NOMES_TIPO = {"limpo": "Limpo", "gaussian": "Ruído gaussiano", "sensor": "Ruído de sensor",
              "fog": "Névoa", "illumination": "Iluminação", "occlusion": "Oclusão"}
NOMES_CURTOS = {"limpo": "Limpo", "gaussian": "Ruído\ngaussiano", "sensor": "Ruído de\nsensor",
                "fog": "Névoa", "illumination": "Iluminação", "occlusion": "Oclusão"}
ORDEM_TIPOS = ["gaussian", "sensor", "fog", "illumination", "occlusion"]
COR_VP = "#1baf7a"   # acerto
COR_FP = "#eb6834"   # falso positivo
COR_FN = "#2a78d6"   # telhado não detectado
CLASSES = ["Fundo", "Interior", "Borda"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": COR_TEXTO_2, "axes.labelcolor": COR_TEXTO, "xtick.color": COR_TEXTO_2,
    "ytick.color": COR_TEXTO_2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": DPI, "savefig.bbox": "tight", "figure.dpi": 100,
})


def virgula(v, casas=2):
    # número com vírgula decimal
    return f"{v:.{casas}f}".replace(".", ",")


FORMATO_VIRGULA = FuncFormatter(lambda v, _: virgula(v, 2))


def para_numero(texto):
    # converte "0,8123" para 0.8123; se não for número, devolve o texto
    try:
        return float(str(texto).replace(",", "."))
    except ValueError:
        return texto


def ler_csv(caminho):
    linhas = []
    with open(caminho, encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f, delimiter=";"):
            linhas.append({k: para_numero(v) for k, v in linha.items()})
    return linhas


def salvar(fig, caminho):
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  {caminho.name}")


def pegar_limpo(linhas):
    return next(l for l in linhas if l["condicao"] == "limpo")


def tirar_bordas(ax):
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)


def grade_y(ax):
    ax.grid(axis="y", color=COR_GRADE, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------
# matrizes de confusão
# ---------------------------------------------------------------

def desenhar_matriz(ax, m, rotulos_lin, rotulos_col, titulo=None):
    m = np.asarray(m, dtype=np.float64)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap=MAPA_AZUL, vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if norm[i, j] > 0.55:
                cor = "white"
            else:
                cor = COR_TEXTO
            milhar = f"{int(m[i, j]):,}".replace(",", ".")
            ax.text(j, i, f"{virgula(norm[i, j] * 100, 1)}%\n({milhar})",
                    ha="center", va="center", color=cor, fontsize=8)
    ax.set_xticks(range(len(rotulos_col)), rotulos_col)
    ax.set_yticks(range(len(rotulos_lin)), rotulos_lin)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    tirar_bordas(ax)
    if titulo:
        ax.set_title(titulo, color=COR_TEXTO)


def matriz_binaria(m):
    # junta interior e borda em "telhado": [[VP, FN], [FP, VN]]
    m = np.asarray(m)
    vn = m[0, 0]
    fp = m[0, 1:].sum()
    fn = m[1:, 0].sum()
    vp = m[1:, 1:].sum()
    return np.array([[vp, fn], [fp, vn]])


def figuras_matrizes(matrizes, saida):
    limpo = matrizes["limpo"]
    pert = sum(m for k, m in matrizes.items() if k != "limpo")

    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, LARGURA * 0.52))
    desenhar_matriz(ax, limpo, CLASSES, CLASSES)
    salvar(fig, saida / "fig01_matriz_confusao_3classes_limpo.png")

    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, LARGURA * 0.52))
    desenhar_matriz(ax, pert, CLASSES, CLASSES)
    salvar(fig, saida / "fig02_matriz_confusao_3classes_perturbado.png")

    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, LARGURA * 0.42))
    for ax, m, t in zip(axs, [limpo, pert], ["Imagens limpas", "Imagens perturbadas"]):
        desenhar_matriz(ax, matriz_binaria(m), ["Telhado", "Fundo"], ["Telhado", "Fundo"], t)
    axs[1].set_ylabel("")
    salvar(fig, saida / "fig03_matriz_confusao_telhado_fundo.png")


# ---------------------------------------------------------------
# mapas de calor (tipo x nível)
# ---------------------------------------------------------------

def tabela_tipo_nivel(linhas, chave):
    # tabela com uma linha por tipo e uma coluna por nível; a 1a coluna é o limpo
    limpo = pegar_limpo(linhas)[chave]
    niveis = sorted({l["nivel"] for l in linhas if l["condicao"] != "limpo"})
    tab = np.full((len(ORDEM_TIPOS), len(niveis) + 1), np.nan)
    tab[:, 0] = limpo
    for l in linhas:
        if l["tipo"] in ORDEM_TIPOS:
            tab[ORDEM_TIPOS.index(l["tipo"]), niveis.index(l["nivel"]) + 1] = l[chave]
    return tab, [0.0] + niveis


def mapa_calor(linhas, chave, arquivo, saida):
    tab, niveis = tabela_tipo_nivel(linhas, chave)
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.48))
    vmin = np.nanmin(tab) - 0.02
    vmax = np.nanmax(tab)
    ax.imshow(tab, cmap=MAPA_AZUL, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            if np.isnan(tab[i, j]):
                continue
            relativo = (tab[i, j] - vmin) / max(vmax - vmin, 1e-9)
            if relativo > 0.6:
                cor = "white"
            else:
                cor = COR_TEXTO
            ax.text(j, i, virgula(tab[i, j], 3), ha="center", va="center", fontsize=8, color=cor)
    ax.set_xticks(range(len(niveis)), ["0 (limpo)"] + [virgula(n, 2) for n in niveis[1:]])
    ax.set_yticks(range(len(ORDEM_TIPOS)), [NOMES_TIPO[t] for t in ORDEM_TIPOS])
    ax.set_xlabel("Intensidade da perturbação")
    tirar_bordas(ax)
    ax.axvline(0.5, color="white", lw=3)   # separa a coluna do limpo
    salvar(fig, saida / arquivo)


# ---------------------------------------------------------------
# gráficos de linha e de barras
# ---------------------------------------------------------------

def figura_niveis(linhas, saida):
    metricas = [("px_iou", "IoU por pixel"), ("tel_f1", "F1 por telhado"),
                ("AP50_95", "AP@0,5:0,95")]
    fig, axs = plt.subplots(1, 3, figsize=(LARGURA, LARGURA * 0.42), sharex=True)
    for ax, (chave, nome) in zip(axs, metricas):
        tab, niveis = tabela_tipo_nivel(linhas, chave)
        for i, t in enumerate(ORDEM_TIPOS):
            ax.plot(niveis, tab[i], color=CORES_TIPO[t], lw=1.6, marker="o", ms=3.5,
                    label=NOMES_TIPO[t])
        ax.set_title(nome, color=COR_TEXTO)
        ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
        ax.set_xticks(niveis, ["0"] + [virgula(n, 2) for n in niveis[1:]], rotation=0)
        grade_y(ax)
    axs[1].set_xlabel("Intensidade da perturbação (0 = limpo)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    salvar(fig, saida / "fig07_metricas_por_nivel.png")


def medias_por_tipo(linhas, chaves):
    # limpo + média dos 4 níveis de cada tipo
    limpo = pegar_limpo(linhas)
    medias = {"limpo": {k: limpo[k] for k in chaves}}
    for t in ORDEM_TIPOS:
        do_tipo = [l for l in linhas if l["tipo"] == t]
        if do_tipo:
            medias[t] = {k: float(np.mean([l[k] for l in do_tipo])) for k in chaves}
    return medias


def figura_barras_telhado(linhas, saida):
    chaves = [("tel_precisao", "Precisão"), ("tel_recall", "Recall"), ("tel_f1", "F1")]
    medias = medias_por_tipo(linhas, [c for c, _ in chaves])
    grupos = list(medias)
    cores = ["#2a78d6", "#eb6834", "#1baf7a"]
    x = np.arange(len(grupos))
    largura_barra = 0.26
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.45))
    for k, ((chave, nome), cor) in enumerate(zip(chaves, cores)):
        valores = [medias[g][chave] for g in grupos]
        barras = ax.bar(x + (k - 1) * (largura_barra + 0.015), valores, largura_barra,
                        color=cor, label=nome)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, virgula(v, 2), ha="center",
                    va="bottom", fontsize=6.5, color=COR_TEXTO_2)
    ax.set_xticks(x, [NOMES_CURTOS[g] for g in grupos])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
    grade_y(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    salvar(fig, saida / "fig08_precisao_recall_f1_por_condicao.png")


def contar_acertos(l):
    # VP, FP e FN a partir da precisão e das quantidades salvas no metricas.csv
    vp = round(l["tel_precisao"] * l["tel_previstos"])
    return vp, l["tel_previstos"] - vp, l["tel_reais"] - vp


def figura_contagens(linhas, saida):
    limpo = pegar_limpo(linhas)
    grupos = ["Limpo"]
    dados = [contar_acertos(limpo)]
    for t in ORDEM_TIPOS:
        l = next((l for l in linhas if l["tipo"] == t and abs(l["nivel"] - 1.0) < 1e-6), None)
        if l:
            grupos.append(f"{NOMES_CURTOS[t]}\n(1,00)")
            dados.append(contar_acertos(l))
    dados = np.array(dados, dtype=float)

    x = np.arange(len(grupos))
    largura_barra = 0.26
    series = [("Acertos (VP)", COR_VP), ("Falsos positivos (FP)", COR_FP),
              ("Telhados perdidos (FN)", COR_FN)]
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.45))
    for k, (nome, cor) in enumerate(series):
        barras = ax.bar(x + (k - 1) * (largura_barra + 0.015), dados[:, k], largura_barra,
                        color=cor, label=nome)
        for b, v in zip(barras, dados[:, k]):
            ax.text(b.get_x() + b.get_width() / 2, v + dados.max() * 0.01, f"{int(v)}",
                    ha="center", va="bottom", fontsize=6.5, color=COR_TEXTO_2)
    ax.set_xticks(x, grupos)
    ax.set_ylabel("Número de telhados")
    grade_y(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    salvar(fig, saida / "fig09_acertos_erros_por_telhado.png")


# ---------------------------------------------------------------
# evolução das versões (validação)
# ---------------------------------------------------------------

def figura_evolucao(versoes, saida):
    dados = []
    for v in versoes:
        arq = PASTA / "experimentos" / v / "avaliacao_valid_calibrado" / "metricas.csv"
        if not arq.exists():
            print(f"  (pulando {v}: {arq} não encontrado)")
            continue
        linhas = ler_csv(arq)
        limpo = pegar_limpo(linhas)
        pert = [l for l in linhas if l["condicao"] != "limpo"]
        dados.append((v.replace("unet_", "").upper(),
                      limpo["tel_f1"], np.mean([l["tel_f1"] for l in pert]),
                      limpo["AP50_95"], np.mean([l["AP50_95"] for l in pert])))
    if not dados:
        return

    nomes = [d[0] for d in dados]
    x = np.arange(len(nomes))
    largura_barra = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, LARGURA * 0.42))
    # posições em "dados": 1 e 2 = F1 limpo/perturbado, 3 e 4 = AP limpo/perturbado
    for ax, (i, j), titulo in zip(axs, [(1, 2), (3, 4)], ["F1 por telhado", "AP@0,5:0,95"]):
        for k, (idx, nome, cor) in enumerate([(i, "Limpo", "#2a78d6"),
                                              (j, "Perturbado (média)", "#eb6834")]):
            valores = [d[idx] for d in dados]
            barras = ax.bar(x + (k - 0.5) * (largura_barra + 0.02), valores, largura_barra,
                            color=cor, label=nome)
            for b, v in zip(barras, valores):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, virgula(v, 3), ha="center",
                        va="bottom", fontsize=5.5, color=COR_TEXTO_2, rotation=90)
        ax.set_xticks(x, nomes)
        ax.set_title(titulo, color=COR_TEXTO)
        topo = max(max(d[i], d[j]) for d in dados)
        ax.set_ylim(0, topo * 1.18)
        ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
        grade_y(ax)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    salvar(fig, saida / "fig10_evolucao_versoes.png")


# ---------------------------------------------------------------
# mapa de acertos e erros sobre a imagem
# ---------------------------------------------------------------

def escolher_imagem(indice, instancias):
    # se não for informado o índice, usa a imagem com mais telhados anotados
    if indice is not None:
        return indice % len(instancias)
    contagem = [int(inst.max()) for inst in instancias]
    k = int(np.argmax(contagem))
    print(f"  imagem dos mapas: índice {k} ({contagem[k]} telhados). Para outra, use --imagem N")
    return k


def hex_para_rgb(cor):
    return np.array([int(cor[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def figura_mapas_erro(args, exp, saida):
    import torch
    from avaliar_unet import (carregar_modelos, carregar_pos, montar_condicoes, prever_modelos,
                              separar_telhados)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = carregar_modelos(args.modelos or [args.experimento], dispositivo)
    pp, _ = carregar_pos(exp, args.tta)

    nomes_img, condicoes, mascaras, instancias = montar_condicoes(args.conjunto == "teste")
    k = escolher_imagem(args.imagem, instancias)
    nomes = ["limpo"] + [f"{t}_1.00" for t in ORDEM_TIPOS if f"{t}_1.00" in condicoes]
    real = mascaras[k] > 0

    paineis = []
    for nome in nomes:
        caminho = condicoes[nome][k]
        prob = prever_modelos(modelos, [caminho], dispositivo, args.tta)[0]
        rotulos, _ = separar_telhados(prob, pp)
        previsto = rotulos > 0
        img = cv2.cvtColor(cv2.imread(str(caminho), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # pinta acertos, falsos positivos e telhados perdidos
        pintada = img.astype(np.float32).copy()
        for regiao, cor in [(previsto & real, COR_VP), (previsto & ~real, COR_FP),
                            (~previsto & real, COR_FN)]:
            pintada[regiao] = 0.4 * pintada[regiao] + 0.6 * hex_para_rgb(cor)
        pintada = pintada.astype(np.uint8)

        # contorno branco em cada telhado previsto
        for i in range(1, int(rotulos.max()) + 1):
            contornos, _ = cv2.findContours((rotulos == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(pintada, contornos, -1, (255, 255, 255), 1)

        if nome == "limpo":
            titulo = NOMES_TIPO["limpo"]
        else:
            titulo = NOMES_TIPO[nome.rsplit("_", 1)[0]] + " (1,00)"
        paineis.append((pintada, titulo))

    colunas = 3
    n_linhas = int(np.ceil(len(paineis) / colunas))
    fig, axs = plt.subplots(n_linhas, colunas,
                            figsize=(LARGURA, LARGURA / colunas * n_linhas + 0.4))
    eixos = np.atleast_1d(axs).ravel()
    for ax in eixos:
        ax.axis("off")
    for ax, (img, titulo) in zip(eixos, paineis):
        ax.imshow(img)
        ax.set_title(titulo, color=COR_TEXTO, fontsize=8)
    fig.legend(handles=[Patch(color=COR_VP, label="Acerto (telhado previsto e real)"),
                        Patch(color=COR_FP, label="Falso positivo"),
                        Patch(color=COR_FN, label="Telhado não detectado")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    salvar(fig, saida / f"fig11_mapa_acertos_erros_img{k:02d}.png")


# ---------------------------------------------------------------
# curvas de treino
# ---------------------------------------------------------------

def figura_curvas(exp, saida):
    arq = exp / "historico.csv"
    if not arq.exists():
        return
    hist = ler_csv(arq)
    epocas = [l["epoca"] for l in hist]
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, LARGURA * 0.4))

    axs[0].plot(epocas, [l["perda"] for l in hist], color="#2a78d6", lw=1.4, label="Perda total")
    axs[0].plot(epocas, [l["ce"] for l in hist], color="#eb6834", lw=1.4, label="Entropia cruzada")
    axs[0].plot(epocas, [l["dice"] for l in hist], color="#1baf7a", lw=1.4, label="Dice")
    axs[0].set_title("Perda no treinamento", color=COR_TEXTO)

    axs[1].plot(epocas, [l["iou_limpo"] for l in hist], color="#2a78d6", lw=1.4,
                label="Validação limpa")
    axs[1].plot(epocas, [l["iou_perturbado"] for l in hist], color="#eb6834", lw=1.4,
                label="Validação perturbada")
    # linha tracejada na época escolhida
    melhor = max(hist, key=lambda l: l["criterio"])
    axs[1].axvline(melhor["epoca"], color=COR_TEXTO_2, lw=0.8, ls="--")
    axs[1].text(melhor["epoca"], axs[1].get_ylim()[0], f"melhor época ({int(melhor['epoca'])}) ",
                fontsize=7, color=COR_TEXTO_2, va="bottom", ha="right")
    axs[1].set_title("IoU de telhado por pixel", color=COR_TEXTO)

    for ax in axs:
        ax.set_xlabel("Época")
        ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
        ax.grid(axis="y", color=COR_GRADE, lw=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    salvar(fig, saida / "fig12_curvas_treino.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimento", default="unet_v3")
    parser.add_argument("--conjunto", choices=["valid", "teste"], default="valid")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--comparar", nargs="*", default=[])
    parser.add_argument("--imagem", type=int, default=None,
                        help="índice da imagem do mapa de erros (padrão: a com mais telhados)")
    parser.add_argument("--sem-mapas", action="store_true", help="não gera a fig11 (não precisa de GPU)")
    parser.add_argument("--modelos", nargs="+", default=None, help="conjunto de modelos (fig11)")
    args = parser.parse_args()

    exp = PASTA / "experimentos" / args.experimento
    sufixo = "_tta" if args.tta else ""
    aval = exp / f"avaliacao_{args.conjunto}{sufixo}_calibrado"
    if not aval.exists():
        aval = exp / f"avaliacao_{args.conjunto}{sufixo}"
    if not (aval / "metricas.csv").exists():
        raise SystemExit(f"{aval / 'metricas.csv'} não encontrado. Rodar antes o avaliar_unet.py.")
    saida = exp / f"figuras_{args.conjunto}{sufixo}"
    saida.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {aval}\nFiguras em {saida}")

    linhas = ler_csv(aval / "metricas.csv")
    matrizes = {}
    for l in ler_csv(aval / "matriz_confusao.csv"):
        matrizes.setdefault(l["condicao"], []).append(
            [l["prev_fundo"], l["prev_interior"], l["prev_borda"]])
    matrizes = {k: np.array(v, dtype=np.float64) for k, v in matrizes.items()}

    figuras_matrizes(matrizes, saida)
    mapa_calor(linhas, "tel_f1", "fig04_mapa_calor_f1_telhado.png", saida)
    mapa_calor(linhas, "tel_recall", "fig05_mapa_calor_recall_telhado.png", saida)
    mapa_calor(linhas, "px_iou", "fig06_mapa_calor_iou_pixel.png", saida)
    figura_niveis(linhas, saida)
    figura_barras_telhado(linhas, saida)
    figura_contagens(linhas, saida)
    if args.comparar:
        figura_evolucao(args.comparar, saida)
    if not args.sem_mapas:
        figura_mapas_erro(args, exp, saida)
    figura_curvas(exp, saida)


if __name__ == "__main__":
    main()