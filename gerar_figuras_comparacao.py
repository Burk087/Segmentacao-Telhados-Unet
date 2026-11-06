"""
Figuras de comparação entre os modelos finais

Compara, no conjunto de teste:
    v3    U-Net v3 (modelo final), 3 sementes (média e desvio padrão)
    v5    U-Net v5 (segunda melhor versão)
    conj  conjunto das 3 redes v3 (média das probabilidades)

Antes, é preciso rodar o avaliar_unet.py no teste para todos esses modelos.

Como rodar (dentro da pasta do projeto):
    python gerar_figuras_comparacao.py
    python gerar_figuras_comparacao.py --imagem 36
    python gerar_figuras_comparacao.py --sem-mapas

Gera em experimentos/comparacao_final/
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # serve só pra salva as figuras sem abrir janela
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from gerar_figuras import (COR_FN, COR_FP, COR_GRADE, COR_TEXTO, COR_TEXTO_2, COR_VP,
                           FORMATO_VIRGULA, LARGURA, NOMES_CURTOS, NOMES_TIPO, ORDEM_TIPOS,
                           escolher_imagem, hex_para_rgb, ler_csv, salvar, virgula)

CORES = {"v3": "#2a78d6", "v5": "#eb6834", "conj": "#1baf7a"}
ROTULOS = {"v3": "U-Net v3 (final, média de 3 sementes)", "v5": "U-Net v5 (segundo melhor)",
           "conj": "Conjunto das 3 redes v3"}
# azul = vantagem da v3, vermelho = vantagem da v5
DIVERGENTE = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#f0efec", "#f19897", "#b8302f"])
NIVEIS = [0.25, 0.5, 0.75, 1.0]


def carregar(nome, conjunto):
    # lê o metricas.csv do experimento (de preferência o calibrado)
    pasta = PASTA / "experimentos" / nome
    for sub in (f"avaliacao_{conjunto}_calibrado", f"avaliacao_{conjunto}"):
        arq = pasta / sub / "metricas.csv"
        if arq.exists():
            return ler_csv(arq)
    raise SystemExit(f"Métricas de {nome} ({conjunto}) não encontradas. Rodar o avaliar_unet.py.")


def valor(linhas, chave, tipo=None, nivel=None):
    # tipo="limpo": valor do limpo
    # tipo=t: média dos 4 níveis do tipo t / nivel=n: média dos 5 tipos no nível n
    # sem tipo e sem nível: média das 20 condições perturbadas
    if tipo == "limpo":
        return next(l[chave] for l in linhas if l["condicao"] == "limpo")
    selecionadas = []
    for l in linhas:
        if l["condicao"] == "limpo":
            continue
        if tipo is not None and l["tipo"] != tipo:
            continue
        if nivel is not None and abs(l["nivel"] - nivel) >= 1e-6:
            continue
        selecionadas.append(l[chave])
    return float(np.mean(selecionadas))


def media_desvio(lista_linhas, chave, **kw):
    # média e desvio padrão entre as sementes
    v = np.array([valor(l, chave, **kw) for l in lista_linhas])
    if len(v) > 1:
        return v.mean(), v.std(ddof=1)
    return v.mean(), 0.0


def valor_condicao(linhas, chave, tipo, nivel):
    return next(l[chave] for l in linhas if l["tipo"] == tipo and abs(l["nivel"] - nivel) < 1e-6)


def grade_y(ax):
    ax.grid(axis="y", color=COR_GRADE, lw=0.6)
    ax.set_axisbelow(True)


def fig_barras(final, v5, conj, saida):
    grupos = ["limpo"] + ORDEM_TIPOS
    x = np.arange(len(grupos))
    largura_barra = 0.26
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.45))

    v3 = [media_desvio(final, "tel_f1", tipo=g) for g in grupos]
    series = [("v3", [m for m, _ in v3], [s for _, s in v3]),
              ("v5", [valor(v5, "tel_f1", tipo=g) for g in grupos], None),
              ("conj", [valor(conj, "tel_f1", tipo=g) for g in grupos], None)]
    for k, (chave, valores, erro) in enumerate(series):
        barras = ax.bar(x + (k - 1) * (largura_barra + 0.015), valores, largura_barra,
                        color=CORES[chave], label=ROTULOS[chave], yerr=erro,
                        error_kw=dict(ecolor=COR_TEXTO, lw=0.8, capsize=2))
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, 0.705, virgula(v, 3), ha="center",
                    va="bottom", fontsize=5.5, color="white", rotation=90)
    ax.set_xticks(x, [NOMES_CURTOS[g] for g in grupos])
    ax.set_ylim(0.7, 0.86)   # eixo começa em 0,70 para as diferenças aparecerem
    ax.set_ylabel("F1 por telhado")
    ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
    grade_y(ax)
    ax.legend(frameon=False, ncol=1, loc="lower center", fontsize=7, bbox_to_anchor=(0.5, 1.0))
    salvar(fig, saida / "fig13_comparacao_f1_por_condicao.png")


def fig_niveis(final, v5, conj, saida):
    eixo_x = [0.0] + NIVEIS
    metricas = [("px_iou", "IoU por pixel"), ("tel_f1", "F1 por telhado"),
                ("AP50_95", "AP@0,5:0,95")]
    fig, axs = plt.subplots(1, 3, figsize=(LARGURA, LARGURA * 0.42))
    for ax, (chave, nome) in zip(axs, metricas):
        # v3: linha da média e faixa de +- 1 desvio padrão entre as sementes
        estat = [media_desvio(final, chave, tipo="limpo")]
        estat += [media_desvio(final, chave, nivel=n) for n in NIVEIS]
        media = np.array([a for a, _ in estat])
        desvio = np.array([b for _, b in estat])
        ax.fill_between(eixo_x, media - desvio, media + desvio, color=CORES["v3"], alpha=0.18, lw=0)
        ax.plot(eixo_x, media, color=CORES["v3"], lw=1.6, marker="o", ms=3.5, label=ROTULOS["v3"])

        for nome_modelo, linhas in (("v5", v5), ("conj", conj)):
            valores = [valor(linhas, chave, tipo="limpo")]
            valores += [valor(linhas, chave, nivel=n) for n in NIVEIS]
            ax.plot(eixo_x, valores, color=CORES[nome_modelo], lw=1.6, marker="o", ms=3.5,
                    label=ROTULOS[nome_modelo])
        ax.set_title(nome, color=COR_TEXTO)
        ax.set_xticks(eixo_x, ["0"] + [virgula(n, 2) for n in NIVEIS])
        ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
        ax.grid(axis="y", color=COR_GRADE, lw=0.6)
    axs[1].set_xlabel("Intensidade da perturbação (média dos 5 tipos; 0 = limpo)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.tight_layout()
    salvar(fig, saida / "fig14_comparacao_por_nivel.png")


def fig_diferenca(final, v5, saida):
    # tabela da diferença v5 - v3 (média das sementes); 1a coluna = limpo
    tab = np.zeros((len(ORDEM_TIPOS), len(NIVEIS) + 1))
    for i, t in enumerate(ORDEM_TIPOS):
        tab[i, 0] = valor(v5, "tel_f1", tipo="limpo") - media_desvio(final, "tel_f1", tipo="limpo")[0]
        for j, n in enumerate(NIVEIS):
            f1_v5 = valor_condicao(v5, "tel_f1", t, n)
            f1_v3 = np.mean([valor_condicao(f, "tel_f1", t, n) for f in final])
            tab[i, j + 1] = f1_v5 - f1_v3

    limite = max(abs(tab).max(), 0.01)
    fig, ax = plt.subplots(figsize=(LARGURA, LARGURA * 0.48))
    ax.imshow(tab, cmap=DIVERGENTE, vmin=-limite, vmax=limite, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab[i, j]
            texto = virgula(v, 3)
            if v > 0:
                texto = "+" + texto
            if abs(v) > 0.6 * limite:
                cor = "white"
            else:
                cor = COR_TEXTO
            ax.text(j, i, texto, ha="center", va="center", fontsize=8, color=cor)
    ax.set_xticks(range(len(NIVEIS) + 1), ["0 (limpo)"] + [virgula(n, 2) for n in NIVEIS])
    ax.set_yticks(range(len(ORDEM_TIPOS)), [NOMES_TIPO[t] for t in ORDEM_TIPOS])
    ax.set_xlabel("Intensidade da perturbação")
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    salvar(fig, saida / "fig15_diferenca_f1_v5_menos_v3.png")


def fig_val_teste(final_val, final_teste, saida):
    itens = [("tel_f1", "limpo", "F1\nlimpo"), ("tel_f1", None, "F1\nperturbado"),
             ("AP50_95", "limpo", "AP@0,5:0,95\nlimpo"), ("AP50_95", None, "AP@0,5:0,95\nperturbado")]
    x = np.arange(len(itens))
    largura_barra = 0.36
    fig, ax = plt.subplots(figsize=(LARGURA * 0.8, LARGURA * 0.42))
    for k, (linhas, nome, cor) in enumerate([(final_val, "Validação", "#2a78d6"),
                                             (final_teste, "Teste", "#eb6834")]):
        estat = [media_desvio(linhas, c, tipo=t) for c, t, _ in itens]
        valores = [m for m, _ in estat]
        barras = ax.bar(x + (k - 0.5) * (largura_barra + 0.02), valores, largura_barra,
                        color=cor, label=nome, yerr=[s for _, s in estat],
                        error_kw=dict(ecolor=COR_TEXTO, lw=0.8, capsize=2))
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, virgula(v, 3), ha="center",
                    fontsize=6.5, color=COR_TEXTO_2)
    ax.set_xticks(x, [r for _, _, r in itens])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FORMATO_VIRGULA)
    grade_y(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    salvar(fig, saida / "fig16_validacao_x_teste.png")


def desenhar_contornos(img, rotulos):
    # contorno branco em cada telhado
    for i in range(1, int(rotulos.max()) + 1):
        contornos, _ = cv2.findContours((rotulos == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contornos, -1, (255, 255, 255), 1)


def pintar_erros(img, previsto, real, rotulos):
    # verde = acerto, laranja = falso positivo, azul = telhado não detectado
    pintada = img.astype(np.float32).copy()
    for regiao, cor in [(previsto & real, COR_VP), (previsto & ~real, COR_FP),
                        (~previsto & real, COR_FN)]:
        pintada[regiao] = 0.4 * pintada[regiao] + 0.6 * hex_para_rgb(cor)
    pintada = pintada.astype(np.uint8)
    desenhar_contornos(pintada, rotulos)
    return pintada


def fig_mapas(args, saida):
    import torch
    from avaliar_unet import (carregar_modelos, carregar_pos, montar_condicoes, prever_modelos,
                              separar_telhados)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = {
        "v3": (args.final[0], carregar_modelos([args.final[0]], dispositivo)),
        "v5": (args.segundo, carregar_modelos([args.segundo], dispositivo)),
        "conj": (args.complementar, carregar_modelos(args.final, dispositivo)),
    }
    nomes_img, condicoes, mascaras, instancias = montar_condicoes(args.conjunto == "teste")
    k = escolher_imagem(args.imagem, instancias)
    linhas = ["limpo", "gaussian_1.00", "fog_1.00", "illumination_1.00"]
    linhas = [c for c in linhas if c in condicoes]
    real = mascaras[k] > 0
    titulos = {"v3": "U-Net v3", "v5": "U-Net v5", "conj": "Conjunto v3"}
    colunas = ["Real"] + [titulos[m] for m in modelos]

    fig, axs = plt.subplots(len(linhas), len(colunas),
                            figsize=(LARGURA, LARGURA / len(colunas) * len(linhas) + 0.5))
    for i, cond in enumerate(linhas):
        caminho = condicoes[cond][k]
        img = cv2.cvtColor(cv2.imread(str(caminho), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # 1a coluna: telhados anotados
        painel_real = img.copy()
        desenhar_contornos(painel_real, instancias[k])
        paineis = [painel_real]

        # demais colunas: previsão de cada modelo
        for chave, (nome_exp, mods) in modelos.items():
            pp, _ = carregar_pos(PASTA / "experimentos" / nome_exp, False)
            prob = prever_modelos(mods, [caminho], dispositivo, False)[0]
            rotulos, _ = separar_telhados(prob, pp)
            paineis.append(pintar_erros(img, rotulos > 0, real, rotulos))

        for j, painel in enumerate(paineis):
            ax = axs[i, j]
            ax.imshow(painel)
            ax.set_xticks([])
            ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_visible(False)
            if i == 0:
                ax.set_title(colunas[j], fontsize=8, color=COR_TEXTO)
            if j == 0:
                if cond == "limpo":
                    nome = "Limpo"
                else:
                    nome = NOMES_TIPO[cond.rsplit("_", 1)[0]] + "\n(1,00)"
                ax.set_ylabel(nome, fontsize=7.5, color=COR_TEXTO)

    fig.legend(handles=[Patch(color=COR_VP, label="Acerto"),
                        Patch(color=COR_FP, label="Falso positivo"),
                        Patch(color=COR_FN, label="Telhado não detectado")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    salvar(fig, saida / f"fig17_mapas_comparacao_img{k:02d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", nargs="+", default=["unet_v3", "unet_v3_s7", "unet_v3_s2026"])
    parser.add_argument("--segundo", default="unet_v5")
    parser.add_argument("--complementar", default="conjunto_v3")
    parser.add_argument("--conjunto", choices=["valid", "teste"], default="teste")
    parser.add_argument("--imagem", type=int, default=None,
                        help="índice da imagem dos mapas (padrão: a com mais telhados)")
    parser.add_argument("--sem-mapas", action="store_true")
    args = parser.parse_args()

    saida = PASTA / "experimentos" / "comparacao_final"
    saida.mkdir(parents=True, exist_ok=True)
    final = [carregar(n, args.conjunto) for n in args.final]
    v5 = carregar(args.segundo, args.conjunto)
    conj = carregar(args.complementar, args.conjunto)
    print(f"Figuras em {saida}")

    fig_barras(final, v5, conj, saida)
    fig_niveis(final, v5, conj, saida)
    fig_diferenca(final, v5, saida)
    if args.conjunto == "teste":
        fig_val_teste([carregar(n, "valid") for n in args.final], final, saida)
    if not args.sem_mapas:
        fig_mapas(args, saida)


if __name__ == "__main__":
    main()