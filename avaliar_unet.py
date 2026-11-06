"""
Avaliação da U-Net

Faz a previsão das imagens, separa os telhados (pós-processamento) e calcula as métricas por pixel e por telhado, nas imagens limpas e em cada perturbação.

Como rodar (dentro da pasta do projeto):
    python avaliar_unet.py --experimento unet_v3                 (validação)
    python avaliar_unet.py --experimento unet_v3 --teste-final   (teste)
    python avaliar_unet.py --experimento unet_v3 --tta           (previsão com 8 versões da imagem)
    python avaliar_unet.py --experimento conjunto_v3 --modelos unet_v3 unet_v3_s7 unet_v3_s2026

Pós-processamento:
    1. classe do pixel = maior probabilidade (fundo, interior ou borda)
    2. sementes = regiões de interior (ou P(interior) >= limiar_semente)
    3. cada semente cresce sobre os pixels de telhado vizinhos
    4. descarta telhados com área menor que area_minima
    5. confiança = média de P(interior) + P(borda) no telhado; descarta telhados com confiança menor que confianca_minima
    Os parâmetros são lidos do posprocessamento.json do experimento (gerado pelo calibrar_posprocessamento.py).

Métricas:
    por pixel (telhado contra fundo): acurácia, precisão, recall, F1 e IoU
    por telhado (acerto quando IoU >= 0,5): precisão, recall, F1, AP50 e AP50:95

Resultados em experimentos/<nome>/avaliacao_<valid|teste>/
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

DADOS = PASTA / "dataset_unet"
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# valores usados quando o experimento não tem posprocessamento.json
POS_PADRAO = {
    "limiar_semente": None,   # None = usa o interior pela maior probabilidade
    "area_minima": 30,        # em pixels
    "confianca_minima": 0.0,
    "crescimento": 6,         # passos de dilatação (a borda tem 2 px)
}
LIMIAR_IOU = 0.5                                     # IoU mínimo para contar como acerto
LIMIARES_AP = np.round(np.arange(0.5, 0.96, 0.05), 2)  # 0.50, 0.55, ..., 0.95 (AP50:95)
LOTE = 16
N_EXEMPLOS = 3


# ---------------------------------------------------------------
# leitura dos dados
# ---------------------------------------------------------------

def procurar_imagem(pasta, nome):
    for ext in EXTENSOES:
        caminho = pasta / (nome + ext)
        if caminho.exists():
            return caminho
    raise SystemExit(f"Imagem {nome} não encontrada em {pasta}")


def montar_condicoes(teste):
    # monta a lista de imagens de cada condição (limpo + cada tipo/nível de perturbação)
    divisao = "test" if teste else "valid"
    imagens = sorted(p for p in (DADOS / divisao / "imagens").iterdir()
                     if p.suffix.lower() in EXTENSOES)
    nomes = [p.stem for p in imagens]
    condicoes = {"limpo": imagens}

    if teste:
        pasta_pert = DADOS / "teste_perturbado"
        if not pasta_pert.exists():
            raise SystemExit("Pasta dataset_unet\\teste_perturbado não encontrada. "
                             "Rodar antes: python gerar_validacao_perturbada.py --teste")
    else:
        pasta_pert = DADOS / "valid_perturbada"
    for sub in sorted(p for p in pasta_pert.iterdir() if p.is_dir()):
        condicoes[sub.name] = [procurar_imagem(sub, n) for n in nomes]

    mascaras = []
    instancias = []
    for n in nomes:
        mascaras.append(cv2.imread(str(DADOS / divisao / "mascaras" / (n + ".png")),
                                   cv2.IMREAD_UNCHANGED))
        instancias.append(cv2.imread(str(DADOS / divisao / "instancias" / (n + ".png")),
                                     cv2.IMREAD_UNCHANGED))
    return nomes, condicoes, mascaras, instancias


# ---------------------------------------------------------------
# previsão
# ---------------------------------------------------------------

@torch.no_grad()
def prever(modelo, caminhos, media, desvio, dispositivo, tta=False):
    # retorna as probabilidades (N, 3, H, W) em float16 para ocupar menos memória
    # com tta, cada imagem é prevista em 8 versões (4 rotações, com e sem espelho)
    # e a média das 8 é usada
    if tta:
        versoes = [(k, e) for k in range(4) for e in (False, True)]
    else:
        versoes = [(0, False)]

    resultado = []
    for i in range(0, len(caminhos), LOTE):
        lote = np.stack([cv2.imread(str(p), cv2.IMREAD_COLOR) for p in caminhos[i:i + LOTE]])
        x = torch.from_numpy(lote).to(dispositivo).permute(0, 3, 1, 2).float() / 255.0
        x = (x - media) / desvio

        soma = 0
        for k, espelho in versoes:
            xv = torch.rot90(x, k, dims=(2, 3))
            if espelho:
                xv = xv.flip(3)
            with torch.autocast("cuda", enabled=dispositivo.type == "cuda"):
                p = torch.softmax(modelo(xv).float(), dim=1)
            # desfaz a rotação/espelho para voltar à posição original
            if espelho:
                p = p.flip(3)
            soma = soma + torch.rot90(p, -k, dims=(2, 3))
        resultado.append((soma / len(versoes)).half().cpu().numpy())
    return np.concatenate(resultado)


def carregar_modelos(nomes, dispositivo):
    # carrega o melhor.pt de cada experimento
    modelos = []
    for nome in nomes:
        checkpoint = torch.load(PASTA / "experimentos" / nome / "melhor.pt",
                                map_location="cpu", weights_only=False)
        modelo = UNet(checkpoint["config"]).to(dispositivo)
        modelo.load_state_dict(checkpoint["modelo"])
        modelo.eval()
        norm = checkpoint["normalizacao"]
        media = torch.tensor(norm["media"], device=dispositivo).view(1, 3, 1, 1)
        desvio = torch.tensor(norm["desvio"], device=dispositivo).view(1, 3, 1, 1)
        modelos.append((modelo, media, desvio, checkpoint["epoca"]))
    return modelos


def prever_modelos(modelos, caminhos, dispositivo, tta=False):
    # com mais de um modelo, faz a média das probabilidades
    soma = None
    for modelo, media, desvio, _ in modelos:
        p = prever(modelo, caminhos, media, desvio, dispositivo, tta).astype(np.float32)
        if soma is None:
            soma = p
        else:
            soma = soma + p
    return (soma / len(modelos)).astype(np.float16)


def descrever_modelos(nomes, modelos):
    return ", ".join(f"{n} (época {m[3]})" for n, m in zip(nomes, modelos))


# ---------------------------------------------------------------
# pós-processamento
# ---------------------------------------------------------------

def crescer_sementes(prob, limiar_semente, crescimento):
    # as sementes de interior crescem sobre os pixels de telhado ainda sem rótulo
    classe = prob.argmax(0)
    telhado = classe > 0
    if limiar_semente is None:
        semente = classe == 1
    else:
        semente = (prob[1].astype(np.float32) >= limiar_semente) & telhado

    n, rotulos = cv2.connectedComponents(semente.astype(np.uint8), connectivity=4)
    rotulos = rotulos.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(crescimento):
        livres = telhado & (rotulos == 0)
        if not livres.any():
            break
        dilatado = cv2.dilate(rotulos, kernel)
        rotulos[livres] = dilatado[livres]
    return n, rotulos.astype(np.int32)


def filtrar_telhados(prob, n, rotulos, area_minima, confianca_minima):
    # remove os telhados pequenos ou com pouca confiança e renumera os que sobram
    p_telhado = prob[1].astype(np.float32) + prob[2].astype(np.float32)
    areas = np.bincount(rotulos.ravel(), minlength=n)
    somas = np.bincount(rotulos.ravel(), weights=p_telhado.ravel(), minlength=n)
    confianca = somas / np.maximum(areas, 1)

    manter = (areas >= area_minima) & (confianca >= confianca_minima)
    manter[0] = False  # 0 é o fundo
    novo_rotulo = np.zeros(n, np.int32)
    novo_rotulo[manter] = np.arange(1, manter.sum() + 1)
    return novo_rotulo[rotulos], confianca[manter]


def separar_telhados(prob, pp):
    # prob (3, H, W) -> mapa com cada telhado numerado e a confiança de cada um
    n, rotulos = crescer_sementes(prob, pp["limiar_semente"], pp["crescimento"])
    return filtrar_telhados(prob, n, rotulos, pp["area_minima"], pp["confianca_minima"])


def arquivo_pos(exp, tta):
    if tta:
        return exp / "posprocessamento_tta.json"
    return exp / "posprocessamento.json"


def carregar_pos(exp, tta=False):
    arq = arquivo_pos(exp, tta)
    if arq.exists():
        pp = json.loads(arq.read_text(encoding="utf-8"))
        return {k: pp[k] for k in POS_PADRAO}, f"calibrado ({arq.name})"
    return dict(POS_PADRAO), "padrão da v1"


# ---------------------------------------------------------------
# métricas por telhado
# ---------------------------------------------------------------

def matriz_iou(pred, real):
    # IoU entre cada telhado previsto e cada telhado real (n_pred x n_real)
    n_p = int(pred.max())
    n_r = int(real.max())
    if n_p == 0 or n_r == 0:
        return np.zeros((n_p, n_r))
    # conta os pixels de cada par (real, previsto) de uma vez só
    par = real.astype(np.int64) * (n_p + 1) + pred.astype(np.int64)
    inter = np.bincount(par.ravel(), minlength=(n_r + 1) * (n_p + 1)).reshape(n_r + 1, n_p + 1)
    area_real = inter.sum(1)
    area_pred = inter.sum(0)
    i = inter[1:, 1:].T.astype(np.float64)
    uniao = area_pred[1:, None] + area_real[None, 1:] - i
    return np.where(uniao > 0, i / uniao, 0)


def casar(iou, confiancas, limiar):
    # começa pelo telhado mais confiante; cada telhado real só
    # pode ser usado uma vez. retorna 1 (acerto) ou 0 para cada previsão
    acertos = np.zeros(len(confiancas), np.int8)
    if iou.size == 0:
        return acertos
    usados = np.zeros(iou.shape[1], bool)
    for k in np.argsort(-confiancas):
        candidatos = np.where(~usados, iou[k], -1)
        j = int(candidatos.argmax())
        if candidatos[j] >= limiar:
            usados[j] = True
            acertos[k] = 1
    return acertos


def calcular_ap(confiancas, acertos, n_real):
    # AP com interpolação de 101 pontos de recall
    if n_real == 0:
        return float("nan")
    if len(confiancas) == 0:
        return 0.0
    ordem = np.argsort(-confiancas, kind="stable")
    vp = np.cumsum(acertos[ordem])
    fp = np.cumsum(1 - acertos[ordem])
    recall = vp / n_real
    precisao = vp / (vp + fp)
    # precisão interpolada
    for i in range(len(precisao) - 2, -1, -1):
        precisao[i] = max(precisao[i], precisao[i + 1])
    pontos = []
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        if idx < len(precisao):
            pontos.append(precisao[idx])
        else:
            pontos.append(0.0)
    return float(np.mean(pontos))


# ---------------------------------------------------------------
# avaliação de uma condição (limpo ou uma perturbação)
# ---------------------------------------------------------------

def avaliar_condicao(probs, mascaras, instancias, pp):
    matriz = np.zeros((3, 3), np.int64)   # linha = real, coluna = previsto
    todas_conf = []
    n_real = 0
    acertos_limiar = {t: [] for t in LIMIARES_AP}
    previsoes = []

    for prob, mascara, inst in zip(probs, mascaras, instancias):
        classe = prob.argmax(0)
        matriz += np.bincount((mascara.astype(np.int64) * 3 + classe).ravel(),
                              minlength=9).reshape(3, 3)
        rotulos, conf = separar_telhados(prob, pp)
        previsoes.append(rotulos)
        iou = matriz_iou(rotulos, inst)
        n_real += int(inst.max())
        todas_conf.append(conf)
        for t in LIMIARES_AP:
            acertos_limiar[t].append(casar(iou, conf, t))

    if todas_conf:
        todas_conf = np.concatenate(todas_conf)
    else:
        todas_conf = np.zeros(0)
    acertos = {t: np.concatenate(v) for t, v in acertos_limiar.items()}

    # métricas por pixel: telhado (interior + borda) contra fundo
    vp = matriz[1:, 1:].sum()
    fn = matriz[1:, 0].sum()
    fp = matriz[0, 1:].sum()
    vn = matriz[0, 0]
    r = {
        "px_acuracia": (vp + vn) / matriz.sum(),
        "px_precisao": vp / (vp + fp) if vp + fp else 0.0,
        "px_recall": vp / (vp + fn) if vp + fn else 0.0,
        "px_f1": 2 * vp / (2 * vp + fp + fn) if vp + fp + fn else 1.0,
        "px_iou": vp / (vp + fp + fn) if vp + fp + fn else 1.0,
    }

    # métricas por telhado
    a = acertos[LIMIAR_IOU]
    vp_t = int(a.sum())
    fp_t = len(a) - vp_t
    fn_t = n_real - vp_t
    r["tel_reais"] = n_real
    r["tel_previstos"] = len(a)
    r["tel_precisao"] = vp_t / len(a) if len(a) else 0.0
    r["tel_recall"] = vp_t / n_real if n_real else 0.0
    r["tel_f1"] = 2 * vp_t / (2 * vp_t + fp_t + fn_t) if (vp_t + fp_t + fn_t) else 1.0
    r["AP50"] = calcular_ap(todas_conf, acertos[0.5], n_real)
    r["AP50_95"] = float(np.mean([calcular_ap(todas_conf, acertos[t], n_real)
                                  for t in LIMIARES_AP]))
    return r, matriz, previsoes


# ---------------------------------------------------------------
# imagens de exemplo
# ---------------------------------------------------------------

def gerar_cores(n):
    rng = np.random.default_rng(0)
    c = rng.integers(60, 255, (n + 1, 3), dtype=np.uint8)
    c[0] = 0
    return c


def pintar_telhados(img, rotulos):
    # pinta cada telhado de uma cor e desenha o contorno em branco
    cor = gerar_cores(int(rotulos.max()))[rotulos]
    saida = img.copy()
    m = rotulos > 0
    saida[m] = (0.45 * img[m] + 0.55 * cor[m]).astype(np.uint8)
    for i in range(1, int(rotulos.max()) + 1):
        contornos, _ = cv2.findContours((rotulos == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(saida, contornos, -1, (255, 255, 255), 1)
    return saida


def salvar_exemplo(caminho_img, inst_real, rot_prev, destino):
    # original / telhados reais / telhados previstos
    img = cv2.imread(str(caminho_img), cv2.IMREAD_COLOR)
    painel = np.hstack([img, pintar_telhados(img, inst_real.astype(np.int32)),
                        pintar_telhados(img, rot_prev)])
    cv2.imwrite(str(destino), painel)


def formatar(v):
    # número com vírgula para abrir certo no Excel
    if isinstance(v, float):
        return f"{v:.4f}".replace(".", ",")
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimento", default="unet_v1")
    parser.add_argument("--teste-final", action="store_true",
                        help="avalia no conjunto de teste")
    parser.add_argument("--tta", action="store_true",
                        help="previsão com 8 versões de cada imagem")
    parser.add_argument("--modelos", nargs="+", default=None,
                        help="lista de experimentos para o conjunto de modelos")
    args = parser.parse_args()

    nomes_modelos = args.modelos or [args.experimento]
    exp = PASTA / "experimentos" / args.experimento
    exp.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = carregar_modelos(nomes_modelos, dispositivo)
    desc = descrever_modelos(nomes_modelos, modelos)

    nome_conj = "teste" if args.teste_final else "valid"
    pp, origem_pp = carregar_pos(exp, args.tta)
    sufixo = ""
    if args.tta:
        sufixo += "_tta"
    if origem_pp.startswith("calibrado"):
        sufixo += "_calibrado"
    saida = exp / f"avaliacao_{nome_conj}{sufixo}"
    (saida / "exemplos").mkdir(parents=True, exist_ok=True)
    print(f"Experimento {args.experimento} | modelos: {desc} | conjunto: {nome_conj.upper()}"
          f" | TTA: {'sim (8 versões)' if args.tta else 'não'}")
    print(f"Pós-processamento ({origem_pp}): {pp}")

    nomes, condicoes, mascaras, instancias = montar_condicoes(args.teste_final)
    idx_exemplos = [0, len(nomes) // 2, len(nomes) - 1][:N_EXEMPLOS]

    resultados = {}
    matrizes = {}
    for nome, caminhos in condicoes.items():
        probs = prever_modelos(modelos, caminhos, dispositivo, args.tta)
        r, matriz, previsoes = avaliar_condicao(probs, mascaras, instancias, pp)
        resultados[nome] = r
        matrizes[nome] = matriz
        print(f"{nome:<18} IoU px {r['px_iou']:.4f} | F1 px {r['px_f1']:.4f} | "
              f"telhados P {r['tel_precisao']:.3f} R {r['tel_recall']:.3f} F1 {r['tel_f1']:.3f} | "
              f"AP50 {r['AP50']:.3f} | AP50:95 {r['AP50_95']:.3f}")
        # salva exemplos só da imagem limpa e da intensidade máxima
        if nome == "limpo" or nome.endswith("1.00"):
            for k in idx_exemplos:
                salvar_exemplo(caminhos[k], instancias[k], previsoes[k],
                               saida / "exemplos" / f"{nome}_{k:02d}.png")

    # tabela com todas as métricas
    chaves = list(next(iter(resultados.values())).keys())
    with open(saida / "metricas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["condicao", "tipo", "nivel"] + chaves)
        for nome, r in resultados.items():
            if nome == "limpo":
                tipo, nivel = "limpo", "0.00"
            else:
                partes = nome.rsplit("_", 1)
                tipo = partes[0]
                nivel = partes[1] if len(partes) > 1 else ""
            w.writerow([nome, tipo, nivel.replace(".", ",")] + [formatar(r[k]) for k in chaves])

    with open(saida / "matriz_confusao.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["condicao", "real", "prev_fundo", "prev_interior", "prev_borda"])
        for nome, m in matrizes.items():
            for i, c in enumerate(["fundo", "interior", "borda"]):
                w.writerow([nome, c] + list(m[i]))

    # resumo: limpo, média perturbada, média por tipo e por nível
    perturbadas = [k for k in resultados if k != "limpo"]
    tipos = sorted({k.rsplit("_", 1)[0] for k in perturbadas})
    niveis = sorted({k.rsplit("_", 1)[1] for k in perturbadas})
    principais = ["px_iou", "px_f1", "tel_precisao", "tel_recall", "tel_f1", "AP50", "AP50_95"]

    def linha(rotulo, valores):
        return f"{rotulo:<20}" + "".join(f"{v:>14.4f}" for v in valores)

    def media(conds, k):
        return np.mean([resultados[c][k] for c in conds])

    linhas = [f"Experimento: {args.experimento} | modelos: {desc} | conjunto: {nome_conj}"
              f" | TTA: {'sim' if args.tta else 'não'}",
              f"Pós-processamento ({origem_pp}): {pp}", "",
              f"{'condição':<20}" + "".join(f"{k:>14}" for k in principais)]
    linhas.append(linha("limpo", [resultados["limpo"][k] for k in principais]))
    linhas.append(linha("perturbado (média)", [media(perturbadas, k) for k in principais]))

    linhas += ["", "Média por TIPO de perturbação (4 níveis):"]
    for t in tipos:
        cs = [c for c in perturbadas if c.startswith(t + "_")]
        linhas.append(linha(t, [media(cs, k) for k in principais]))

    linhas += ["", "Média por NÍVEL (5 tipos):"]
    for n in niveis:
        cs = [c for c in perturbadas if c.endswith("_" + n)]
        linhas.append(linha(f"nível {n}", [media(cs, k) for k in principais]))

    m = matrizes["limpo"]
    linhas += ["", "Matriz de confusão por pixel (limpo) — linhas: real, colunas: previsto",
               f"{'':<10}{'fundo':>12}{'interior':>12}{'borda':>12}"]
    for i, c in enumerate(["fundo", "interior", "borda"]):
        linhas.append(f"{c:<10}" + "".join(f"{v:>12d}" for v in m[i]))
    recall_classe = np.diag(m) / m.sum(1)
    linhas.append("Recall por classe (limpo): " +
                  ", ".join(f"{c} {v:.3f}" for c, v in zip(["fundo", "interior", "borda"],
                                                          recall_classe)))

    texto = "\n".join(linhas)
    print("\n" + "=" * 120 + "\n" + texto + "\n" + "=" * 120)
    (saida / "resumo.txt").write_text(texto + "\n", encoding="utf-8")
    print(f"\nArquivos em {saida}")
    print("Exemplos: original | telhados reais | telhados previstos (cada cor = um telhado)")


if __name__ == "__main__":
    main()
