"""
Calibração do pós-processamento

Testa as combinações de parâmetros do pós-processamento no conjunto de validação e escolhe a que tiver o maior critério:
    critério = (F1 por telhado no limpo + F1 médio nas 20 condições perturbadas) / 2

Não treina a rede e não usa o conjunto de teste.

Parâmetros testados (4 x 2 x 6 x 6 = 288 combinações):
    limiar_semente   : None (interior pela maior probabilidade), 0.5, 0.6, 0.7
    crescimento      : 6, 12
    area_minima      : 30, 60, 100, 150, 200, 300 pixels
    confianca_minima : 0.0, 0.6, 0.7, 0.8, 0.85, 0.9

Como rodar (dentro da pasta do projeto):
    python calibrar_posprocessamento.py --experimento unet_v3
    python calibrar_posprocessamento.py --experimento unet_v3 --tta

"""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import torch

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from avaliar_unet import (POS_PADRAO, arquivo_pos, carregar_modelos, casar,
                          crescer_sementes, descrever_modelos, matriz_iou, montar_condicoes,
                          prever_modelos)

# valores testados de cada parâmetro
LIMIARES_SEMENTE = [None, 0.5, 0.6, 0.7]
CRESCIMENTOS = [6, 12]
AREAS = [30, 60, 100, 150, 200, 300]
CONFIANCAS = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9]
LIMIAR_IOU = 0.5   # IoU mínimo para contar como acerto


def virgula(v):
    return str(v).replace(".", ",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimento", default="unet_v1")
    parser.add_argument("--tta", action="store_true",
                        help="calibra para a previsão com 8 versões de cada imagem")
    parser.add_argument("--modelos", nargs="+", default=None,
                        help="lista de experimentos para o conjunto de modelos")
    args = parser.parse_args()

    nomes_modelos = args.modelos or [args.experimento]
    exp = PASTA / "experimentos" / args.experimento
    exp.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = carregar_modelos(nomes_modelos, dispositivo)
    print(f"Modelos: {descrever_modelos(nomes_modelos, modelos)}")

    nomes, condicoes, mascaras, instancias = montar_condicoes(teste=False)

    # o crescimento depende só da semente e do número de passos; a área e a confiança são só filtros depois. Então o crescimento é feito uma vez por grupo
    grupos = list(itertools.product(LIMIARES_SEMENTE, CRESCIMENTOS))
    filtros = list(itertools.product(AREAS, CONFIANCAS))
    combinacoes = []
    for s, c in grupos:
        for a, k in filtros:
            combinacoes.append((s, c, a, k))
    print(f"Experimento {args.experimento} / {len(combinacoes)} combinações / "
          f"{len(condicoes)} condições da validação / TTA: {'sim' if args.tta else 'não'}\n")

    # para cada condição: acertos e previstos de cada combinação, e total de telhados reais
    vp = {n: np.zeros(len(combinacoes), np.int64) for n in condicoes}
    previstos = {n: np.zeros(len(combinacoes), np.int64) for n in condicoes}
    reais = {n: 0 for n in condicoes}

    for nome, caminhos in condicoes.items():
        probs = prever_modelos(modelos, caminhos, dispositivo, args.tta)
        for prob, inst in zip(probs, instancias):
            reais[nome] += int(inst.max())
            p_telhado = prob[1].astype(np.float32) + prob[2].astype(np.float32)
            idx = 0
            for semente, cresc in grupos:
                n, rotulos = crescer_sementes(prob, semente, cresc)
                areas = np.bincount(rotulos.ravel(), minlength=n)[1:]
                somas = np.bincount(rotulos.ravel(), weights=p_telhado.ravel(), minlength=n)[1:]
                conf = somas / np.maximum(areas, 1)
                iou = matriz_iou(rotulos, inst)   # o IoU não muda com os filtros
                for area_min, conf_min in filtros:
                    manter = (areas >= area_min) & (conf >= conf_min)
                    acertos = casar(iou[manter], conf[manter], LIMIAR_IOU)
                    vp[nome][idx] += int(acertos.sum())
                    previstos[nome][idx] += int(manter.sum())
                    idx += 1
        print(f"  {nome} concluído")

    def f1(n):
        fp = previstos[n] - vp[n]
        fn = reais[n] - vp[n]
        return 2 * vp[n] / np.maximum(2 * vp[n] + fp + fn, 1)

    f1_limpo = f1("limpo")
    perturbadas = [n for n in condicoes if n != "limpo"]
    f1_pert = np.mean([f1(n) for n in perturbadas], axis=0)
    criterio = (f1_limpo + f1_pert) / 2
    prec_limpo = vp["limpo"] / np.maximum(previstos["limpo"], 1)
    rec_limpo = vp["limpo"] / max(reais["limpo"], 1)

    # tabela com todas as combinações
    with open(exp / "calibracao_posprocessamento.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["limiar_semente", "crescimento", "area_minima", "confianca_minima",
                    "precisao_limpo", "recall_limpo", "f1_limpo", "f1_perturbado", "criterio"])
        for i, (s, c, a, k) in enumerate(combinacoes):
            semente = "maior_prob" if s is None else virgula(s)
            valores = [prec_limpo[i], rec_limpo[i], f1_limpo[i], f1_pert[i], criterio[i]]
            w.writerow([semente, c, a, virgula(k)] + [f"{v:.4f}".replace(".", ",") for v in valores])

    def linha(i, marca=""):
        s, c, a, k = combinacoes[i]
        semente = "maior_prob" if s is None else s
        return (f"{semente:>12} {c:>6} {a:>6} {k:>6} / "
                f"P {prec_limpo[i]:.3f} R {rec_limpo[i]:.3f} F1 {f1_limpo[i]:.3f} / "
                f"F1 pert. {f1_pert[i]:.3f} / critério {criterio[i]:.4f}{marca}")

    padrao = combinacoes.index((POS_PADRAO["limiar_semente"], POS_PADRAO["crescimento"],
                                POS_PADRAO["area_minima"], POS_PADRAO["confianca_minima"]))
    ordem = np.argsort(-criterio)
    melhor = ordem[0]

    print("\n" + "=" * 100)
    print(f"{'semente':>12} {'cresc':>6} {'área':>6} {'conf':>6} / métricas por telhado (limpo) / ...")
    print("Padrão da v1:")
    print(linha(padrao))
    print("10 melhores:")
    for i in ordem[:10]:
        if i == melhor:
            print(linha(i, "  <- escolhida"))
        else:
            print(linha(i))
    print("=" * 100)

    # salva a combinação escolhida
    s, c, a, k = combinacoes[melhor]
    escolha = {"limiar_semente": s, "crescimento": c, "area_minima": a, "confianca_minima": k,
               "criterio": float(criterio[melhor]), "calibrado_em": "validação",
               "tta": args.tta}
    destino = arquivo_pos(exp, args.tta)
    destino.write_text(json.dumps(escolha, indent=2), encoding="utf-8")
    print(f"\nEscolha salva em {destino}")

    comando = "python avaliar_unet.py"
    if args.experimento != "unet_v1":
        comando += f" --experimento {args.experimento}"
    if args.tta:
        comando += " --tta"
    if args.modelos:
        comando += " --modelos " + " ".join(args.modelos)
    print("Próximo passo: " + comando)


if __name__ == "__main__":
    main()