"""
Treinamento da U-Net

Implementação e treinamento da U-Net para segmentação de telhados em imagens aéreas com perturbações. 
A arquitetura, a inicialização, a função de perda, o aumento de dados, a taxa de aprendizado e o critério de parada estão todos definidos neste arquivo.

Como rodar (dentro da pasta do projeto):
    python treinar_unet.py                                  (treina do zero)
    python treinar_unet.py --continuar                      (continua um treino interrompido)
    python treinar_unet.py --semente 7 --nome unet_v3_s7    (mesma configuração, outra semente)

"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import perturb_aerial_v3 as pa

# ---------------------------------------------------------------
# configuração do treino
# (as versões v1 a v6 mudam alguns destes valores, os valores são ajustados apra encontrar o melhor modelo/versão)
# ---------------------------------------------------------------
CONFIG = {
    "nome": "unet_v3",                    # versão final escolhida
    "semente": 42,

    # arquitetura: número de filtros em cada nível; o último é o gargalo
    #   v1 a v3: [32, 64, 128, 256, 512]       -> 4 descidas, gargalo de 32x32
    #   v4 e v5: [32, 64, 128, 256, 512, 512]  -> 5 descidas, gargalo de 16x16
    "filtros": [32, 64, 128, 256, 512],
    "dropout_gargalo": 0.3,
    "bn_momentum": 0.1,
    "bn_eps": 1e-5,
    "n_classes": 3,                       # 0 fundo, 1 interior, 2 borda

    # perda
    # peso da classe = (1 / frequência) ^ expoente, dividido pelo peso do fundo (fundo = 1)
    #   v1: 0.5   -> fundo 1, interior 2,06, borda 4,58
    #   v2 em diante: 0.25 (com 0.5 a borda prevista ficava grossa demais)
    "expoente_pesos": 0.25,
    # mapa de separação: peso maior nos pixels entre
    # d1 e d2 = distância até o telhado mais próximo e até o segundo mais próximo
    "mapa_separacao": True,               # v1: False
    "sep_w0": 5.0,                        # v2 a v4: 5.0 / v5: 10.0
    "sep_sigma": 5.0,                     # em pixels
    "peso_dice": 1.0,                     # perda = CE ponderada + peso_dice * Dice
    "dice_suavizacao": 1.0,
    "classes_dice": [1, 2],               # interior e borda

    # aumento de dados
    # escala (v6): amplia ou reduz a imagem; se ampliada, recorta 512x512,
    # se reduzida, completa até 512x512 por reflexão. v6: 0.5 (sem ganho) / demais: 0.0
    "p_escala": 0.0,
    "escala_min": 0.75,
    "escala_max": 1.25,
    "p_rotacao": 1.0,                     # sorteia 0, 90, 180 ou 270 graus
    "p_espelho_h": 0.5,
    "p_espelho_v": 0.5,
    "p_cor": 0.5,
    "variacao_brilho": 0.10,              # fator entre 1 - v e 1 + v
    "variacao_contraste": 0.10,
    "p_perturbacao": 0.7,                 # v1 e v2: 0.5
    "tipos_perturbacao": ["gaussian", "sensor", "fog", "illumination", "occlusion"],
    "intensidade_min": 0.2,               # v1 e v2: 0.0
    "intensidade_max": 1.0,
    "limiar_branco": 245,                 # o fundo branco sem dados não é perturbado

    # otimização
    "epocas": 150,
    "lote": 8,
    "lr_max": 1e-3,
    "lr_min": 1e-5,                       # cai em cosseno de lr_max até lr_min
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 1e-4,
    "paciencia": 30,                      # se tiverem 30 épocas sem uma melhora, para
    "precisao_mista": True,

    # execução
    "workers": 4,
    "lote_validacao": 16,
}

PASTA = Path(__file__).resolve().parent
DADOS = PASTA / "dataset_unet"
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------
# dados
# ---------------------------------------------------------------

def listar_pares(divisao):
    # lista os pares (imagem, máscara) de uma divisão (train, valid ou test)
    imagens = sorted(p for p in (DADOS / divisao / "imagens").iterdir()
                     if p.suffix.lower() in EXTENSOES)
    pares = [(p, DADOS / divisao / "mascaras" / (p.stem + ".png")) for p in imagens]
    for _, mascara in pares:
        if not mascara.exists():
            raise SystemExit(f"Máscara não encontrada: {mascara}")
    return pares


def ler_mascara(caminho):
    return cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED).astype(np.int64)


def calcular_normalizacao(pares):
    soma = np.zeros(3)
    soma_quadrados = np.zeros(3)
    n = 0
    for img, _ in pares:
        x = cv2.imread(str(img), cv2.IMREAD_COLOR).reshape(-1, 3).astype(np.float64) / 255.0
        soma += x.sum(0)
        soma_quadrados += (x ** 2).sum(0)
        n += x.shape[0]
    media = soma / n
    desvio = np.sqrt(soma_quadrados / n - media ** 2)
    return media.tolist(), desvio.tolist()


def calcular_pesos_classes(pares, n_classes, expoente):
    # peso = (1 / frequência) ^ expoente, dividido pelo peso do fundo
    contagem = np.zeros(n_classes, np.int64)
    for _, mascara in pares:
        contagem += np.bincount(ler_mascara(mascara).ravel(), minlength=n_classes)[:n_classes]
    freq = contagem / contagem.sum()
    pesos = (1.0 / freq) ** expoente
    pesos = pesos / pesos[0]
    return freq.tolist(), pesos.tolist()


def mapa_separacao(instancias, sigma):
    #distância ao 1o e ao 2o telhado mais próximo
    ids = [i for i in np.unique(instancias) if i != 0]
    if len(ids) < 2:
        return np.zeros(instancias.shape, np.float32)
    # distância de cada pixel até cada telhado
    distancias = np.stack([cv2.distanceTransform((instancias != i).astype(np.uint8), cv2.DIST_L2, 5)
                           for i in ids])
    duas_menores = np.partition(distancias, 1, axis=0)[:2]
    d1, d2 = duas_menores[0], duas_menores[1]
    return np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def preparar_mapas(pares, cfg, pasta):
    # calcula o mapa de separação de cada imagem de treino e salva em .npy (só na primeira vez; depois reaproveita os arquivos)
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for k, (img, _) in enumerate(pares):
        destino = pasta / (img.stem + ".npy")
        if not destino.exists():
            inst = cv2.imread(str(DADOS / "train" / "instancias" / (img.stem + ".png")),
                              cv2.IMREAD_UNCHANGED)
            np.save(destino, mapa_separacao(inst, cfg["sep_sigma"]).astype(np.float16))
            if k < 3:
                # salva 3 exemplos para conferir o mapa visualmente
                mapa = np.load(destino).astype(np.float32)
                foto = cv2.imread(str(img), cv2.IMREAD_COLOR)
                calor = cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(str(pasta.parent / f"exemplo_mapa_separacao_{k}.png"),
                            np.hstack([foto, cv2.addWeighted(foto, 0.4, calor, 0.6, 0)]))
        caminhos.append(destino)
    return caminhos


def escalar(img, mascara, sep, cfg):
    # muda a escala da imagem, da máscara e do mapa de separação, mantendo 512x512
    h, w = mascara.shape
    fator = random.uniform(cfg["escala_min"], cfg["escala_max"])
    nova_h = max(8, int(round(h * fator)))
    nova_w = max(8, int(round(w * fator)))
    img = cv2.resize(img, (nova_w, nova_h), interpolation=cv2.INTER_LINEAR)
    mascara = cv2.resize(mascara, (nova_w, nova_h), interpolation=cv2.INTER_NEAREST)  # não mistura as classes
    sep = cv2.resize(sep, (nova_w, nova_h), interpolation=cv2.INTER_LINEAR)

    if nova_h >= h and nova_w >= w:
        # ampliada: recorte aleatório de 512x512
        y0 = random.randint(0, nova_h - h)
        x0 = random.randint(0, nova_w - w)
        return (img[y0:y0 + h, x0:x0 + w], mascara[y0:y0 + h, x0:x0 + w],
                sep[y0:y0 + h, x0:x0 + w])

    # reduzida: completa até 512x512 refletindo a imagem
    topo = random.randint(0, h - nova_h)
    esquerda = random.randint(0, w - nova_w)
    borda = (topo, h - nova_h - topo, esquerda, w - nova_w - esquerda)
    img = cv2.copyMakeBorder(img, *borda, cv2.BORDER_REFLECT_101)
    mascara = cv2.copyMakeBorder(mascara, *borda, cv2.BORDER_REFLECT_101)
    sep = cv2.copyMakeBorder(sep, *borda, cv2.BORDER_REFLECT_101)
    return img, mascara, sep


class DadosTreino(torch.utils.data.Dataset):
    # le cada imagem de treino e aplica o aumento de dados

    def __init__(self, pares, cfg, mapas=None):
        self.pares = pares
        self.cfg = cfg
        self.mapas = mapas

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, i):
        cfg = self.cfg
        img = cv2.imread(str(self.pares[i][0]), cv2.IMREAD_COLOR)
        mascara = cv2.imread(str(self.pares[i][1]), cv2.IMREAD_UNCHANGED)
        if self.mapas:
            sep = np.load(self.mapas[i]).astype(np.float32)
        else:
            sep = np.zeros(mascara.shape, np.float32)

        # 1) escala (antes da perturbação, para o ruído ficar no tamanho real do pixel)
        if random.random() < cfg.get("p_escala", 0.0):
            img, mascara, sep = escalar(img, mascara, sep, cfg)

        # 2) perturbação, com tipo e intensidade sorteados
        if random.random() < cfg["p_perturbacao"]:
            tipo = random.choice(cfg["tipos_perturbacao"])
            intensidade = random.uniform(cfg["intensidade_min"], cfg["intensidade_max"])
            valida = pa.mascara_valida(img, cfg["limiar_branco"])
            perturbada = pa.PERTURBACOES[tipo](img, intensidade)
            img = np.where(valida[..., None], perturbada, img)

        # 3) rotação e espelhamento (iguais na imagem, na máscara e no mapa de separação)
        if random.random() < cfg["p_rotacao"]:
            k = random.randint(0, 3)
            img, mascara, sep = np.rot90(img, k), np.rot90(mascara, k), np.rot90(sep, k)
        if random.random() < cfg["p_espelho_h"]:
            img, mascara, sep = img[:, ::-1], mascara[:, ::-1], sep[:, ::-1]
        if random.random() < cfg["p_espelho_v"]:
            img, mascara, sep = img[::-1], mascara[::-1], sep[::-1]

        # 4) brilho e contraste
        x = np.ascontiguousarray(img).astype(np.float32)
        if random.random() < cfg["p_cor"]:
            brilho = random.uniform(1 - cfg["variacao_brilho"], 1 + cfg["variacao_brilho"])
            contraste = random.uniform(1 - cfg["variacao_contraste"], 1 + cfg["variacao_contraste"])
            media = x.mean()
            x = np.clip(((x - media) * contraste + media) * brilho, 0, 255)

        x = torch.from_numpy(x.transpose(2, 0, 1) / 255.0).float()
        y = torch.from_numpy(np.ascontiguousarray(mascara).astype(np.int64))
        s = torch.from_numpy(np.ascontiguousarray(sep))
        return x, y, s


def iniciar_worker(_):
    # cada worker do dataloader recebe uma semente própria
    semente = torch.initial_seed() % (2 ** 32)
    random.seed(semente)
    np.random.seed(semente)


def carregar_validacao():
    # carrega na memória a validação limpa e as 20 versões perturbadas
    pares = listar_pares("valid")
    mascaras = np.stack([ler_mascara(m) for _, m in pares]).astype(np.uint8)
    conjuntos = {"limpo": np.stack([cv2.imread(str(p), cv2.IMREAD_COLOR) for p, _ in pares])}

    pasta_pert = DADOS / "valid_perturbada"
    if not pasta_pert.exists():
        raise SystemExit("Pasta dataset_unet\\valid_perturbada não encontrada. "
                         "Rodar antes: python gerar_validacao_perturbada.py")
    for sub in sorted(pasta_pert.iterdir()):
        imagens = []
        for p, _ in pares:
            arq = sub / (p.stem + ".png")
            if not arq.exists():
                raise SystemExit(f"Arquivo faltando: {arq}")
            imagens.append(cv2.imread(str(arq), cv2.IMREAD_COLOR))
        conjuntos[sub.name] = np.stack(imagens)
    return conjuntos, mascaras


# ---------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------

class BlocoDuplo(nn.Module):
    # (convolução 3x3 -> BatchNorm -> ReLU) duas vezes

    def __init__(self, entrada, saida, cfg):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Conv2d(entrada, saida, 3, padding=1, bias=False),
            nn.BatchNorm2d(saida, eps=cfg["bn_eps"], momentum=cfg["bn_momentum"]),
            nn.ReLU(inplace=True),
            nn.Conv2d(saida, saida, 3, padding=1, bias=False),
            nn.BatchNorm2d(saida, eps=cfg["bn_eps"], momentum=cfg["bn_momentum"]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.camadas(x)


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        filtros = cfg["filtros"]

        # caminho de descida
        self.descida = nn.ModuleList()
        canais = 3
        for n in filtros[:-1]:
            self.descida.append(BlocoDuplo(canais, n, cfg))
            canais = n
        self.pool = nn.MaxPool2d(2)

        # gargalo
        self.gargalo = BlocoDuplo(filtros[-2], filtros[-1], cfg)
        self.dropout = nn.Dropout2d(cfg["dropout_gargalo"])

        # caminho de subida: convolução transposta + bloco duplo
        self.subida_conv = nn.ModuleList()
        self.subida_bloco = nn.ModuleList()
        for i in range(len(filtros) - 1, 0, -1):
            self.subida_conv.append(nn.ConvTranspose2d(filtros[i], filtros[i - 1], 2, stride=2))
            self.subida_bloco.append(BlocoDuplo(filtros[i - 1] * 2, filtros[i - 1], cfg))

        # convolução 1x1 final: uma saída por classe
        self.saida = nn.Conv2d(filtros[0], cfg["n_classes"], 1)

        self.inicializar()

    def inicializar(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # camada de saída com pesos pequenos: no começo, as 3 classes tem probabilidade parecida
        nn.init.normal_(self.saida.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.saida.bias)

    def forward(self, x):
        atalhos = []
        for bloco in self.descida:
            x = bloco(x)
            atalhos.append(x)   # guardado para a conexão de atalho
            x = self.pool(x)
        x = self.dropout(self.gargalo(x))
        for conv, bloco, atalho in zip(self.subida_conv, self.subida_bloco, reversed(atalhos)):
            x = conv(x)
            x = bloco(torch.cat([atalho, x], dim=1))
        return self.saida(x)


# ---------------------------------------------------------------
# função de perda e taxa de aprendizado
# ---------------------------------------------------------------

def perda(logits, alvo, pesos, sep, cfg):
    # entropia cruzada ponderada
    logits = logits.float()
    log_p = F.log_softmax(logits, dim=1)                          # (N, C, H, W)
    log_p_alvo = log_p.gather(1, alvo.unsqueeze(1)).squeeze(1)    # (N, H, W)
    peso_pixel = pesos[alvo]
    if cfg["mapa_separacao"]:
        peso_pixel = peso_pixel + cfg["sep_w0"] * sep
    ce = -(peso_pixel * log_p_alvo).sum() / peso_pixel.sum()

    p = log_p.exp()
    suav = cfg["dice_suavizacao"]
    dices = []
    for c in cfg["classes_dice"]:
        pc = p[:, c]
        gc = (alvo == c).float()
        dices.append((2 * (pc * gc).sum() + suav) / (pc.sum() + gc.sum() + suav))
    dice = 1 - torch.stack(dices).mean()
    return ce + cfg["peso_dice"] * dice, ce.item(), dice.item()


def taxa_aprendizado(epoca, cfg):
    t = epoca / max(1, cfg["epocas"] - 1)
    return cfg["lr_min"] + 0.5 * (cfg["lr_max"] - cfg["lr_min"]) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------
# validação: IoU de telhado por pixel (interior + borda contra fundo)
# ---------------------------------------------------------------

@torch.no_grad()
def iou_telhado(modelo, imagens, mascaras, media, desvio, dispositivo, cfg):
    modelo.eval()
    inter = 0
    uniao = 0
    lote = cfg["lote_validacao"]
    for i in range(0, len(imagens), lote):
        x = torch.from_numpy(imagens[i:i + lote]).to(dispositivo)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - media) / desvio
        with torch.autocast("cuda", enabled=cfg["precisao_mista"] and dispositivo.type == "cuda"):
            previsto = modelo(x).argmax(1)
        real = torch.from_numpy(mascaras[i:i + lote]).to(dispositivo)
        tel_previsto = previsto > 0
        tel_real = real > 0
        inter += (tel_previsto & tel_real).sum().item()
        uniao += (tel_previsto | tel_real).sum().item()
    return inter / uniao if uniao else 1.0


# ---------------------------------------------------------------
# treino
# ---------------------------------------------------------------

def salvar_seguro(objeto, caminho):
    # salva primeiro num arquivo temporário e só depois substitui o definitivo
    # tava caindo a luz, por isso inseri essa parte
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    torch.save(objeto, temporario)
    os.replace(temporario, caminho)


def salvar_historico(historico, caminho):
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historico[0].keys()), delimiter=";")
        escritor.writeheader()
        for h in historico:
            linha = {}
            for k, v in h.items():
                if isinstance(v, float):
                    linha[k] = f"{v:.6f}".replace(".", ",")
                else:
                    linha[k] = v
            escritor.writerow(linha)


def salvar_curvas(historico, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = [h["epoca"] for h in historico]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epocas, [h["perda"] for h in historico], label="perda total")
    ax[0].plot(epocas, [h["ce"] for h in historico], label="entropia cruzada")
    ax[0].plot(epocas, [h["dice"] for h in historico], label="Dice")
    ax[0].set_xlabel("época")
    ax[0].set_title("Perda no treino")
    ax[0].legend()
    ax[1].plot(epocas, [h["iou_limpo"] for h in historico], label="validação limpa")
    ax[1].plot(epocas, [h["iou_perturbado"] for h in historico], label="validação perturbada (média)")
    ax[1].plot(epocas, [h["criterio"] for h in historico], label="critério (média das duas)")
    ax[1].set_xlabel("época")
    ax[1].set_title("IoU de telhado")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuar", action="store_true")
    parser.add_argument("--semente", type=int, default=None,
                        help="treina a mesma configuração com outra semente")
    parser.add_argument("--nome", default=None, help="nome da pasta do experimento")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.semente is not None:
        cfg["semente"] = args.semente
    if args.nome:
        cfg["nome"] = args.nome
    saida = PASTA / "experimentos" / cfg["nome"]

    if saida.exists() and not args.continuar:
        raise SystemExit(f"A pasta {saida} já existe. Mudar o 'nome' no CONFIG, apagar a pasta "
                         "ou usar --continuar.")
    saida.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["semente"])
    np.random.seed(cfg["semente"])
    torch.manual_seed(cfg["semente"])

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type == "cuda":
        print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
    else:
        print("Dispositivo: CPU")

    # normalização e pesos das classes (calculados só com o treino)
    pares_treino = listar_pares("train")
    if args.continuar:
        norm = json.loads((saida / "normalizacao.json").read_text(encoding="utf-8"))
        pc = json.loads((saida / "pesos_classes.json").read_text(encoding="utf-8"))
    else:
        lista_media, lista_desvio = calcular_normalizacao(pares_treino)
        freq, lista_pesos = calcular_pesos_classes(pares_treino, cfg["n_classes"],
                                                   cfg["expoente_pesos"])
        norm = {"ordem_canais": "BGR", "media": lista_media, "desvio": lista_desvio}
        pc = {"frequencia": freq, "pesos": lista_pesos}
        (saida / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (saida / "normalizacao.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
        (saida / "pesos_classes.json").write_text(json.dumps(pc, indent=2), encoding="utf-8")
    print(f"Normalização (BGR): média {np.round(norm['media'], 4)}, desvio {np.round(norm['desvio'], 4)}")
    print(f"Frequência das classes no treino: {np.round(np.array(pc['frequencia']) * 100, 2)} %")
    print(f"Pesos da perda (fundo, interior, borda): {np.round(pc['pesos'], 3)}")

    media = torch.tensor(norm["media"], device=dispositivo).view(1, 3, 1, 1)
    desvio = torch.tensor(norm["desvio"], device=dispositivo).view(1, 3, 1, 1)
    pesos = torch.tensor(pc["pesos"], device=dispositivo)

    mapas = None
    if cfg["mapa_separacao"]:
        print("Calculando os mapas de separação do treino (só na primeira vez)...")
        mapas = preparar_mapas(pares_treino, cfg, saida / "mapas_separacao")

    gerador = torch.Generator()
    gerador.manual_seed(cfg["semente"])
    carregador = torch.utils.data.DataLoader(
        DadosTreino(pares_treino, cfg, mapas), batch_size=cfg["lote"], shuffle=True, drop_last=True,
        num_workers=cfg["workers"], worker_init_fn=iniciar_worker, generator=gerador,
        pin_memory=dispositivo.type == "cuda", persistent_workers=False)

    print("Carregando validação (limpa + perturbada)...")
    val_conjuntos, val_mascaras = carregar_validacao()
    print(f"  {len(val_conjuntos)} conjuntos x {len(val_mascaras)} imagens")

    # modelo e otimizador
    modelo = UNet(cfg).to(dispositivo)
    n_param = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"U-Net: {n_param:,} parâmetros treináveis".replace(",", "."))

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=cfg["lr_max"], betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
    usar_amp = cfg["precisao_mista"] and dispositivo.type == "cuda"
    escala = torch.amp.GradScaler("cuda", enabled=usar_amp)

    inicio = 0
    melhor = -1.0
    sem_melhora = 0
    historico = []
    if args.continuar:
        checkpoint = torch.load(saida / "ultimo.pt", map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(checkpoint["modelo"])
        otimizador.load_state_dict(checkpoint["otimizador"])
        escala.load_state_dict(checkpoint["escala"])
        inicio = checkpoint["epoca"] + 1
        melhor = checkpoint["melhor"]
        sem_melhora = checkpoint["sem_melhora"]
        historico = checkpoint["historico"]
        gerador.manual_seed(cfg["semente"] + inicio)
        print(f"Retomando da época {inicio + 1}")

    for epoca in range(inicio, cfg["epocas"]):
        t0 = time.time()
        lr = taxa_aprendizado(epoca, cfg)
        for grupo in otimizador.param_groups:
            grupo["lr"] = lr

        # uma passada por todo o treino
        modelo.train()
        soma = {"perda": 0.0, "ce": 0.0, "dice": 0.0}
        n_lotes = 0
        for x, y, s in carregador:
            x = x.to(dispositivo, non_blocking=True)
            y = y.to(dispositivo, non_blocking=True)
            s = s.to(dispositivo, non_blocking=True)
            x = (x - media) / desvio
            otimizador.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=usar_amp):
                logits = modelo(x)
            loss, ce, dc = perda(logits, y, pesos, s, cfg)
            escala.scale(loss).backward()
            escala.step(otimizador)
            escala.update()
            soma["perda"] += loss.item()
            soma["ce"] += ce
            soma["dice"] += dc
            n_lotes += 1

        # validação: critério = média entre o IoU limpo e o IoU médio das 20 perturbações
        ious = {}
        for nome, imagens in val_conjuntos.items():
            ious[nome] = iou_telhado(modelo, imagens, val_mascaras, media, desvio, dispositivo, cfg)
        iou_limpo = ious["limpo"]
        iou_pert = float(np.mean([v for k, v in ious.items() if k != "limpo"]))
        criterio = (iou_limpo + iou_pert) / 2

        registro = {"epoca": epoca + 1, "lr": lr,
                    "perda": soma["perda"] / n_lotes, "ce": soma["ce"] / n_lotes,
                    "dice": soma["dice"] / n_lotes,
                    "iou_limpo": iou_limpo, "iou_perturbado": iou_pert, "criterio": criterio,
                    "segundos": time.time() - t0}
        historico.append(registro)

        # guarda o modelo quando o critério melhora
        marca = ""
        if criterio > melhor:
            melhor = criterio
            sem_melhora = 0
            marca = "  <- melhor"
            salvar_seguro({"modelo": modelo.state_dict(), "epoca": epoca + 1, "criterio": criterio,
                           "config": cfg, "normalizacao": norm}, saida / "melhor.pt")
        else:
            sem_melhora += 1

        print(f"Época {epoca + 1:3d}/{cfg['epocas']} / lr {lr:.2e} / perda {registro['perda']:.4f} "
              f"/ IoU limpo {iou_limpo:.4f} / IoU pert. {iou_pert:.4f} / critério {criterio:.4f} "
              f"/ {registro['segundos']:.0f}s{marca}")

        # estado completo da época, para poder continuar se o treino parar
        salvar_seguro({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(),
                       "escala": escala.state_dict(), "epoca": epoca, "melhor": melhor,
                       "sem_melhora": sem_melhora, "historico": historico}, saida / "ultimo.pt")
        salvar_historico(historico, saida / "historico.csv")

        if sem_melhora >= cfg["paciencia"]:
            print(f"\nParada antecipada: {cfg['paciencia']} épocas sem melhorar.")
            break

    salvar_curvas(historico, saida / "curvas.png")
    melhor_ep = max(historico, key=lambda h: h["criterio"])
    print("\n" + "=" * 70)
    print(f"Melhor época: {melhor_ep['epoca']}")
    print(f"  IoU de telhado na validação limpa:       {melhor_ep['iou_limpo']:.4f}")
    print(f"  IoU de telhado na validação perturbada:  {melhor_ep['iou_perturbado']:.4f}")
    print(f"  Critério (média):                        {melhor_ep['criterio']:.4f}")
    print("=" * 70)
    print(f"Resultados em {saida}")


if __name__ == "__main__":   # precisa do windows por causa dos workers do dataloader
    main()