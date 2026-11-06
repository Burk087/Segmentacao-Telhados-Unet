"""
Gerador de perturbações

Aplica perturbações em imagens para simular problemas reais de captura:
    gaussian      ruído gaussiano com leve correlação espacial (granulação)
    sensor        ruído de sensor (Poisson + ruído de leitura + faixas + pixels defeituosos)
    fog           névoa/nuvens em partes da imagem (nunca cobre a imagem toda)
    illumination  variação de iluminação (exposição + gamma + gradiente de luz)
    occlusion     oclusões (nuvem, sombra ou perda de dados do sensor)

A intensidade vai de 0.0 (leve) a 1.0 (forte). As funções deste arquivo também são usadas pelo treinar_unet.py e pelo gerar_validacao_perturbada.py

"""

import random
import cv2
import numpy as np


# ---------------------------------------------------------------
# funções auxiliares
# ---------------------------------------------------------------

def carregar_imagem(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return img


def limitar_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def mascara_valida(img, limiar_branco=245):
    # True = pixel da imagem; False = fundo branco do recorte
    # serve para não aplicar perturbação nesse fundo
    branco = (img.min(axis=2) >= limiar_branco).astype(np.uint8)
    n, rotulos = cv2.connectedComponents(branco, connectivity=4)
    na_borda = np.unique(np.concatenate([rotulos[0, :], rotulos[-1, :],
                                         rotulos[:, 0], rotulos[:, -1]]))
    na_borda = na_borda[na_borda != 0]
    fundo = np.isin(rotulos, na_borda)
    valida = ~fundo
    if valida.mean() < 0.05:   # imagem quase toda branca: não tem o que proteger
        return np.ones(img.shape[:2], dtype=bool)
    return valida


def transicao_suave(inicio, fim, x):
    # vai de 0 a 1 de forma suave entre inicio e fim
    t = np.clip((x - inicio) / (fim - inicio + 1e-8), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def ruido_fractal(formato, celulas_base=3, oitavas=5, persistencia=0.5):
    # ruído fractal entre 0 e 1, usado para dar forma irregular às nuvens e sombras
    h, w = formato
    resultado = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    for o in range(oitavas):
        celulas = celulas_base * (2 ** o)
        if h >= w:
            gh = celulas
            gw = max(2, int(round(celulas * w / h)))
        else:
            gw = celulas
            gh = max(2, int(round(celulas * h / w)))
        grade = np.random.rand(max(2, gh), max(2, gw)).astype(np.float32)
        camada = cv2.resize(grade, (w, h), interpolation=cv2.INTER_CUBIC)
        resultado += camada * amplitude
        total += amplitude
        amplitude *= persistencia
    resultado /= total
    resultado -= resultado.min()
    resultado /= (resultado.max() + 1e-6)
    return resultado


def gradiente_direcional(h, w):
    # rampa de 0 a 1 numa direção aleatória (luz vindo de um lado)
    angulo = random.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (xx / max(w, 1)) * np.cos(angulo) + (yy / max(h, 1)) * np.sin(angulo)
    g -= g.min()
    g /= (g.max() + 1e-6)
    return g.astype(np.float32)


# ---------------------------------------------------------------
# 1 ruído gaussiano
# ---------------------------------------------------------------

def ruido_gaussiano(img, intensidade=0.5):
    h, w = img.shape[:2]
    sigma = (3 + intensidade * 28) * random.uniform(0.85, 1.15)

    # o ruído é levemente borrado para ter correlação entre pixels vizinhos, parecido com a granulação de uma câmera real
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    def gerar_ruido(escala):
        r = np.random.normal(0, 1, (h, w)).astype(np.float32)
        r = cv2.GaussianBlur(r, (0, 0), 0.6)
        return (r / (r.std() + 1e-6)) * escala

    ycc[..., 0] += gerar_ruido(sigma)          # luminância: ruído mais forte
    ycc[..., 1] += gerar_ruido(sigma * 0.35)   # cor: mais fraco, como nas câmeras
    ycc[..., 2] += gerar_ruido(sigma * 0.35)

    return cv2.cvtColor(limitar_uint8(ycc), cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------
# 2 ruído de sensor
# ---------------------------------------------------------------

def ruido_sensor(img, intensidade=0.5):
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    # perda leve de nitidez (lente/sensor)
    img_f = cv2.GaussianBlur(img_f, (0, 0), 0.2 + 0.4 * intensidade)

    # ruído de Poisson (shot noise): mais forte nas partes escuras
    escala = 60.0 * (1 - intensidade) + 6.0 * intensidade
    escala = max(escala, 2.0)
    saida = np.random.poisson(np.clip(img_f, 0, 255) * escala) / escala
    saida = saida.astype(np.float32)

    # ruído de leitura do circuito
    saida += np.random.normal(0, 1.5 + 5.0 * intensidade, img.shape).astype(np.float32)

    # faixas leves por coluna e por linha (leitura do sensor CMOS)
    saida += np.random.normal(0, 0.8 + 2.0 * intensidade, (1, w, 1)).astype(np.float32)
    saida += np.random.normal(0, 0.4 + 1.2 * intensidade, (h, 1, 1)).astype(np.float32)

    # alguns pixels defeituosos (pretos ou brancos)
    n_defeitos = max(1, int(h * w * 1.5e-5 * (1 + 6 * intensidade)))
    ys = np.random.randint(0, h, n_defeitos)
    xs = np.random.randint(0, w, n_defeitos)
    for y, x in zip(ys, xs):
        saida[y, x] = random.choice([0, 0, 255])

    saida = limitar_uint8(saida)

    # compressão
    if intensidade > 0.15:
        qualidade = int(95 - 35 * intensidade)
        ok, buffer = cv2.imencode(".jpg", saida, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
        if ok:
            saida = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return saida


# ---------------------------------------------------------------
# 3 névoa
# ---------------------------------------------------------------

def nevoa(img, intensidade=0.5, cor_nevoa=(235, 235, 235), cobertura_max=0.55):
    # manchas de névoa em partes da imagem. a intensidade controla a área coberta (até a cobertura maxima) e a opacidade das manchas
    h, w = img.shape[:2]

    # mapa de onde a névoa pode aparecer
    densidade = ruido_fractal((h, w), celulas_base=3, oitavas=6, persistencia=0.55)

    # fração coberta: aleatória e proporcional a intensidade
    cobertura = random.uniform(0.10, cobertura_max) * (0.4 + 0.6 * intensidade)
    cobertura = min(cobertura, 0.85)

    limiar = np.quantile(densidade, 1 - cobertura)
    transicao = 0.06 + 0.05 * intensidade
    opacidade = transicao_suave(limiar - transicao, limiar + transicao, densidade)

    # mais densa no centro das manchas
    opacidade *= (0.45 + 0.55 * densidade) * (0.55 + 0.45 * intensidade)
    opacidade = cv2.GaussianBlur(opacidade, (0, 0), sigmaX=max(h, w) * 0.004 + 1.5)
    opacidade = np.clip(opacidade, 0, 0.92)   # nunca 100% opaca, o terreno sempre aparece

    cor = np.array(cor_nevoa, dtype=np.float32).reshape(1, 1, 3)
    a = opacidade[:, :, None]

    # perda de nitidez onde a névoa é mais densa
    img_borrada = cv2.GaussianBlur(img, (0, 0), 1.2)
    base = img.astype(np.float32) * (1 - a * 0.5) + img_borrada.astype(np.float32) * (a * 0.5)

    resultado = base * (1 - a) + cor * a
    return limitar_uint8(resultado)


# ---------------------------------------------------------------
# 4 variação de iluminação
# ---------------------------------------------------------------

def variacao_iluminacao(img, intensidade=0.5):
    # 1) exposição: clareia ou escurece a cena toda
    # 2) gamma: muda os meios-tons
    # 3) gradiente: luz mais forte de um lado da imagem 
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0

    ev = random.uniform(-1.2, 1.2) * (0.3 + 0.9 * intensidade)
    x = x * (2.0 ** ev)

    # gamma > 1 escurece os meios-tons, < 1 clareia
    gamma = 1.0 + random.uniform(-0.5, 0.5) * (0.4 + 0.8 * intensidade)
    gamma = max(0.35, gamma)
    x = np.clip(x, 0, None) ** gamma

    # gradiente de luz em direção aleatória
    grad = gradiente_direcional(h, w)
    forca = 0.25 + 0.45 * intensidade
    ganho = 1.0 + (grad - 0.5) * forca
    x *= ganho[:, :, None]

    return limitar_uint8(x * 255.0)


# ---------------------------------------------------------------
# 5 oclusões
# ---------------------------------------------------------------

def mancha_organica(h, w, fracao_area):
    # máscara (0 a 1) de uma mancha irregular em posição aleatória
    cy, cx = random.uniform(0, h), random.uniform(0, w)
    raio = np.sqrt(max(fracao_area, 0.001) * h * w / np.pi)
    sigma = raio * 1.15
    achatamento = random.uniform(0.65, 1.5)
    angulo = random.uniform(0, np.pi)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(angulo) + dy * np.sin(angulo)
    v = -dx * np.sin(angulo) + dy * np.cos(angulo)
    elipse = np.exp(-((u / (sigma * achatamento)) ** 2 +
                      (v / (sigma / achatamento)) ** 2) / 2).astype(np.float32)

    textura = ruido_fractal((h, w), celulas_base=4, oitavas=5, persistencia=0.55)
    campo = elipse + 0.4 * (textura - 0.5)

    limiar = np.quantile(campo, 1 - min(fracao_area, 0.9))
    return transicao_suave(limiar - 0.04, limiar + 0.04, campo)


def oclusao(img, intensidade=0.5, n_max=3):
    # adiciona de 1 a n_max oclusões, de três tipos:
    #   nuvem:  mancha clara semitransparente
    #   sombra: mancha escura semitransparente
    #   perda:  retângulo ou faixa de perda de dados do sensor

    saida = img.astype(np.float32)
    h, w = saida.shape[:2]
    n = random.randint(1, max(1, int(round(n_max * intensidade)) + 1))

    for _ in range(n):
        tipo = random.choices(["nuvem", "sombra", "perda"], weights=[0.4, 0.35, 0.25])[0]
        area = random.uniform(0.015, 0.07) * (0.5 + 1.1 * intensidade)

        if tipo == "nuvem":
            m = mancha_organica(h, w, area)
            textura = ruido_fractal((h, w), celulas_base=10, oitavas=4, persistencia=0.6)
            cor = (215 + 25 * textura)[..., None] * np.array([1.0, 0.99, 0.97], np.float32)
            alpha = (m * random.uniform(0.55, 0.85))[..., None]
            saida = saida * (1 - alpha) + cor * alpha

        elif tipo == "sombra":
            m = mancha_organica(h, w, area)
            forca = random.uniform(0.35, 0.65)
            tom = np.array([0.80, 0.90, 1.0], np.float32)   # sombra um pouco azulada
            saida = saida * (1 - (m * forca)[..., None] * tom)

        else:
            # perda de dados: bloco retangular ou faixa fina
            if random.random() < 0.55:
                area_px = area * h * w
                proporcao = random.uniform(0.6, 1.7)
                rh = int(np.sqrt(area_px / proporcao))
                rw = int(rh * proporcao)
            else:
                if random.random() < 0.5:   # faixa horizontal
                    rh = max(3, int(random.uniform(0.012, 0.03) * h))
                    rw = int(random.uniform(0.25, 0.7) * w)
                else:                       # faixa vertical
                    rw = max(3, int(random.uniform(0.012, 0.03) * w))
                    rh = int(random.uniform(0.25, 0.7) * h)

            y0 = random.randint(0, max(1, h - rh))
            x0 = random.randint(0, max(1, w - rw))
            m = np.zeros((h, w), dtype=np.float32)
            m[y0:y0 + rh, x0:x0 + rw] = 1.0
            # borda suavizada, proporcional ao tamanho da falha
            sigma_borda = max(1.5, min(rh, rw) * 0.12)
            m = cv2.GaussianBlur(m, (0, 0), sigma_borda)

            nivel = random.choice([
                random.uniform(0, 25),      # escuro
                random.uniform(230, 255),   # estourado
                random.uniform(90, 150),    # cinza
            ])
            linhas = np.linspace(0, 1, h).reshape(h, 1, 1)
            listras = 6 * np.sin(linhas * h * np.pi / 3 + random.uniform(0, 6))
            ruido = np.random.normal(0, 3, (h, w, 1)).astype(np.float32)
            preenchimento = np.clip(nivel + listras + ruido, 0, 255)
            saida = saida * (1 - m[..., None]) + preenchimento * m[..., None]

    return limitar_uint8(saida)


PERTURBACOES = {
    "gaussian": ruido_gaussiano,
    "sensor": ruido_sensor,
    "fog": nevoa,
    "illumination": variacao_iluminacao,
    "occlusion": oclusao,
}