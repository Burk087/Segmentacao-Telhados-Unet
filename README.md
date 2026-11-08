<div align="center">

# Segmentação de Telhados em Imagens Aéreas com U-Net

Algoritmo de segmentação de telhados robusto a perturbações de imagem

Trabalho de Conclusão de Curso — Ciência da Computação<br>
Universidade Tuiuti do Paraná (UTP) — 2026

</div>

---

## Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Como o algoritmo funciona](#como-o-algoritmo-funciona)
- [Resultados](#resultados)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como usar o modelo](#como-usar-o-modelo)
- [Como reproduzir o treinamento](#como-reproduzir-o-treinamento)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Versões desenvolvidas](#versões-desenvolvidas)
- [Problemas comuns](#problemas-comuns)
- [Autores](#autores)

---

## Sobre o projeto

Redes neurais convolucionais costumam ter ótimo desempenho em imagens de boa qualidade, mas
perdem desempenho quando a imagem tem ruído, névoa, variação de iluminação ou partes encobertas,
situações comuns na captura aérea real.

Este trabalho desenvolveu um algoritmo de segmentação de telhados em imagens aéreas que mantém o
desempenho mesmo com essas perturbações. Para cada imagem, o algoritmo encontra os telhados e
separa cada telhado individualmente, inclusive quando estão encostados uns nos outros.

Principais características:

- U-Net implementada e treinada do zero, a partir de pesos aleatórios
- Sem transferência de aprendizado, sem modelos pré-treinados e sem dados externos
- Treinada apenas com um conjunto de dados próprio, de imagens aéreas do bairro Adhemar Garcia, em Joinville/SC
- Gerador de perturbações próprio, com 5 tipos e intensidade controlada
- Avaliação em imagens limpas e em 20 condições perturbadas (5 tipos × 4 intensidades)

---

## Como o algoritmo funciona

<div align="center">

Ortofoto<br>
↓<br>
Recorte em 1024 x 1024<br>
↓<br>
Anotação dos telhados<br>
↓<br>
Máscaras (fundo, interior e borda)<br>
↓<br>
Treino da U-Net com perturbações<br>
↓<br>
Pós-processamento<br>
↓<br>
Telhados individuais

</div>

1. Recorte: cada ortofoto é dividida em imagens de 1024 × 1024 pixels.
2. Máscaras: as anotações viram máscaras de 3 classes. A borda do telhado é uma classe própria,
   e é ela que permite separar telhados encostados.
3. Perturbações: durante o treino, 70% das imagens recebem uma das perturbações da tabela abaixo,
   sorteada, com intensidade também sorteada.
4. Treino: a U-Net aprende a classificar cada pixel como fundo, interior ou borda. A perda
   combina entropia cruzada ponderada, Dice e um mapa de pesos que reforça a região entre
   telhados próximos.
5. Pós-processamento: a partir do interior de cada telhado, o algoritmo cresce a região até a
   borda, separando cada telhado, e descarta regiões pequenas ou com baixa confiança.

<div align="center">

| Perturbação | O que simula |
|:---:|:---:|
| Ruído gaussiano | granulação da imagem |
| Ruído de sensor | ruído do sensor da câmera, pixels defeituosos e compressão |
| Névoa | névoa ou nuvens em partes da imagem |
| Iluminação | mudança de exposição e luz mais forte de um lado |
| Oclusão | nuvem, sombra ou perda de dados do sensor |

</div>

---

## Resultados

Resultados do modelo final (V3) no conjunto de teste, formado por ortofotos que não foram usadas
em nenhuma etapa do desenvolvimento:

<div align="center">

| Métrica (por telhado) | Imagens limpas | Média das 20 perturbações | Intensidade máxima |
|:---:|:---:|:---:|:---:|
| F1 | 0,825 | 0,811 | 0,791 |

</div>

Nas imagens limpas, a precisão por telhado foi de 86,1% e o recall foi de 79,2%. Com
perturbações, o F1 caiu apenas 1,4 ponto percentual em média.

Para medir o efeito das perturbações no treino, a mesma rede foi treinada também sem elas
(modelo de referência, mesma configuração e mesma semente 42 da V3) e avaliada no mesmo teste:

<div align="center">

| F1 por telhado (teste, semente 42) | Imagens limpas | Média das 20 perturbações | Intensidade máxima | Ruído gaussiano |
|:---:|:---:|:---:|:---:|:---:|
| V3 (com perturbações no treino) | 0,832 | 0,819 | 0,803 | 0,805 |
| Referência (sem perturbações no treino) | 0,819 | 0,692 | 0,607 | 0,540 |

</div>

Sem as perturbações no treino, a queda nas imagens perturbadas passa de 1,3 para 12,7 pontos
percentuais, enquanto nas imagens limpas os dois modelos ficam praticamente iguais.

<div align="center">

| Dados | Treino | Validação | Teste |
|:---:|:---:|:---:|:---:|
| Ortofotos | 8 | 2 | 2 |
| Imagens | 288 | 72 | 72 |
| Telhados anotados | 5.216 | 1.446 | 1.002 |

</div>

A divisão foi feita por ortofoto, para que recortes da mesma região não ficassem em conjuntos
diferentes.

---

## Requisitos

<div align="center">

| Item | Versão | Obrigatório? |
|:---:|:---:|:---:|
| Sistema operacional | Windows 10/11, Linux ou macOS | sim |
| Python | 3.12 | sim |
| Git | qualquer versão recente | sim (para baixar o projeto) |
| Placa de vídeo NVIDIA | com suporte a CUDA 12.8 e 8 GB de memória | não, mas recomendado |
| Memória RAM | 16 GB ou mais | recomendado para o treinamento |
| Espaço em disco | cerca de 5 GB (bibliotecas e modelos) | sim |

</div>

<div align="center">

| Biblioteca Python | Versão | Uso |
|:---:|:---:|:---:|
| PyTorch | 2.11.0 | rede neural e treinamento |
| NumPy | 2.2.6 | operações com matrizes |
| OpenCV | 4.12.0 | leitura e processamento das imagens |
| Pillow | 12.1.1 | recorte das ortofotos |
| Matplotlib | 3.11.2 | gráficos e figuras |

</div>

> Sem placa NVIDIA o projeto também funciona, mas na CPU. Usar o modelo em imagens novas
> continua rápido; já o treinamento fica muito mais lento. No macOS o projeto sempre roda na CPU,
> porque o Mac não tem placa NVIDIA, e só funciona em Macs com chip Apple (M1 ou mais novo).
> No Linux, os comandos de instalação abaixo usam o Python 3.12 do Ubuntu 24.04 ou mais novo.

O desenvolvimento foi feito no Windows 11, com uma placa NVIDIA RTX 5060 de 8 GB.

---

## Instalação

No Windows, os comandos são digitados no Prompt de Comando (aperte `Windows + R`, digite `cmd`
e aperte Enter). No Linux e no macOS, no Terminal.

### 1. Instalar o Python 3.12 e o Git

Windows:

```
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```

Depois de instalar, feche e abra o cmd de novo.

Linux (Ubuntu/Debian):

```
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

macOS (precisa do Homebrew; se ainda não tiver, instale com a primeira linha):

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 git
```

### 2. Instalar o driver da placa NVIDIA (só para quem tem NVIDIA)

Windows: baixar e instalar o driver mais recente pelo site da NVIDIA
([nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)).

Linux (Ubuntu):

```
sudo ubuntu-drivers install
```

Depois, reinicie o computador.

macOS: não se aplica.

Não é preciso instalar o CUDA separadamente, porque ele já vem junto com o PyTorch.

### 3. Baixar o projeto

Windows:

```
cd %USERPROFILE%
git clone https://github.com/Burk087/segmentacao-telhados-unet.git
cd segmentacao-telhados-unet
```

Linux e macOS:

```
cd ~
git clone https://github.com/Burk087/segmentacao-telhados-unet.git
cd segmentacao-telhados-unet
```

O projeto fica na pasta do usuário, em `segmentacao-telhados-unet`.

### 4. Criar e ativar o ambiente virtual

O ambiente virtual é uma pasta com um Python separado só para o projeto. Assim, as bibliotecas
nas versões usadas no TCC não entram em conflito com outros programas do computador. No
Windows ele é opcional (sem ele, é só pular para a etapa 5); no Linux e no macOS ele é
necessário, porque o sistema bloqueia o `pip install` fora de um ambiente virtual.

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Linux:

```
python3 -m venv venv
source venv/bin/activate
```

macOS:

```
python3.12 -m venv venv
source venv/bin/activate
```

O nome `(venv)` aparece no começo da linha quando o ambiente está ativo. Sempre que abrir um
novo terminal para usar o projeto, entre na pasta e ative o ambiente de novo (só a segunda linha).

### 5. Instalar o PyTorch

Windows ou Linux, com placa NVIDIA:

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Windows, sem placa NVIDIA:

```
pip install torch==2.11.0
```

Linux, sem placa NVIDIA:

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

macOS:

```
pip install torch==2.11.0
```

### 6. Instalar as outras bibliotecas

Igual nos três sistemas:

```
pip install numpy==2.2.6 opencv-python==4.12.0.88 pillow==12.1.1 matplotlib==3.11.2
```

Pronto. Com o ambiente virtual ativo, os comandos das próximas seções usam `python` nos três
sistemas.

---

## Como usar o modelo

O modelo final já treinado está em `experimentos/unet_v3`. Os comandos são rodados dentro da
pasta do projeto, com o ambiente virtual ativo (etapa 4 da instalação).

Para encontrar os telhados em todas as imagens de uma pasta:

```
python prever_imagem.py --entrada pasta_com_imagens --saida previsoes
```

Também é possível passar uma imagem só:

```
python prever_imagem.py --entrada foto.png
```

Para usar outro modelo treinado, por exemplo a V5, basta indicar a pasta dele:

```
python prever_imagem.py --entrada foto.png --modelo experimentos/unet_v5
```

Formatos de imagem aceitos: JPG, PNG, TIF e BMP. Se a pasta de saída não for informada, os
resultados vão para a pasta `previsoes`.

Para cada imagem, são gerados na pasta de saída:

<div align="center">

| Arquivo | Conteúdo |
|:---:|:---:|
| `<nome>_telhados.png` | imagem com cada telhado pintado de uma cor e contornado |
| `<nome>_mascara.png` | classe de cada pixel (0 = fundo, 1 = interior, 2 = borda) |
| `<nome>_instancias.png` | cada telhado com um número próprio (0 = fundo) |
| `telhados.csv` | um telhado por linha: imagem, número, área em pixels e confiança |

</div>

> As imagens devem ser aéreas, vistas de cima, no mesmo formato dos recortes gerados pelo
> `recortar_ortofotos.py`. Para usar uma ortofoto inteira, recorte-a antes.

---

## Como reproduzir o treinamento

> As imagens do conjunto de dados não estão no repositório, porque pertencem à empresa que as
> cedeu. Para treinar de novo, é preciso ter as imagens anotadas, com a divisão em treino,
> validação e teste feita por ortofoto.

As anotações devem estar no formato YOLO de segmentação (um polígono por telhado), como o
Roboflow exporta, organizadas assim dentro da pasta do projeto:

```
dataset/
├── images/
│   ├── train/    imagens de treino
│   ├── valid/    imagens de validação
│   └── test/     imagens de teste
└── labels/
    ├── train/    um .txt por imagem, com o mesmo nome
    ├── valid/
    └── test/
```

Ordem em que os códigos foram rodados:

<div align="center">

| Passo | Comando | O que faz |
|:---:|:---:|:---:|
| 1 | `python recortar_ortofotos.py --entrada ORTOFOTOS --saida RECORTES` | recorta as ortofotos |
| 2 | anotação no Roboflow e exportação para a pasta `dataset` | anota os telhados |
| 3 | `python gerar_mascaras.py` | cria as máscaras |
| 4 | `python gerar_validacao_perturbada.py` | cria a validação perturbada |
| 5 | `python gerar_validacao_perturbada.py --teste` | cria o teste perturbado |
| 6 | `python treinar_unet.py` | treina a U-Net (versão V3) |
| 7 | `python calibrar_posprocessamento.py --experimento unet_v3` | calibra o pós-processamento |
| 8 | `python avaliar_unet.py --experimento unet_v3` | avalia na validação |
| 9 | `python avaliar_unet.py --experimento unet_v3 --teste-final` | avalia no teste |
| 10 | `python gerar_figuras.py --experimento unet_v3 --conjunto teste` | gera as figuras |

</div>

Para treinar com outras sementes e gerar a comparação entre os modelos:

```
python treinar_unet.py --semente 7 --nome unet_v3_s7
python treinar_unet.py --semente 2026 --nome unet_v3_s2026
python gerar_figuras_comparacao.py
```

Para treinar o modelo de referência sem perturbações, trocar `"p_perturbacao": 0.7` por
`"p_perturbacao": 0.0` no `CONFIG` do `treinar_unet.py` e rodar:

```
python treinar_unet.py --nome unet_sem_perturbacao
python calibrar_posprocessamento.py --experimento unet_sem_perturbacao
python avaliar_unet.py --experimento unet_sem_perturbacao
python avaliar_unet.py --experimento unet_sem_perturbacao --teste-final
```

Depois, voltar o `CONFIG` para `"p_perturbacao": 0.7`.

Se o treino for interrompido (por exemplo, por falta de energia), ele pode continuar de onde parou:

```
python treinar_unet.py --continuar
```

Cada treino fica salvo em `experimentos/<nome>/`. Os principais arquivos são o `melhor.pt`
(pesos do melhor modelo na validação), o `historico.csv` (perda e IoU de cada época) e o
`curvas.png` (gráfico do treino). As avaliações e as figuras também são salvas nessa pasta.

---

## Estrutura do repositório

```
segmentacao-telhados-unet/
├── recortar_ortofotos.py          recorta as ortofotos em 1024 x 1024
├── gerar_mascaras.py              anotações -> máscaras (fundo, interior e borda)
├── perturb_aerial_v3.py           gerador das 5 perturbações
├── gerar_validacao_perturbada.py  cria os conjuntos perturbados de validação e teste
├── treinar_unet.py                U-Net e treinamento
├── calibrar_posprocessamento.py   ajusta o pós-processamento na validação
├── avaliar_unet.py                métricas por pixel e por telhado
├── gerar_figuras.py               figuras de um modelo
├── gerar_figuras_comparacao.py    figuras de comparação no teste
├── prever_imagem.py               usa o modelo final em imagens novas
└── experimentos/
    ├── unet_v3/                   modelo final
    ├── unet_v3_s7/                mesma configuração, semente 7
    ├── unet_v3_s2026/             mesma configuração, semente 2026
    ├── unet_v5/                   segunda melhor versão
    ├── unet_sem_perturbacao/      modelo de referência, treinado sem perturbações
    ├── conjunto_v3/               pós-processamento do conjunto das três redes v3
    └── versoes/                   configuração e histórico de cada versão (V1 a V6)
```

O `perturb_aerial_v3.py` não é rodado sozinho, ele é usado pelo treino e pela geração dos
conjuntos perturbados.

---

## Versões desenvolvidas

O algoritmo foi aprimorado em seis versões. Todas usam o mesmo código; o que muda são só os
valores do `CONFIG`, no começo do `treinar_unet.py`. A configuração completa de cada uma está em
`experimentos/versoes/<versão>/config.json`.

<div align="center">

| Versão | Níveis | Expoente dos pesos | Mapa de separação | Prob. de perturbação | Intensidade | Prob. de escala |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V1 | 4 | 0,5 | não | 0,5 | 0,0 a 1,0 | 0 |
| V2 | 4 | 0,25 | sim (w0 = 5) | 0,5 | 0,0 a 1,0 | 0 |
| V3 (final) | 4 | 0,25 | sim (w0 = 5) | 0,7 | 0,2 a 1,0 | 0 |
| V4 | 5 | 0,25 | sim (w0 = 5) | 0,7 | 0,2 a 1,0 | 0 |
| V5 | 5 | 0,25 | sim (w0 = 10) | 0,7 | 0,2 a 1,0 | 0 |
| V6 | 4 | 0,25 | sim (w0 = 5) | 0,7 | 0,2 a 1,0 | 0,5 |
| Referência | 4 | 0,25 | sim (w0 = 5) | 0 | — | 0 |

</div>

A V3 foi escolhida como versão final: a V5, com um nível a mais e quase três vezes mais
parâmetros (20,6 milhões contra 7,8 milhões), não teve desempenho melhor na identificação dos
telhados com perturbações. A linha "Referência" não é uma versão do desenvolvimento: é a V3
treinada sem perturbações, usada só para medir o efeito delas (ver [Resultados](#resultados)).

---

## Problemas comuns

<details>
<summary>"python" não é reconhecido como comando (Windows)</summary>

Feche e abra o cmd de novo depois de instalar o Python. Se continuar, use `py` no lugar de
`python` (por exemplo, `py -m venv venv`).
</details>

<details>
<summary>"winget" não é reconhecido como comando (Windows)</summary>

O winget vem no Windows 10 (atualizado) e no Windows 11. Se não existir, atualize o
"Instalador de Aplicativo" pela Microsoft Store.
</details>

<details>
<summary>O treino roda na CPU mesmo com placa NVIDIA</summary>

Atualize o driver da NVIDIA e reinstale o PyTorch com suporte à placa:

```
pip uninstall -y torch
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```
</details>

<details>
<summary>Erro de memória da GPU (CUDA out of memory)</summary>

Diminua o `"lote"` no `CONFIG` do `treinar_unet.py` (por exemplo, de 8 para 4). Isso pode mudar
um pouco os resultados do treino.
</details>

<details>
<summary>"Pasta ... já existe" ao treinar</summary>

Já existe um treino com esse nome. Use `--nome outro_nome`, apague a pasta em `experimentos`
ou use `--continuar` para retomar o treino.
</details>

---

## Autores

<div align="center">

| Nome | Papel |
|:---:|:---:|
| Luiz Augusto Manfron Matias | Autor |
| Rafael Luiz Cuypers | Autor |
| Alessandra Dickel Molina | Orientadora |

</div>
