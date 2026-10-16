import base64
import io
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageOps
import pymupdf

# Recorte automático do documento em fotos (opcional: se o OpenCV não
# estiver disponível, o servidor segue funcionando sem o recorte).
try:
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
except Exception:  # pragma: no cover
    np = None
    cv2 = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # até 2 arquivos

ALLOWED_ORIGIN = os.getenv(
    "ALLOWED_ORIGIN",
    "https://wallacemapdecampos-ship-it.github.io",
).rstrip("/")

CORS(
    app,
    resources={r"/*": {"origins": [ALLOWED_ORIGIN]}},
    supports_credentials=False,
)

class GroqLimite(RuntimeError):
    """Limite temporário de tokens da Groq (HTTP 429)."""


RATE_WINDOW = 60
RATE_MAX = 20
_rate = defaultdict(deque)

PROMPT = """
Leia o documento brasileiro e responda SOMENTE JSON válido com estas chaves:
tipo_documento, modelo_documento, nome, cpf, rg, data_nascimento, numero_cnh,
orgao_emissor, uf_emissao, data_primeira_habilitacao, data_emissao, validade,
sexo, nacionalidade, naturalidade_municipio, naturalidade_uf, filiacao_1,
filiacao_2, mrz, campos_incertos.

Regras:
- tipo_documento: CNH, RG ou CIN. modelo_documento: ANTIGO, NOVO ou vazio.
- A imagem pode ser FOTO de celular: o documento pode estar pequeno, inclinado,
  girado ou com reflexo. Leia só o documento e ignore o fundo.
- Não invente dados.
- mrz: se existir a zona de leitura mecânica (linhas com muitos '<', no verso
  da CNH nova ou da CIN), copie essas linhas exatamente, separadas por \\n.
  Se não existir, devolva vazio.
- Leia todas as páginas/imagens como partes do MESMO documento.
- Nome, filiacao_1 e filiacao_2: copie LETRA POR LETRA exatamente como está
  impresso. Não corrija a grafia, não acrescente, não duplique e não remova
  letras (ex.: se está impresso MARIO, devolva MARIO, nunca MARRIO).
- RG/CIN: procure CPF, REGISTRO GERAL/RG, FILIAÇÃO, NASCIMENTO, SEXO,
  ÓRGÃO EXPEDIDOR e DATA DE EXPEDIÇÃO/EMISSÃO.
- Para RG/CIN: filiacao_1 = PAI e filiacao_2 = MÃE.
  Se houver duas linhas sem rótulo, identifique pai/mãe apenas quando estiver claro.
  Se ficar ambíguo, deixe os campos vazios e inclua-os em campos_incertos.
- cpf: somente dígitos.
- rg: letras/números sem pontuação.
- RG antigo: mantenha o número do Registro Geral em rg.
- RG novo/CIN: CPF é o identificador principal.
- data_emissao: use DATA DE EXPEDIÇÃO ou DATA DE EMISSÃO; nunca nascimento.
- Todas as datas em DD/MM/AAAA, com o mês em número (ex.: 04/MAR/1994 => 04/03/1994).
- validade: somente se houver VALIDADE explicitamente impressa.
- naturalidade_municipio: somente o nome COMPLETO da cidade, sem abreviações e sem UF; naturalidade_uf: SIGLA de 2 letras da UF de NASCIMENTO impressa junto da cidade (não use a UF de emissão do documento). Exemplos: 'S.ROQUE - SP' => cidade 'SAO ROQUE', UF 'SP'; 'S PAULO' ou 'S. PAULO' => 'SAO PAULO'. Nunca devolva 'S.ROQUE', 'S PAULO' ou 'CIDADE - SP' como nome final da cidade.
- sexo: MASCULINO ou FEMININO quando explicitamente identificável.
- órgão/UF: SSP-SP => orgao_emissor SSP e uf_emissao SP.
- Se aparecer IIRGD/IRGD em SP, mantenha a sigla em orgao_emissor; não converta aqui.
- orgao_emissor: copie a sigla/nome do órgão exatamente como impresso
  (ex.: SSP, IIRGD, SESP, SDS, SEDS, PC, PCMG, IFP, DETRAN, IGP, CASA CIVIL).
- CNH: leia SEMPRE pelos RÓTULOS OFICIAIS, nunca apenas pela posição:
  * data_nascimento = somente a DATA do campo 3 "DATA, LOCAL E UF DE NASCIMENTO";
  * data_emissao = somente o campo 4a "DATA EMISSÃO";
  * validade = somente o campo 4b "VALIDADE";
  * numero_cnh = somente o campo 5 "Nº REGISTRO", normalmente com 11 dígitos.
    IGNORE número vertical, número do espelho, código de segurança, RENACH e outros números;
  * data_primeira_habilitacao = somente o valor ao lado do texto explícito
    "1ª HABILITAÇÃO" / "PRIMEIRA HABILITAÇÃO".
    NUNCA use a validade nem datas da tabela de categorias como 1ª habilitação;
  * NÃO TROQUE 1ª HABILITAÇÃO, DATA EMISSÃO e VALIDADE entre si;
  * antes de responder, confira novamente Nº REGISTRO, 1ª HABILITAÇÃO,
    DATA EMISSÃO e VALIDADE pelos respectivos rótulos;
  * se um desses campos não estiver claramente legível, deixe vazio em vez de adivinhar.
- Se não estiver legível, deixe vazio e inclua a chave em campos_incertos.
""".strip()


def _ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def _rate_ok():
    now = time.time()
    q = _rate[_ip()]
    while q and q[0] < now - RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_MAX:
        return False
    q.append(now)
    return True


def _origin_ok():
    origin = (request.headers.get("Origin") or "").rstrip("/")
    # Permite chamadas sem Origin apenas para health/diagnóstico direto.
    if not origin:
        return request.path == "/health"
    return origin == ALLOWED_ORIGIN


@app.before_request
def _guard():
    if request.method == "OPTIONS":
        return None
    if not _origin_ok():
        return jsonify({"ok": False, "error": "Origem não autorizada."}), 403
    if request.path != "/health" and not _rate_ok():
        return jsonify({"ok": False, "error": "Muitas requisições. Aguarde um minuto."}), 429
    return None


# ============================================================
# PREPARO DA IMAGEM (precisão de leitura)
# A Groq cobra o mesmo número de tokens por imagem, seja qual for o
# tamanho (o Qwen reduz tudo para a mesma resolução). Por isso, quanto
# mais a imagem for "só o documento", maiores ficam as letras para a IA.
# ============================================================

ALVO_LADO_DOCUMENTO = 1600


def _abrir_imagem(data: bytes) -> Image.Image:
    """Abre a imagem respeitando a rotação gravada pelo celular (EXIF)."""
    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def _ordenar_cantos(pts):
    """Ordena 4 pontos: sup-esq, sup-dir, inf-dir, inf-esq."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # Ordena pelo ângulo em volta do centro (funciona mesmo girado a 45°)
    # e começa pelo canto mais próximo do topo-esquerdo.
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(angulos)]
    inicio = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -inicio, axis=0).astype(np.float32)


def _quad_de_contorno(contorno, preenchimento_min=0.85):
    peri = cv2.arcLength(contorno, True)
    if peri <= 0:
        return None
    aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
    if len(aprox) == 4 and cv2.isContourConvex(aprox):
        return aprox.reshape(4, 2).astype(np.float32)
    # Cartões têm cantos arredondados: aceita um contorno "quase retângulo".
    rect = cv2.minAreaRect(contorno)
    rw, rh = rect[1]
    if rw * rh <= 0:
        return None
    if cv2.contourArea(contorno) / (rw * rh) >= preenchimento_min:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def _medidas_quad(q):
    q = _ordenar_cantos(q)
    larg = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    alt = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    return float(larg), float(alt)


def _iou_caixas(a, b):
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    uni = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / uni if uni > 0 else 0.0


def _mapa_texto(cinza):
    """Pixels com cara de letra/número (escuro no claro e claro no escuro)."""
    k = max(9, int(min(cinza.shape) / 40)) | 1
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    escuro = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, nucleo)
    claro = cv2.morphologyEx(cinza, cv2.MORPH_TOPHAT, nucleo)
    _, texto = cv2.threshold(cv2.max(escuro, claro), 40, 255, cv2.THRESH_BINARY)
    return texto


def _texto_fica_dentro(quad, texto):
    """
    Garante que o recorte não deixa texto do documento de fora:
    a densidade de "letras" fora do quadrilátero tem de ser bem menor
    que a de dentro.
    """
    mascara = np.zeros(texto.shape, np.uint8)
    cv2.fillConvexPoly(mascara, _ordenar_cantos(quad).astype(np.int32), 255)
    dentro = mascara > 0
    area_in = int(dentro.sum())
    area_out = int(dentro.size - area_in)
    if area_in == 0:
        return False
    tx = texto > 0
    dens_in = tx[dentro].sum() / area_in
    if dens_in < 0.01:
        return False
    if area_out == 0:
        return True
    dens_out = tx[~dentro].sum() / area_out
    return dens_out <= 0.35 * dens_in


def _localizar_documento(img: Image.Image):
    """
    Procura o contorno do documento na foto.
    Devolve (4 cantos na escala da imagem original, fração da área) ou None.
    Critérios conservadores: na dúvida, não recorta.
    """
    W, H = img.size
    escala = min(1.0, 1000.0 / max(W, H))
    peq = img if escala >= 1.0 else img.resize(
        (max(1, int(W * escala)), max(1, int(H * escala))),
        Image.Resampling.BILINEAR,
    )
    cinza = np.asarray(peq.convert("L"))
    h, w = cinza.shape
    area_total = float(w * h)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    med = float(np.median(suave))
    texto = _mapa_texto(suave)

    mascaras = []
    # 1) Bordas do cartão.
    for lo, hi in (
        (int(max(0, 0.66 * med)), int(min(255, 1.33 * med))),
        (30, 90),
        (60, 180),
    ):
        bordas = cv2.Canny(suave, lo, hi)
        bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=2)
        bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mascaras.append(bordas)
    # 2) Documento mais claro/escuro ou mais colorido que o fundo.
    sat = cv2.GaussianBlur(
        cv2.cvtColor(np.asarray(peq), cv2.COLOR_RGB2HSV)[:, :, 1], (5, 5), 0
    )
    for canal in (suave, sat):
        _, otsu = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for m in (otsu, 255 - otsu):
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            mascaras.append(m)
    # 3) Mancha de texto (útil quando a borda do cartão quase não aparece).
    k = max(15, int(min(h, w) / 25)) | 1
    mancha = cv2.morphologyEx(texto, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mancha = cv2.morphologyEx(mancha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    candidatos = []
    # A mancha de texto é irregular (foto 3x4, campos vazios): aceita menos
    # preenchimento; as travas de proporção e de "texto dentro" seguem valendo.
    for m, preench in [(x, 0.85) for x in mascaras] + [(mancha, 0.6)]:
        contornos, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:12]:
            q = _quad_de_contorno(c, preench)
            if q is None:
                continue
            area = float(cv2.contourArea(_ordenar_cantos(q)))
            frac = area / area_total
            # Documento já grande na foto: o ganho é pequeno e o risco de
            # cortar um pedaço não compensa.
            if not (0.03 <= frac <= 0.6):
                continue
            larg, alt = _medidas_quad(q)
            if min(larg, alt) < 30:
                continue
            razao = max(larg, alt) / min(larg, alt)
            # Cartão/painel de documento (~1,4 a 1,6) ou dois painéis lado a lado.
            if not (1.2 <= razao <= 1.95 or 2.5 <= razao <= 3.6):
                continue
            if any(_iou_caixas(q, c2) > 0.9 for _, c2 in candidatos):
                continue
            candidatos.append((area, q))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[0], reverse=True)

    # Frente e verso lado a lado: junta dois painéis parecidos e vizinhos.
    def _juntar(area, melhor):
        larg, alt = _medidas_quad(melhor)
        centro = melhor.mean(axis=0)
        for area2, q2 in candidatos:
            if q2 is melhor or not (0.6 <= area2 / area <= 1.6):
                continue
            if _iou_caixas(melhor, q2) > 0.2:
                continue
            if np.linalg.norm(q2.mean(axis=0) - centro) > 1.5 * max(larg, alt):
                continue
            juntos = cv2.boxPoints(
                cv2.minAreaRect(np.vstack([melhor, q2]))
            ).astype(np.float32)
            area_j = float(cv2.contourArea(_ordenar_cantos(juntos)))
            if area_j / area_total <= 0.7:
                return area_j, juntos
        return area, melhor

    for area, q in candidatos:
        area, q = _juntar(area, q)
        if _texto_fica_dentro(q, texto):
            return q / escala, area / area_total
    return None


def _recortar_documento(img: Image.Image):
    """
    Recorta e endireita o documento quando ele está pequeno/inclinado na foto.
    Se não tiver certeza, devolve a imagem original.
    """
    if cv2 is None or np is None:
        return img, "sem_opencv"
    try:
        achado = _localizar_documento(img)
        if not achado:
            return img, "sem_recorte"
        quad, frac = achado
        quad = _ordenar_cantos(quad)

        # Margem de segurança para não cortar a borda do documento.
        centro = quad.mean(axis=0)
        quad = centro + (quad - centro) * 1.05
        quad[:, 0] = np.clip(quad[:, 0], 0, img.width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, img.height - 1)

        larg, alt = _medidas_quad(quad)
        larg, alt = int(round(larg)), int(round(alt))
        if larg < 60 or alt < 60:
            return img, "sem_recorte"
        destino = np.array(
            [[0, 0], [larg - 1, 0], [larg - 1, alt - 1], [0, alt - 1]],
            dtype=np.float32,
        )
        matriz = cv2.getPerspectiveTransform(quad.astype(np.float32), destino)
        arr = cv2.warpPerspective(
            np.asarray(img), matriz, (larg, alt),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        recorte = Image.fromarray(arr)

        # Recorte "vazio" (ex.: folha em branco): descarta.
        if float(np.asarray(recorte.convert("L")).std()) < 18:
            return img, "sem_recorte"

        # Documento pequeno na foto: amplia com qualidade e realça as letras.
        maior = max(recorte.size)
        if maior < ALVO_LADO_DOCUMENTO:
            fator = ALVO_LADO_DOCUMENTO / maior
            recorte = recorte.resize(
                (int(recorte.width * fator), int(recorte.height * fator)),
                Image.Resampling.LANCZOS,
            )
            if fator >= 1.3:
                recorte = recorte.filter(
                    ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=2)
                )
        return recorte, f"recortado({frac:.0%})"
    except Exception:
        return img, "erro_recorte"


def _pdf_area_documento(page):
    """
    PDF digital (ex.: CNH exportada do app): o cartão vem como imagens
    coloridas grandes e o resto da página é cabeçalho/QR code.
    Devolve (retângulo só com o cartão, zoom ideal) ou (None, None).
    """
    try:
        pagina = page.rect
        area_pag = pagina.width * pagina.height
        imagens = []
        for info in page.get_image_info():
            larg_px = int(info.get("width") or 0)
            alt_px = int(info.get("height") or 0)
            comps = int(info.get("colorspace") or 0)
            bpc = int(info.get("bpc") or 0)
            if larg_px < 600 or alt_px < 350 or comps < 3 or bpc < 8:
                continue
            caixa = pymupdf.Rect(info["bbox"]) & pagina
            if caixa.is_empty or caixa.width < 50 or caixa.height < 30:
                continue
            razao = max(caixa.width, caixa.height) / min(caixa.width, caixa.height)
            if not (1.2 <= razao <= 1.95):
                continue
            imagens.append((caixa, larg_px))
        if not imagens:
            return None, None

        uniao = pymupdf.Rect(imagens[0][0])
        for caixa, _ in imagens[1:]:
            uniao |= caixa
        frac = (uniao.width * uniao.height) / area_pag
        if not (0.08 <= frac <= 0.8):
            return None, None

        # Se houver dado do cliente (data, CPF) escrito FORA das imagens,
        # não recorta, para não perder informação.
        for bloco in page.get_text("blocks"):
            caixa_txt = pymupdf.Rect(bloco[:4])
            if caixa_txt.intersects(uniao):
                continue
            texto = str(bloco[4] or "")
            if re.search(r"\d{2}/\d{2}/\d{4}|\d{3}\.?\d{3}\.?\d{3}-?\d{2}", texto):
                return None, None

        uniao = (uniao + (-6, -6, 6, 6)) & pagina
        zoom = max(lp / cx.width for cx, lp in imagens)
        zoom = min(4.5, max(2.2, zoom))
        return uniao, zoom
    except Exception:
        return None, None


# Recorte do PDF digital (só os painéis do cartão) DESLIGADO: no teste real,
# a imagem alta e estreita fez a IA errar uma letra da filiação, enquanto a
# página inteira lia certo. Mantido o envio da página inteira, já validado.
RECORTE_PDF_DIGITAL = False


def _pdf_pagina_para_imagem(page):
    """Renderiza a página. Devolve (imagem, pode_recortar_como_foto)."""
    clip, zoom = (
        _pdf_area_documento(page) if RECORTE_PDF_DIGITAL else (None, None)
    )
    if clip is not None:
        try:
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False
            )
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            return img.copy(), False
        except Exception:
            pass

    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").copy()

    # PDF escaneado (uma foto/scan ocupando a página, sem texto digital):
    # trata como foto e tenta recortar o documento.
    pode_recortar = False
    try:
        texto = (page.get_text("text") or "").strip()
        area_pag = page.rect.width * page.rect.height
        grandes = [
            i for i in page.get_image_info()
            if (pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area_pag
        ]
        pode_recortar = len(texto) < 50 and len(grandes) >= 1
    except Exception:
        pode_recortar = False
    return img, pode_recortar


def _to_jpeg(data: bytes, filename: str, mimetype: str) -> bytes:
    """
    Prioriza precisão de leitura para CNH/RG/CIN:
    - foto: corrige a rotação do celular e recorta/endireita o documento;
    - PDF digital: usa só a área do cartão (sem cabeçalho e QR code);
    - PDF escaneado: tratado como foto.
    """
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            for i in range(min(2, len(doc))):
                img, pode_recortar = _pdf_pagina_para_imagem(doc[i])
                if pode_recortar:
                    img, _ = _recortar_documento(img)
                pages.append(img)

            if not pages:
                raise ValueError("PDF sem páginas válidas.")

            if len(pages) == 1:
                img = pages[0]
            else:
                # Mantém as páginas lado a lado para não esmagar o texto verticalmente.
                gap = 18
                max_h = max(p.height for p in pages)
                total_w = sum(p.width for p in pages) + gap * (len(pages) - 1)
                canvas = Image.new("RGB", (total_w, max_h), "white")
                x = 0
                for p in pages:
                    y = (max_h - p.height) // 2
                    canvas.paste(p, (x, y))
                    x += p.width + gap
                img = canvas
        finally:
            doc.close()
    else:
        img = _abrir_imagem(data)
        img, _ = _recortar_documento(img)

    max_side = 2800
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=91)
    return out.getvalue()


def _combinar_arquivos_identificacao(arquivos):
    """
    Recebe 1 ou 2 arquivos (PDF/imagem) do MESMO documento.
    Ex.: frente + verso do RG em arquivos separados.
    """
    imagens = []

    for item in arquivos[:2]:
        data = item["data"]
        filename = item["filename"]
        mimetype = item["mimetype"]

        jpeg = _to_jpeg(data, filename, mimetype)
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")

        # Cada parte fica leve antes de montar a imagem final.
        max_w = 1650
        if img.width > max_w:
            escala = max_w / img.width
            img = img.resize(
                (max_w, max(1, int(img.height * escala))),
                Image.Resampling.LANCZOS,
            )
        imagens.append(img.copy())

    if not imagens:
        raise ValueError("Nenhum arquivo válido recebido.")

    if len(imagens) == 1:
        out = io.BytesIO()
        imagens[0].save(out, format="JPEG", quality=91)
        return out.getvalue()

    # Frente e verso ficam um abaixo do outro para preservar a resolução.
    margem = 16
    largura = max(img.width for img in imagens)
    altura = margem + sum(img.height + margem for img in imagens)

    canvas = Image.new("RGB", (largura + margem * 2, altura), "white")
    y = margem
    for img in imagens:
        x = margem + (largura - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + margem

    # Limita a imagem final sem reduzir demais cada lado do RG.
    max_side = 2400
    escala = min(1.0, max_side / max(canvas.width, canvas.height))
    if escala < 1.0:
        canvas = canvas.resize(
            (
                max(1, int(canvas.width * escala)),
                max(1, int(canvas.height * escala)),
            ),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=91)
    return out.getvalue()


def PathLikeSuffix(name: str) -> str:
    name = (name or "").lower().strip()
    dot = name.rfind(".")
    return name[dot:] if dot >= 0 else ""


def _groq_call(image_bytes: bytes):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
    ]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]

    last_error = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_completion_tokens": 560,
            "stream": False,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "Assistente-SICAQ-Web/1.0",
            },
            method="POST",
        )
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            content = body["choices"][0]["message"]["content"]
            dados = json.loads(content)
            return dados, model, elapsed
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                espera = ""
                m = re.search(r"try again in\s+([0-9.]+)s", detail, re.I)
                if m:
                    espera = f" Aguarde cerca de {float(m.group(1)):.0f} segundos e tente novamente."
                last_error = GroqLimite(
                    "Limite gratuito do Groq atingido temporariamente." + espera
                )
                break

            last_error = RuntimeError(f"Groq HTTP {e.code}: {detail}")
            # Tenta o próximo modelo apenas em erros de modelo/404/400.
            if e.code not in (400, 404):
                break
        except Exception as e:
            last_error = e
            break

    if isinstance(last_error, GroqLimite):
        raise last_error
    raise RuntimeError(str(last_error or "Falha na leitura Groq/Qwen."))




PROMPT_CNH_CRITICOS = """
Esta imagem é uma CNH brasileira. Leia SOMENTE os 4 campos críticos abaixo
pelos RÓTULOS impressos e responda SOMENTE JSON:

numero_cnh, data_primeira_habilitacao, data_emissao, validade

REGRAS OBRIGATÓRIAS:
- numero_cnh = SOMENTE o campo 5 "Nº REGISTRO". Normalmente possui 11 dígitos.
  Ignore números verticais, espelho, segurança, RENACH, CPF e identidade.
- data_primeira_habilitacao = SOMENTE o valor ao lado do rótulo
  "1ª HABILITAÇÃO" / "PRIMEIRA HABILITAÇÃO".
- data_emissao = SOMENTE o campo 4a "DATA EMISSÃO".
- validade = SOMENTE o campo 4b "VALIDADE".
- NÃO use datas da tabela de categorias.
- NÃO copie a VALIDADE para 1ª HABILITAÇÃO.
- NÃO troque DATA EMISSÃO com 1ª HABILITAÇÃO.
- Confira cada valor uma segunda vez antes de responder.
- Datas em DD/MM/AAAA.
- Se um valor não puder ser confirmado pelo rótulo, devolva vazio.
""".strip()


def _tirar_acentos(valor):
    texto = str(valor or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _texto_pdf_arquivos(arquivos):
    partes = []
    for item in arquivos:
        ext = PathLikeSuffix(item.get("filename", ""))
        mime = item.get("mimetype", "")
        if ext != ".pdf" and mime != "application/pdf":
            continue
        try:
            doc = pymupdf.open(stream=item["data"], filetype="pdf")
            try:
                for i in range(min(len(doc), 4)):
                    txt = doc[i].get_text("text") or ""
                    if txt.strip():
                        partes.append(txt)
            finally:
                doc.close()
        except Exception:
            pass
    return "\n".join(partes)[:12000]


def _groq_cnh_criticos(image_bytes: bytes, texto_pdf: str = ""):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    instrucao = PROMPT_CNH_CRITICOS
    if texto_pdf.strip():
        instrucao += (
            "\n\nTEXTO EXTRAÍDO DO PDF PARA CONFERÊNCIA DOS RÓTULOS:\n"
            + texto_pdf
        )

    models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
    ]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]

    ultimo_erro = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instrucao},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }},
                ],
            }],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_completion_tokens": 180,
            "stream": False,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "Assistente-SICAQ-CNH-Criticos/1.0",
            },
            method="POST",
        )

        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            tempo = time.perf_counter() - t0
            conteudo = body["choices"][0]["message"]["content"]
            dados = json.loads(conteudo)
            return dados, tempo
        except urllib.error.HTTPError as e:
            detalhe = e.read().decode("utf-8", errors="replace")
            ultimo_erro = RuntimeError(f"Groq HTTP {e.code}: {detalhe}")
            if e.code not in (400, 404):
                break
        except Exception as e:
            ultimo_erro = e
            break

    raise RuntimeError(str(ultimo_erro or "Falha na conferência da CNH."))


def _mesclar_cnh_criticos(resultado, criticos):
    r = dict(resultado or {})
    c = dict(criticos or {})

    # Nº Registro de CNH é validado como 11 dígitos.
    numero = re.sub(r"\D", "", str(c.get("numero_cnh") or ""))
    if len(numero) == 11:
        r["numero_cnh"] = numero

    for chave in (
        "data_primeira_habilitacao",
        "data_emissao",
        "validade",
    ):
        valor = str(c.get(chave) or "").strip()
        if _parse_data_br(valor):
            r[chave] = valor

    return r


def _validar_cnh_final(resultado):
    """
    Confere localmente (sem gastar tokens) as 3 datas da CNH.
    Ordem esperada: 1ª habilitação <= emissão < validade.
    Só mexe nas datas quando a ordem lógica está violada; CNH lida corretamente
    não é alterada.
    """
    r = dict(resultado or {})
    if str(r.get("tipo_documento") or "").strip().upper() != "CNH":
        return r

    incertos = [str(x) for x in (r.get("campos_incertos") or [])]

    numero = re.sub(r"\D", "", str(r.get("numero_cnh") or ""))
    if len(numero) == 11:
        r["numero_cnh"] = numero
        # Dígitos verificadores não conferem: provável dígito mal lido.
        # Mantém o valor, mas marca para o usuário conferir.
        if not _cnh_registro_valido(numero):
            incertos.append("numero_cnh")
    else:
        r["numero_cnh"] = ""
        incertos.append("numero_cnh")

    # Formato único DD/MM/AAAA
    for chave in ("data_nascimento", "data_primeira_habilitacao",
                  "data_emissao", "validade"):
        r[chave] = _formatar_data_br(r.get(chave))

    nasc = _parse_data_br(r.get("data_nascimento"))
    primeira = _parse_data_br(r.get("data_primeira_habilitacao"))
    emissao = _parse_data_br(r.get("data_emissao"))
    validade = _parse_data_br(r.get("validade"))

    LIMITE_VALIDADE_DIAS = 11 * 366  # CNH vale 1, 3, 5 ou 10 anos

    # 1) Emissão e validade invertidas: emissão nunca é posterior à validade.
    if emissao and validade and emissao > validade:
        r["data_emissao"], r["validade"] = r["validade"], r["data_emissao"]
        emissao, validade = validade, emissao

    # 2) 1ª habilitação copiada da validade.
    if primeira and validade and primeira == validade:
        r["data_primeira_habilitacao"] = ""
        primeira = None
        incertos.append("data_primeira_habilitacao")

    # 3) 1ª habilitação depois da emissão.
    if primeira and emissao and primeira > emissao:
        troca_plausivel = (
            validade
            and primeira < validade
            and (validade - primeira).days <= LIMITE_VALIDADE_DIAS
        )
        if troca_plausivel:
            # Provável troca entre 1ª habilitação e emissão: reordena e avisa.
            r["data_primeira_habilitacao"], r["data_emissao"] = (
                r["data_emissao"], r["data_primeira_habilitacao"]
            )
            primeira, emissao = emissao, primeira
            incertos += ["data_primeira_habilitacao", "data_emissao"]
        else:
            r["data_primeira_habilitacao"] = ""
            primeira = None
            incertos.append("data_primeira_habilitacao")

    # 4) Intervalo emissão -> validade fora do normal: apenas avisa.
    if emissao and validade:
        dias = (validade - emissao).days
        if dias <= 0 or dias > LIMITE_VALIDADE_DIAS:
            incertos += ["data_emissao", "validade"]

    # 5) Coerência com a data de nascimento.
    if nasc:
        if primeira and primeira <= nasc:
            r["data_primeira_habilitacao"] = ""
            primeira = None
            incertos.append("data_primeira_habilitacao")
        if emissao and emissao <= nasc:
            r["data_emissao"] = ""
            emissao = None
            incertos.append("data_emissao")
        if primeira and (primeira - nasc).days < 18 * 365:
            incertos.append("data_primeira_habilitacao")  # CNH exige 18 anos

    r["campos_incertos"] = sorted(set(incertos))
    return r


MESES_TEXTO = {
    "JANEIRO": 1, "FEVEREIRO": 2, "MARCO": 3, "ABRIL": 4, "MAIO": 5,
    "JUNHO": 6, "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10,
    "NOVEMBRO": 11, "DEZEMBRO": 12,
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
    # Inglês (documentos bilíngues)
    "FEB": 2, "APR": 4, "MAY": 5, "AUG": 8, "SEP": 9, "OCT": 10, "DEC": 12,
}


def _parse_data_br(valor):
    """
    Aceita DD/MM/AAAA, DD-MM-AAAA, DD.MM.AAAA e mês escrito, como no RG
    antigo: 04/MAR/1994, 04 MAR 1994, 4 de março de 1994.
    Devolve date ou None.
    """
    valor = str(valor or "").strip()
    m = re.fullmatch(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})", valor)
    if m:
        dia, mes, ano = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        t = _tirar_acentos(valor).upper()
        m = re.fullmatch(
            r"(\d{1,2})\s*(?:[/.\-]|DE)?\s*([A-Z]{3,9})\.?\s*(?:[/.\-]|DE)?\s*(\d{4})",
            re.sub(r"\s+", " ", t),
        )
        if not m or m.group(2) not in MESES_TEXTO:
            return None
        dia, mes, ano = int(m.group(1)), MESES_TEXTO[m.group(2)], int(m.group(3))
    try:
        return date(ano, mes, dia)
    except Exception:
        return None


def _mrz_digito(campo):
    """Dígito verificador padrão ICAO 9303 (pesos 7, 3, 1)."""
    pesos = (7, 3, 1)
    total = 0
    for i, ch in enumerate(campo):
        if ch.isdigit():
            v = int(ch)
        elif "A" <= ch <= "Z":
            v = ord(ch) - 55
        else:
            v = 0
        total += v * pesos[i % 3]
    return str(total % 10)


def _datas_mrz(texto):
    """
    Lê nascimento e validade da linha MRZ (CNH nova / CIN), ex.:
    0306045F3112168BRA<<<<<<<<<<<6 -> nasc 04/06/2003, validade 16/12/2031.
    Só devolve a data cujo dígito verificador confere.
    """
    bruto = str(texto or "").upper()
    if "<" not in bruto:
        return {}
    limpo = re.sub(r"[^0-9A-Z<\n]", "", bruto.replace(" ", ""))
    achados = {}
    for linha in [limpo.replace("\n", "")] + limpo.split("\n"):
        m = re.search(r"([0-9O]{6})([0-9O])([MFX<])([0-9O]{6})([0-9O])", linha)
        if not m:
            continue
        nasc, dv_nasc, _sexo, venc, dv_venc = (
            g.replace("O", "0") for g in m.groups()
        )
        hoje = date.today()
        if _mrz_digito(nasc) == dv_nasc:
            aa, mm, dd = int(nasc[:2]), int(nasc[2:4]), int(nasc[4:])
            ano = 2000 + aa if aa <= hoje.year % 100 else 1900 + aa
            try:
                achados["data_nascimento"] = date(ano, mm, dd)
            except ValueError:
                pass
        if _mrz_digito(venc) == dv_venc:
            aa, mm, dd = int(venc[:2]), int(venc[2:4]), int(venc[4:])
            try:
                achados["validade"] = date(2000 + aa, mm, dd)
            except ValueError:
                pass
        if achados:
            break
    return achados


def _aplicar_mrz(resultado):
    """
    Confere nascimento e validade com a MRZ (sem chamada extra à Groq).
    Se a MRZ tiver dígito verificador válido, ela prevalece sobre a leitura
    do campo impresso. O sexo NÃO é alterado.
    """
    r = dict(resultado or {})
    datas = _datas_mrz(r.get("mrz"))
    for chave, valor in datas.items():
        lido = _parse_data_br(r.get(chave))
        if lido == valor:
            continue
        # Validade verdadeira lida no campo de EMISSÃO: as duas foram
        # trocadas pela IA; devolve a emissão para o lugar certo.
        if (
            chave == "validade"
            and lido
            and _parse_data_br(r.get("data_emissao")) == valor
        ):
            r["data_emissao"] = f"{lido.day:02d}/{lido.month:02d}/{lido.year:04d}"
        r[chave] = f"{valor.day:02d}/{valor.month:02d}/{valor.year:04d}"
        print(f"[MRZ] {chave} conferido e ajustado pela MRZ", flush=True)

    # Nº de registro da CNH (campo 5) também está na 1ª linha da MRZ.
    if str(r.get("tipo_documento") or "").strip().upper() == "CNH":
        numero_mrz = _numero_cnh_mrz(r.get("mrz"))
        lido = re.sub(r"\D", "", str(r.get("numero_cnh") or ""))
        if numero_mrz and numero_mrz != lido:
            r["numero_cnh"] = numero_mrz
            print("[MRZ] numero_cnh conferido e ajustado pela MRZ", flush=True)
    return r


def _cpf_valido(cpf):
    """Confere os 2 dígitos verificadores do CPF."""
    c = re.sub(r"\D", "", str(cpf or ""))
    if len(c) != 11 or len(set(c)) == 1:
        return False
    for n in (9, 10):
        soma = sum(int(c[i]) * ((n + 1) - i) for i in range(n))
        if (soma * 10) % 11 % 10 != int(c[n]):
            return False
    return True


def _tipo_documento(d):
    """
    Padroniza o tipo para CNH, RG ou CIN.
    Devolve (tipo, deduzido): deduzido=True quando veio dos campos lidos,
    e não do tipo informado pela IA.
    """
    t = _tirar_acentos(str(d.get("tipo_documento") or "")).upper()
    t = re.sub(r"[^A-Z ]", " ", t)
    palavras = set(t.split())
    if "CNH" in palavras or "HABILITACAO" in t or "DRIVER" in t:
        return "CNH", False
    if "CIN" in palavras or "IDENTIDADE NACIONAL" in t:
        return "CIN", False
    if palavras & {"RG", "IDENTIDADE", "REGISTRO"} or "REGISTRO GERAL" in t:
        return "RG", False

    # A IA não informou um tipo conhecido: deduz pelos campos.
    numero_cnh = re.sub(r"\D", "", str(d.get("numero_cnh") or ""))
    if len(numero_cnh) == 11 or str(d.get("data_primeira_habilitacao") or "").strip():
        return "CNH", True
    modelo = str(d.get("modelo_documento") or "").strip().upper()
    cpf = re.sub(r"\D", "", str(d.get("cpf") or ""))
    rg = re.sub(r"[^0-9A-Za-z]", "", str(d.get("rg") or ""))
    if modelo.startswith("NOV") and len(cpf) == 11:
        return "CIN", True
    if rg or len(cpf) == 11:
        return "RG", True
    return "", True


def _cnh_registro_valido(numero):
    """
    Confere os 2 dígitos verificadores do Nº de registro da CNH (11 dígitos),
    pelo cálculo oficial do Renach.
    """
    numero = re.sub(r"\D", "", str(numero or ""))
    if len(numero) != 11 or len(set(numero)) == 1:
        return False
    base = [int(c) for c in numero[:9]]
    dv1 = sum(base[i] * (9 - i) for i in range(9)) % 11
    desconto = 0
    if dv1 >= 10:
        dv1, desconto = 0, 2
    dv2 = sum(base[i] * (1 + i) for i in range(9)) % 11 - desconto
    if dv2 < 0:
        dv2 += 11
    if dv2 >= 10:
        dv2 = 0
    return numero[9:] == f"{dv1}{dv2}"


def _numero_cnh_mrz(texto):
    """
    Lê o Nº de registro da CNH na 1ª linha da MRZ, ex.:
    I<BRA066038085<440<<<<<<<<<<<< -> 066038085 + 44 = 06603808544.
    Só devolve se o dígito da MRZ E os dígitos da CNH conferirem.
    """
    bruto = re.sub(r"[^0-9A-Z<\n]", "", str(texto or "").upper().replace(" ", ""))
    for linha in [bruto.replace("\n", "")] + bruto.split("\n"):
        m = re.search(r"I.BRA([0-9O]{9})([0-9O<])([0-9O]*)", linha)
        if not m:
            continue
        inicio, dv, resto = (g.replace("O", "0") for g in m.groups())
        if dv == "<":
            # Número com mais de 9 dígitos: continua no campo opcional,
            # e o último dígito dessa continuação é o verificador.
            if len(resto) < 2:
                continue
            numero, dv = inicio + resto[:-1], resto[-1]
        else:
            numero = inicio
        if _mrz_digito(numero) == dv and _cnh_registro_valido(numero):
            return numero
    return ""


def _formatar_data_br(valor):
    """Padroniza para DD/MM/AAAA quando for uma data válida; senão mantém o texto."""
    d = _parse_data_br(valor)
    if d:
        return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
    return str(valor or "").strip()


def _corrigir_datas_cnh(d):
    """
    Corrige a inversão comum entre 4a DATA EMISSÃO e 4b VALIDADE.
    Em uma CNH, a emissão não pode ser posterior à validade.
    """
    r = dict(d or {})
    tipo = str(r.get("tipo_documento") or "").strip().upper()
    if tipo != "CNH":
        return r

    emissao_txt = str(r.get("data_emissao") or "").strip()
    validade_txt = str(r.get("validade") or "").strip()
    emissao = _parse_data_br(emissao_txt)
    validade = _parse_data_br(validade_txt)

    if emissao and validade and emissao > validade:
        r["data_emissao"], r["validade"] = validade_txt, emissao_txt

    return r



UFS_BR = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

ESTADOS_POR_NOME = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "GOIAS": "GO", "MARANHAO": "MA",
    "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "MINAS GERAIS": "MG",
    "PARA": "PA", "PARAIBA": "PB", "PARANA": "PR", "PERNAMBUCO": "PE",
    "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO", "RORAIMA": "RR",
    "SANTA CATARINA": "SC", "SAO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}


def _uf_naturalidade(uf_valor, municipio_bruto):
    """
    Devolve a sigla da UF de nascimento (ex.: 'SP'), sempre uma das 27 UFs.
    1) usa naturalidade_uf lida pela IA (sigla ou nome do estado);
    2) se vier vazia/inválida, tenta a sigla no fim do texto da cidade
       ('S.ROQUE - SP', 'SAO PAULO/SP', 'SOROCABA (SP)').
    Nunca usa a UF de emissão do documento, que pode ser de outro estado.
    """
    uf = _tirar_acentos(str(uf_valor or "")).upper()
    uf = re.sub(r"[^A-Z ]", " ", uf)
    uf = re.sub(r"\s+", " ", uf).strip()
    if uf in UFS_BR:
        return uf
    if uf in ESTADOS_POR_NOME:
        return ESTADOS_POR_NOME[uf]

    cidade = _tirar_acentos(str(municipio_bruto or "")).upper().strip()
    m = re.search(r"[\s\-/(]+([A-Z]{2})\)?\s*$", cidade)
    if m and m.group(1) in UFS_BR and len(cidade) > 4:
        return m.group(1)
    return ""


def _normalizar_municipio_naturalidade(valor):
    """Expande abreviações comuns de naturalidade e devolve somente o município."""
    texto = _tirar_acentos(str(valor or "").strip().upper())

    # Remove sufixo de UF quando o OCR mistura a UF no campo da cidade.
    texto = re.sub(r"\s*[-/]\s*[A-Z]{2}\s*$", "", texto)
    texto = re.sub(r"\s*\(\s*[A-Z]{2}\s*\)\s*$", "", texto)
    m = re.search(r"^(.{3,}?)\s+([A-Z]{2})$", texto)
    if m and m.group(2) in UFS_BR:
        texto = m.group(1)
    texto = re.sub(r"^(S|STA|STO)\s*\.\s*", r"\1 ", texto)
    texto = re.sub(r"\s+", " ", texto).strip(" .,-")
    texto = re.sub(r"\s+", " ", texto)

    equivalencias = {
        "S PAULO": "SAO PAULO",
        "S ROQUE": "SAO ROQUE",
        "S JOSE DOS CAMPOS": "SAO JOSE DOS CAMPOS",
        "S JOSE DO RIO PRETO": "SAO JOSE DO RIO PRETO",
        "S JOSE DO RIO PARDO": "SAO JOSE DO RIO PARDO",
        "S BERNARDO DO CAMPO": "SAO BERNARDO DO CAMPO",
        "S CAETANO DO SUL": "SAO CAETANO DO SUL",
        "S CARLOS": "SAO CARLOS",
        "S VICENTE": "SAO VICENTE",
        "S SEBASTIAO": "SAO SEBASTIAO",
        "S JOAQUIM DA BARRA": "SAO JOAQUIM DA BARRA",
        "S PEDRO": "SAO PEDRO",
        "S MANUEL": "SAO MANUEL",
        "S MIGUEL ARCANJO": "SAO MIGUEL ARCANJO",
        "S SIMAO": "SAO SIMAO",
        "S BENTO DO SAPUCAI": "SAO BENTO DO SAPUCAI",
        "S JOAO DA BOA VISTA": "SAO JOAO DA BOA VISTA",
        "S JOSE": "SAO JOSE",
    }

    if texto in equivalencias:
        return equivalencias[texto]

    # Regras gerais para abreviações no início do nome do município.
    texto = re.sub(r"^S\s+(?=[A-Z])", "SAO ", texto)
    texto = re.sub(r"^STA\.?\s+(?=[A-Z])", "SANTA ", texto)
    texto = re.sub(r"^STO\.?\s+(?=[A-Z])", "SANTO ", texto)
    return texto

# ============================================================
# MUNICÍPIOS OFICIAIS (IBGE) — municipios.json, o mesmo usado pelo site
# ============================================================
_MUNICIPIOS = None


def _chave_cidade(nome):
    """Forma de comparação: sem acento, maiúscula, sem apóstrofo/hífen."""
    t = _tirar_acentos(str(nome or "")).upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _municipios():
    """{UF: {chave: NOME OFICIAL}} carregado uma vez."""
    global _MUNICIPIOS
    if _MUNICIPIOS is None:
        try:
            caminho = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "municipios.json"
            )
            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)
            _MUNICIPIOS = {
                uf: {_chave_cidade(n): n for n in lista}
                for uf, lista in dados.items() if uf in UFS_BR
            }
        except Exception:
            _MUNICIPIOS = {}
    return _MUNICIPIOS


def _municipio_oficial(nome, uf):
    """
    Confere a cidade lida com a lista oficial do estado.
    Devolve (nome_oficial, uf, certeza):
    - achou igual: nome oficial, certeza;
    - parecido (erro de leitura/abreviação): corrige, mas sem certeza;
    - sem UF: procura no Brasil todo; se a cidade só existe em um estado,
      usa esse estado, sem certeza;
    - não achou: mantém o lido, sem certeza.
    """
    base = _municipios()
    if not base or not nome:
        return nome, uf, True
    chave = _chave_cidade(nome)

    if uf in base:
        cidades = base[uf]
        if chave in cidades:
            return cidades[chave], uf, True
        import difflib
        notas = sorted(
            ((difflib.SequenceMatcher(None, chave, c).ratio(), c) for c in cidades),
            reverse=True,
        )
        if notas and notas[0][0] >= 0.85 and (
            len(notas) == 1 or notas[0][0] - notas[1][0] >= 0.03
        ):
            return cidades[notas[0][1]], uf, False
        return nome, uf, False

    achados = [(u, c[chave]) for u, c in base.items() if chave in c]
    if len(achados) == 1:
        return achados[0][1], achados[0][0], False
    return nome, uf, False


# Opções do campo "Órgão Emissor" do SICAQ para RG/CIN (texto exato da lista).
ORGAO_SSP = "SECRETARIA DE SEGURANÇA PÚBLICA"
ORGAO_PC = "POLÍCIA CIVIL"
ORGAO_SDS = "SECRETARIA DEFESA SOCIAL"
ORGAO_CASA_CIVIL = "SECRETARIA DE ESTADO DA CASA CIVIL"
ORGAO_OUTRO = "OUTRO TIPO DE ÓRGÃO EMISSOR"


def _orgao_sicaq(orgao_bruto):
    """
    Converte o órgão expedidor lido no RG/CIN para a opção exata do SICAQ.
    Devolve (opção, certeza). Sem certeza, o site mostra o campo em amarelo.
    """
    t = _tirar_acentos(str(orgao_bruto or "")).upper()
    t = re.sub(r"[^A-Z0-9 /\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Tira a UF do final: SSP/SP, SSP-SP, SSP SP, PCMG -> PC.
    sem_uf = re.sub(r"[\s/\-]+(" + "|".join(sorted(UFS_BR)) + r")$", "", t).strip()
    s = sem_uf.replace(" ", "")
    if not s:
        return "", False

    def tem(*termos):
        return any(x in t for x in termos)

    # Casa Civil
    if tem("CASA CIVIL"):
        return ORGAO_CASA_CIVIL, True
    if s in ("SCC", "SECC", "SECCIVIL", "SECASACIVIL"):
        return ORGAO_CASA_CIVIL, False
    # Defesa Social (PE: SDS; MG: SEDS). SSPDS (CE) tem as duas palavras.
    if s.startswith("SSPDS") or tem("SEGURANCA PUBLICA E DEFESA SOCIAL"):
        return ORGAO_SSP, False
    if s in ("SDS", "SEDS") or tem("DEFESA SOCIAL"):
        return ORGAO_SDS, True
    # Siglas combinadas, ex.: IIRGD.SSP.SP, II-SSP, SDS PE.
    siglas = set(re.split(r"[\s/\-]+", sem_uf))
    if siglas & {"SDS", "SEDS"}:
        return ORGAO_SDS, True
    if siglas & {"SSP", "SESP", "IIRGD", "IRGD", "SEJUSP", "SESEG", "SESDEC", "SSPTC"}:
        return ORGAO_SSP, True
    # Segurança Pública (inclui IIRGD/IRGD de SP e Instituto de Identificação).
    if (
        s.startswith(("SSP", "SESP", "SESEG", "SEJUSP", "SESDEC", "SSPTC"))
        or s in ("IIRGD", "IRGD", "II", "IIPR", "IISP")
        or tem("SEGURANCA PUBLICA", "INSTITUTO DE IDENTIFICACAO", "RICARDO GUMBLETON")
    ):
        return ORGAO_SSP, True
    # Polícia Civil (PC, PCMG, PC/GO, PCIVIL, DGPC...).
    if (
        re.fullmatch(r"PC([A-Z]{2})?", s)
        or s.startswith(("PCIVIL", "DGPC", "DPC"))
        or tem("POLICIA CIVIL")
    ):
        return ORGAO_PC, True
    if s in ("IFP", "DIC", "IPF"):
        return ORGAO_PC, False
    # Outros emissores (trânsito, perícia, forças armadas, conselhos, PF).
    if s.startswith("DETRAN") or tem("TRANSITO"):
        return ORGAO_OUTRO, True
    if s.startswith(("MEX", "EB", "MAER", "COMAER", "MMA", "MD", "DPF", "PF", "OAB", "CR", "CO", "MTE")) or tem(
        "EXERCITO", "MARINHA", "AERONAUTICA", "POLICIA FEDERAL", "CONSELHO", "ORDEM DOS ADVOGADOS"
    ):
        return ORGAO_OUTRO, True
    if s.startswith(("IGP", "ITEP", "IPC")):
        return ORGAO_OUTRO, False
    # Sigla desconhecida: a opção mais comum, marcada para conferir.
    return ORGAO_SSP, False


def _normalizar_saida(d):
    d = _corrigir_datas_cnh(d)
    d = dict(d or {})
    tipo = str(d.get("tipo_documento") or "").strip().upper()
    nacionalidade = str(d.get("nacionalidade") or "").strip().upper()
    # CIN imprime "BRA"; documentos bilíngues podem trazer BRAZIL/BRAZILIAN.
    if nacionalidade.startswith("BRASILEIR") or nacionalidade in (
        "BR", "BRA", "BRASIL", "BRAZIL", "BRAZILIAN"
    ):
        nacionalidade = "BRASILEIRA"

    nat_mun = _normalizar_municipio_naturalidade(
        d.get("naturalidade_municipio") or ""
    )
    nat_uf = _uf_naturalidade(
        d.get("naturalidade_uf"),
        d.get("naturalidade_municipio"),
    )

    sexo = _tirar_acentos(str(d.get("sexo") or "").strip().upper())
    if sexo in ("M", "MASC", "MASCULINO"):
        sexo = "MASCULINO"
    elif sexo in ("F", "FEM", "FEMININO"):
        sexo = "FEMININO"

    # Exibição no site: CIDADE/UF (ex.: SOROCABA/SP).
    # O site separa pela barra e envia à extensão o município (sem UF)
    # e a UF em campos próprios, como o SICAQ exige.
    incertos = list(d.get("campos_incertos") or [])

    # Confere a cidade com a lista oficial do IBGE (a mesma da caixa do site).
    if nat_mun:
        uf_lida = nat_uf
        nat_mun, nat_uf, certeza = _municipio_oficial(nat_mun, nat_uf)
        nat_uf = nat_uf or ""
        if not certeza and "naturalidade_municipio" not in incertos:
            incertos.append("naturalidade_municipio")
        if nat_uf and not uf_lida and "naturalidade_uf" not in incertos:
            incertos.append("naturalidade_uf")  # UF deduzida pela cidade

    if nat_mun and nat_uf:
        naturalidade = f"{nat_mun}/{nat_uf}"
    else:
        naturalidade = nat_mun
        if nat_mun and "naturalidade_uf" not in incertos:
            incertos.append("naturalidade_uf")

    # CPF lido que não fecha os dígitos verificadores: provável erro de leitura.
    cpf_lido = re.sub(r"\D", "", str(d.get("cpf") or ""))
    if cpf_lido and not _cpf_valido(cpf_lido) and "cpf" not in incertos:
        incertos.append("cpf")

    # Órgão emissor do RG/CIN já no texto exato da opção do SICAQ.
    orgao_sicaq = ""
    if tipo in ("RG", "CIN"):
        orgao_sicaq, certeza = _orgao_sicaq(d.get("orgao_emissor"))
        if not certeza and "orgao_emissor" not in incertos:
            incertos.append("orgao_emissor")

    return {
        "documento": tipo,
        "nome": _tirar_acentos(str(d.get("nome") or "").strip().upper()),
        "cpf": "".join(c for c in str(d.get("cpf") or "") if c.isdigit()),
        "rg": "".join(c for c in str(d.get("rg") or "").upper() if c.isalnum()),
        "numero_cnh": "".join(c for c in str(d.get("numero_cnh") or "") if c.isdigit()),
        "orgao_emissor": _tirar_acentos(str(d.get("orgao_emissor") or "").strip().upper()),
        "orgao_sicaq": orgao_sicaq,
        "uf": _tirar_acentos(str(d.get("uf_emissao") or "").strip().upper()),
        "nascimento": _formatar_data_br(d.get("data_nascimento")),
        "primeira": _formatar_data_br(d.get("data_primeira_habilitacao")),
        "emissao": _formatar_data_br(d.get("data_emissao")),
        "validade": _formatar_data_br(d.get("validade")),
        "sexo": sexo,
        "nacionalidade": nacionalidade or "BRASILEIRA",
        "naturalidade": naturalidade,
        "naturalidade_uf": nat_uf,
        "filiacao1": _tirar_acentos(str(d.get("filiacao_1") or "").strip().upper()),
        "filiacao2": _tirar_acentos(str(d.get("filiacao_2") or "").strip().upper()),
        "modelo_documento": str(d.get("modelo_documento") or "").strip().upper(),
        "campos_incertos": incertos,
    }



# ============================================================
# HOLERITE / RENDA FORMAL
# ============================================================

def _sem_acentos(txt):
    txt = str(txt or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", txt)
        if unicodedata.category(c) != "Mn"
    )


def _holerite_competencias_validas():
    """Mês atual + os 2 meses anteriores, no fuso de São Paulo."""
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    competencias = []

    ano = agora.year
    mes = agora.month

    for deslocamento in range(3):
        m = mes - deslocamento
        a = ano
        while m <= 0:
            m += 12
            a -= 1
        competencias.append(f"{m:02d}/{a:04d}")

    return competencias


def _holerite_prompt():
    atual, anterior, retrasado = _holerite_competencias_validas()
    return f"""
Leia este HOLERITE / CONTRACHEQUE brasileiro e devolva SOMENTE JSON válido.

Chaves obrigatórias:
cnpj, data_admissao, competencia, bruta, liquido_original,
adiantamento_salarial_total, descricao_adiantamento, irrf, campos_incertos.

Regras:
- cnpj: CNPJ do empregador, somente dígitos.
- data_admissao: DD/MM/AAAA. Não confunda com data de pagamento.
- competencia: mês/ano do holerite em MM/AAAA.
- bruta: TOTAL DE PROVENTOS / TOTAL BRUTO. Não use salário base, base INSS,
  base FGTS ou líquido.
- liquido_original: somente LÍQUIDO A RECEBER / LÍQUIDO DO HOLERITE.
- adiantamento_salarial_total: some SOMENTE antecipação do salário mensal.
  Exemplos válidos: ADIANTAMENTO DE SALÁRIO, ADIANTAMENTO SALARIAL,
  ADIANT. SALÁRIO, ADIANTAMENTO QUINZENAL, VALE SALÁRIO, ANTECIPAÇÃO SALARIAL.
- NÃO trate como adiantamento salarial: empréstimo/consignado, férias,
  adiantamento de férias, 13º, vale transporte, vale refeição, vale alimentação,
  INSS, IRRF, pensão ou contribuições.
- descricao_adiantamento: copie a descrição usada para formar o adiantamento;
  se não houver, deixe vazio.
- irrf: valor DESCONTADO de IRRF / I.R.R.F. / IMPOSTO DE RENDA RETIDO.
  Se não houver, devolva "0,00".
- Valores monetários no formato brasileiro, ex.: "2304,21".
- Não invente. Campo ilegível = vazio e inclua a chave em campos_incertos.

REGRA DE PERÍODO DO SICAQ:
- As únicas competências aceitas hoje são {atual}, {anterior} e {retrasado}.
- Se houver mais de um holerite/página, examine TODAS as páginas.
- Ignore holerites fora dessas três competências.
- Se encontrar {atual}, use {atual}.
- Se não houver {atual}, mas houver {anterior}, use {anterior}.
- Se não houver {atual} nem {anterior}, mas houver {retrasado}, use {retrasado}.
- Nunca misture valores de meses diferentes.
- CNPJ, admissão, bruto, líquido, adiantamento e IRRF devem vir da MESMA
  página/competência escolhida.
- Se nenhuma página estiver dentro do período aceito, devolva competencia vazia
  e inclua "competencia" em campos_incertos.
""".strip()


def _holerite_competencia_mm_aaaa(valor):
    valor = _sem_acentos(valor).upper().strip()
    if not valor:
        return ""

    m = re.search(r"\b(0?[1-9]|1[0-2])\s*[/.\-]\s*((?:19|20)\d{2})\b", valor)
    if m:
        return f"{int(m.group(1)):02d}/{m.group(2)}"

    meses = {
        "JANEIRO": 1, "FEVEREIRO": 2, "MARCO": 3, "ABRIL": 4,
        "MAIO": 5, "JUNHO": 6, "JULHO": 7, "AGOSTO": 8,
        "SETEMBRO": 9, "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12,
    }
    ano = re.search(r"\b((?:19|20)\d{2})\b", valor)
    if ano:
        for nome, numero in meses.items():
            if nome in valor:
                return f"{numero:02d}/{ano.group(1)}"
    return valor


def _holerite_numero(valor):
    if valor is None:
        return 0.0
    s = str(valor).strip().upper().replace("R$", "").replace("\xa0", " ")
    s = re.sub(r"[^0-9,.\-]", "", s)
    if not s:
        return 0.0
    try:
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(".") > 1:
            partes = s.split(".")
            s = "".join(partes[:-1]) + "." + partes[-1]
        return float(s)
    except Exception:
        return 0.0


def _holerite_valor(valor):
    return f"{float(valor or 0):.2f}".replace(".", ",")


def _compactar_texto_holerite(texto, limite=5000):
    """Remove linhas vazias/espaços repetidos e limita o tamanho (início + fim),
    para economizar tokens no limite gratuito da Groq."""
    linhas = [re.sub(r"[ \t]+", " ", l).strip() for l in str(texto or "").splitlines()]
    texto = "\n".join(l for l in linhas if l)
    if len(texto) <= limite:
        return texto
    inicio = int(limite * 0.55)
    fim = limite - inicio
    return texto[:inicio] + "\n[...]\n" + texto[-fim:]


def _pdf_texto_holerite(data):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            partes = []
            for i in range(min(len(doc), 8)):
                texto = doc[i].get_text("text") or ""
                if texto.strip():
                    partes.append(f"\n--- PÁGINA {i+1} ---\n{texto}")
            return "".join(partes).strip()
        finally:
            doc.close()
    except Exception:
        return ""


def _imagem_holerite(data, filename, mimetype):
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if not is_pdf:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        max_w = 1500
        if img.width > max_w:
            escala = max_w / img.width
            img = img.resize((max_w, max(1, int(img.height * escala))))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        paginas = []
        for i in range(min(len(doc), 3)):
            pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            if img.width > 1350:
                escala = 1350 / img.width
                img = img.resize((1350, max(1, int(img.height * escala))))
            paginas.append(img.copy())

        if not paginas:
            raise RuntimeError("O PDF do holerite não possui páginas válidas.")

        margem = 16
        largura = max(p.width for p in paginas)
        altura = margem + sum(p.height + margem for p in paginas)
        composta = Image.new("RGB", (largura + 2*margem, altura), "white")
        y = margem
        for pagina in paginas:
            x = margem + (largura - pagina.width)//2
            composta.paste(pagina, (x, y))
            y += pagina.height + margem

        out = io.BytesIO()
        composta.save(out, format="JPEG", quality=87, optimize=True)
        return out.getvalue()
    finally:
        doc.close()


def _groq_holerite_call(arquivos):
    """
    Lê 1 ou 2 arquivos de holerite na mesma chamada.
    Se houver dois meses diferentes, o prompt escolhe a competência
    válida mais recente. Nunca soma valores de meses distintos.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    conteudo = [
        {"type": "text", "text": _holerite_prompt()}
    ]

    for idx, item in enumerate(arquivos[:2], start=1):
        data = item["data"]
        filename = item["filename"]
        mimetype = item["mimetype"]

        texto_pdf = ""
        if PathLikeSuffix(filename) == ".pdf" or mimetype == "application/pdf":
            texto_pdf = _pdf_texto_holerite(data)

        texto_pdf = _compactar_texto_holerite(texto_pdf, 20000)
        if len(texto_pdf) >= 120:
            # Limita cada arquivo para não desperdiçar tokens no plano gratuito.
            conteudo.append({
                "type": "text",
                "text": (
                    f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====\n"
                    + _compactar_texto_holerite(texto_pdf)
                ),
            })
        else:
            imagem = _imagem_holerite(data, filename, mimetype)
            b64 = base64.b64encode(imagem).decode("ascii")
            conteudo.append({
                "type": "text",
                "text": f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====",
            })
            conteudo.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
    ]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]

    ultimo = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": conteudo}],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_completion_tokens": 650,
            "stream": False,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "Assistente-SICAQ-Holerite-Web/1.1",
            },
            method="POST",
        )

        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            content = body["choices"][0]["message"]["content"]
            dados = json.loads(content)
            return dados, model, elapsed

        except urllib.error.HTTPError as e:
            detalhe = e.read().decode("utf-8", errors="replace")

            if e.code == 429:
                espera = ""
                m = re.search(r"try again in\s+([0-9.]+)s", detalhe, re.I)
                if m:
                    espera = f" Aguarde cerca de {float(m.group(1)):.0f} segundos e tente novamente."
                ultimo = GroqLimite(
                    "Limite gratuito do Groq atingido temporariamente." + espera
                )
                break

            ultimo = RuntimeError(f"Groq HTTP {e.code}: {detalhe}")
            if e.code not in (400, 404):
                break

        except Exception as e:
            ultimo = e
            break

    if isinstance(ultimo, GroqLimite):
        raise ultimo
    raise RuntimeError(str(ultimo or "Falha na leitura do holerite."))


def _normalizar_holerite(resultado):
    resultado = dict(resultado or {})
    cnpj = re.sub(r"\D", "", str(resultado.get("cnpj") or ""))
    competencia = _holerite_competencia_mm_aaaa(resultado.get("competencia"))
    validas = _holerite_competencias_validas()

    if competencia not in validas:
        identificada = competencia or "não identificada"
        raise ValueError(
            f"HOLERITE_FORA_PERIODO|Competência identificada: {identificada}. "
            f"Período aceito: {validas[0]}, {validas[1]} e {validas[2]}."
        )

    bruta = _holerite_numero(resultado.get("bruta"))
    liquido_original = _holerite_numero(resultado.get("liquido_original"))
    adiantamento = _holerite_numero(resultado.get("adiantamento_salarial_total"))
    irrf = _holerite_numero(resultado.get("irrf"))

    # Regra SICAQ: somente adiantamento salarial volta ao líquido.
    liquida_sicaq = liquido_original + adiantamento

    return {
        "caracteristica_renda": "COMPROVADA",
        "tipo_fonte": "JURIDICA",
        "documento": "CONTRACHEQUE/HOLLERITH",
        "cnpj": cnpj,
        "data_admissao": str(resultado.get("data_admissao") or "").strip(),
        "competencia": competencia,
        "bruta": _holerite_valor(bruta),
        "liquida": _holerite_valor(liquida_sicaq),
        "irrf": _holerite_valor(irrf),
        "liquido_original": _holerite_valor(liquido_original),
        "adiantamento": _holerite_valor(adiantamento),
        "descricao_adiantamento": str(resultado.get("descricao_adiantamento") or "").strip().upper(),
        "campos_incertos": list(resultado.get("campos_incertos") or []),
    }


@app.post("/ler-holerite")
def ler_holerite():
    recebidos = request.files.getlist("files")

    # Compatibilidade com a versão anterior.
    if not recebidos and "file" in request.files:
        recebidos = [request.files["file"]]

    recebidos = [f for f in recebidos if f and (f.filename or "").strip()]

    if not recebidos:
        return jsonify({"ok": False, "error": "Holerite não enviado."}), 400

    if len(recebidos) > 2:
        return jsonify({
            "ok": False,
            "error": "Selecione no máximo 2 arquivos de holerite."
        }), 400

    arquivos = []
    for f in recebidos:
        data = f.read()
        if not data:
            continue
        arquivos.append({
            "data": data,
            "filename": f.filename or "",
            "mimetype": f.mimetype or "",
        })

    if not arquivos:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400

    inicio = time.perf_counter()

    try:
        resultado, modelo, tempo_api = _groq_holerite_call(arquivos)
        dados = _normalizar_holerite(resultado)

        return jsonify({
            "ok": True,
            "dados": dados,
            "modelo": modelo,
            "arquivos_lidos": len(arquivos),
            "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - inicio, 3),
            "competencias_validas": _holerite_competencias_validas(),
        })

    except ValueError as e:
        txt = str(e)
        if txt.startswith("HOLERITE_FORA_PERIODO|"):
            return jsonify({
                "ok": False,
                "code": "HOLERITE_FORA_PERIODO",
                "error": txt.split("|", 1)[1],
                "competencias_validas": _holerite_competencias_validas(),
            }), 422
        return jsonify({"ok": False, "error": txt}), 400

    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Assistente SICAQ Web",
        "groq_configurada": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "holerite_competencias": _holerite_competencias_validas(),
    })


@app.post("/ler")
def ler():
    # Novo formato: "files" pode conter 1 ou 2 arquivos.
    recebidos = request.files.getlist("files")

    # Compatibilidade com a versão anterior do site.
    if not recebidos and "file" in request.files:
        recebidos = [request.files["file"]]

    recebidos = [f for f in recebidos if f and (f.filename or "").strip()]

    if not recebidos:
        return jsonify({"ok": False, "error": "Arquivo não enviado."}), 400

    if len(recebidos) > 2:
        return jsonify({
            "ok": False,
            "error": "Selecione no máximo 2 arquivos do mesmo documento."
        }), 400

    arquivos = []
    for f in recebidos:
        data = f.read()
        if not data:
            continue
        arquivos.append({
            "data": data,
            "filename": f.filename or "",
            "mimetype": f.mimetype or "",
        })

    if not arquivos:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400

    start = time.perf_counter()
    try:
        t_preparo = time.perf_counter()
        image_bytes = _combinar_arquivos_identificacao(arquivos)
        tempo_preparo = time.perf_counter() - t_preparo

        dados_raw, modelo, tempo_api = _groq_call(image_bytes)

        # Tipo do documento sempre como CNH, RG ou CIN (ou vazio se não der).
        tipo_lido = str(dados_raw.get("tipo_documento") or "")
        tipo, deduzido = _tipo_documento(dados_raw)
        dados_raw["tipo_documento"] = tipo
        if deduzido or not tipo:
            dados_raw["campos_incertos"] = list(dados_raw.get("campos_incertos") or []) + ["tipo_documento"]
        # Só o rótulo do tipo vai para o log (nenhum dado pessoal).
        print(f"[LER] tipo lido={tipo_lido[:40]!r} -> {tipo or 'NAO IDENTIFICADO'}", flush=True)

        # Confere nascimento/validade pela MRZ (rodapé '<<<'), quando houver.
        dados_raw = _aplicar_mrz(dados_raw)

        # CNH: uma única chamada ao Groq para não estourar o limite gratuito.
        # A conferência final é feita localmente, sem consumir tokens.
        if str(dados_raw.get("tipo_documento") or "").strip().upper() == "CNH":
            dados_raw = _validar_cnh_final(dados_raw)

        dados = _normalizar_saida(dados_raw)

        return jsonify({
            "ok": True,
            "dados": dados,
            "modelo": modelo,
            "arquivos_lidos": len(arquivos),
            "tempo_preparo": round(tempo_preparo, 3),
            "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - start, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
