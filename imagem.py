"""Preparo das imagens para a IA (recorte de foto, PDF, rotação, páginas)."""

import base64
import hashlib
import hmac
import io
import secrets
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
from flask import Flask, g, jsonify, request
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



def _recortar_documento(img: Image.Image, frac_min: float = 0.0):
    """
    Recorta e endireita o documento quando ele está pequeno/inclinado na foto.
    Se não tiver certeza, devolve a imagem original.
    frac_min: só recorta se o achado ocupar ao menos essa fração da foto
    (certidão: evita recortar um quadro de dentro da folha).
    """
    if cv2 is None or np is None:
        return img, "sem_opencv"
    try:
        achado = _localizar_documento(img)
        if not achado:
            return img, "sem_recorte"
        quad, frac = achado
        if frac < frac_min:
            return img, "sem_recorte"
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



def _texto_de_lado(img):
    """
    O texto do documento está de lado (girado 90°)? Olha as letras: em cada
    linha de texto, a letra vizinha mais próxima fica do lado (horizontal);
    com o documento de lado, fica em cima/embaixo. Devolve True, False ou
    None (poucas letras para decidir).
    """
    if cv2 is None or np is None:
        return None
    try:
        W, H = img.size
        e = min(1.0, 1400 / max(W, H))
        peq = img if e >= 1.0 else img.resize((max(1, int(W * e)), max(1, int(H * e))), Image.Resampling.BILINEAR)
        g = np.asarray(peq.convert("L"))
        tinta = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)
        n, _, st, cen = cv2.connectedComponentsWithStats(tinta, connectivity=8)
        w, h, a = st[1:, 2], st[1:, 3], st[1:, 4]
        maior, menor = np.maximum(w, h), np.maximum(1, np.minimum(w, h))
        letra = (a >= 15) & (maior >= 5) & (maior <= 45) & (maior <= 5 * menor) & (a <= 0.9 * w * h)
        idx = np.where(letra)[0] + 1
        if len(idx) < 30:
            return None
        if len(idx) > 2500:
            idx = idx[np.linspace(0, len(idx) - 1, 2500).astype(int)]
        pts = cen[idx]
        tam = np.maximum(st[idx, 2], st[idx, 3]).astype(float)
        hor = ver = 0
        for i in range(len(pts)):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            d[i] = 1e9
            ok = (tam > 0.6 * tam[i]) & (tam < 1.7 * tam[i]) & (d < 2.2 * tam[i])
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, d, 1e9)))
            dx, dy = abs(pts[j, 0] - pts[i, 0]), abs(pts[j, 1] - pts[i, 1])
            if dx > 2.5 * dy:
                hor += 1
            elif dy > 2.5 * dx:
                ver += 1
        if ver >= 25 and ver > 1.8 * hor:
            return True
        if hor >= 25 and hor > 1.8 * ver:
            return False
        return None
    except Exception:
        return None



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



PAINEIS_PDF_DIGITAL = True



def _pdf_paineis(page):
    """
    PDF digital do gov.br (CNH/CIN): o cartão vem como 2-4 imagens (painéis)
    pequenas numa página A4. Devolve os painéis (de cima para baixo) na
    resolução original, ou None quando a página não é desse tipo.
    """
    pagina = page.rect
    paineis = []
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
        paineis.append((caixa, larg_px))
    if not (2 <= len(paineis) <= 4):
        return None

    # Dado do cliente escrito como texto FORA dos painéis: não arrisca.
    for bloco in page.get_text("blocks"):
        caixa_txt = pymupdf.Rect(bloco[:4])
        if any(caixa_txt.intersects(c) for c, _ in paineis):
            continue
        if re.search(r"\d{2}/\d{2}/\d{4}|\d{3}\.?\d{3}\.?\d{3}-?\d{2}", str(bloco[4] or "")):
            return None

    paineis.sort(key=lambda p: (round(p[0].y0), p[0].x0))
    imgs = []
    for caixa, larg_px in paineis:
        zoom = max(1.0, min(5.0, larg_px / caixa.width))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=caixa, alpha=False)
        imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return imgs



def _pdf_paineis_lado_a_lado(page):
    """
    Monta uma faixa com os painéis do PDF digital LADO A LADO, na resolução
    original, sem cabeçalho/QR code: as letras chegam à IA com cerca do
    dobro do tamanho. Qualquer dúvida -> None (usa a página inteira).
    """
    try:
        imgs = _pdf_paineis(page)
        if not imgs:
            return None

        altura = max(i.height for i in imgs)
        imgs = [
            i if i.height == altura else
            i.resize((max(1, int(i.width * altura / i.height)), altura), Image.Resampling.LANCZOS)
            for i in imgs
        ]
        gap = 16
        largura = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
        faixa = Image.new("RGB", (largura, altura), "white")
        x = 0
        for i in imgs:
            faixa.paste(i, (x, 0))
            x += i.width + gap

        # Painéis pequenos (CIN 650x410): amplia até ~1,6 MP, o tamanho que a IA usa.
        fator = min(2.0, (1_600_000 / (largura * altura)) ** 0.5)
        if fator > 1.05:
            faixa = faixa.resize(
                (int(largura * fator), int(altura * fator)), Image.Resampling.LANCZOS
            )
        print(f"[LER] PDF digital: {len(imgs)} painéis lado a lado {faixa.size}", flush=True)
        return faixa
    except Exception as e:
        print(f"[LER] painéis do PDF não montados ({type(e).__name__}); usando a página", flush=True)
        return None



# Recorte do PDF digital (só os painéis do cartão) DESLIGADO: no teste real,
# a imagem alta e estreita fez a IA errar uma letra da filiação, enquanto a
# página inteira lia certo. Mantido o envio da página inteira, já validado.
RECORTE_PDF_DIGITAL = False



def _pdf_pagina_para_imagem(page):
    """Renderiza a página. Devolve (imagem, pode_recortar_como_foto)."""
    if PAINEIS_PDF_DIGITAL:
        faixa = _pdf_paineis_lado_a_lado(page)
        if faixa is not None:
            return faixa, False

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

    zoom_pagina = 2.2
    try:
        area_pag0 = page.rect.width * page.rect.height
        if any((pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area_pag0
               for i in page.get_image_info()):
            zoom_pagina = 3.0   # scan/foto: aproveita a resolução original
    except Exception:
        pass
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom_pagina, zoom_pagina), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").copy()

    # PDF de foto/scan (uma imagem ocupando a página): trata como foto e tenta
    # recortar o documento. Vale mesmo com camada de texto escondida (OCR de
    # aplicativos de digitalização). PDFs digitais (CNH/CIN do gov.br) têm
    # várias imagens pequenas e não entram aqui.
    pode_recortar = False
    try:
        area_pag = page.rect.width * page.rect.height
        grandes = [
            i for i in page.get_image_info()
            if (pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area_pag
        ]
        pode_recortar = len(grandes) >= 1
    except Exception:
        pode_recortar = False
    return img, pode_recortar



def _pdf_so_uma_imagem(page):
    """Página com exatamente 1 imagem e sem texto (documento "impresso" como imagem)."""
    try:
        return len(page.get_image_info()) == 1 and not (page.get_text("text") or "").strip()
    except Exception:
        return False



def _fundo_branco(img, minimo=0.6):
    """Imagem de documento impresso/digital (fundo quase todo branco), não foto."""
    try:
        cinza = np.asarray(img.convert("L").resize((300, max(1, int(300 * img.height / img.width)))))
        return float((cinza > 225).mean()) >= minimo
    except Exception:
        return False



def _cortar_margem_branca(img, limiar=235, folga=0.015):
    """Recorta a área com conteúdo (sem as margens brancas da folha)."""
    try:
        cinza = np.asarray(img.convert("L"))
        tinta = cinza < limiar
        linhas = np.where(tinta.sum(axis=1) > max(2, 0.002 * tinta.shape[1]))[0]
        colunas = np.where(tinta.sum(axis=0) > max(2, 0.002 * tinta.shape[0]))[0]
        if len(linhas) < 10 or len(colunas) < 10:
            return img
        H, W = cinza.shape
        my, mx = int(folga * H), int(folga * W)
        caixa = (max(0, colunas[0] - mx), max(0, linhas[0] - my), min(W, colunas[-1] + mx), min(H, linhas[-1] + my))
        if (caixa[2] - caixa[0]) * (caixa[3] - caixa[1]) > 0.92 * W * H:
            return img
        return img.crop(caixa)
    except Exception:
        return img



def _to_jpeg(data: bytes, filename: str, mimetype: str) -> bytes:
    img, _modo = _preparar_imagem(data, filename, mimetype)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=91)
    return out.getvalue()



def _preparar_imagem(data: bytes, filename: str, mimetype: str):
    """
    Devolve (imagem, em_pe). em_pe=True quando o documento recortado ficou
    mais alto que largo: CNH, RG e CIN são "deitados" (paisagem), então ele
    foi fotografado de lado e precisa ser girado.

    Prioriza precisão de leitura para CNH/RG/CIN:
    - foto: corrige a rotação do celular e recorta/endireita o documento;
    - PDF digital: usa só a área do cartão (sem cabeçalho e QR code);
    - PDF escaneado: tratado como foto.
    """
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    em_pe = False
    dividir = False
    paginas_de_lado = None
    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            de_lado = []
            for i in range(min(2, len(doc))):
                img, pode_recortar = _pdf_pagina_para_imagem(doc[i])
                lado = False
                if len(doc) == 1 and _pdf_so_uma_imagem(doc[i]):
                    # PDF "impresso" com o documento como 1 imagem branca no
                    # meio da folha (ex.: CNH nova em pé). Não é foto: o
                    # recorte de foto pegaria só um quadro (a tabela de
                    # categorias). Tira a margem branca e, se ficou alta,
                    # manda em 2 metades (letras maiores).
                    cortada = _cortar_margem_branca(img)
                    if cortada.size != img.size and _fundo_branco(cortada) and cortada.height > 1.3 * cortada.width:
                        img, pode_recortar, dividir = cortada, False, True
                        print(f"[LER] documento impresso em pé (1 imagem na folha): {img.size}", flush=True)
                if pode_recortar:
                    img, info = _recortar_documento(img)
                    recortado = info.startswith("recortado")
                    # De lado? Pelas letras; sem letras suficientes, pela
                    # regra antiga (recorte mais alto que largo).
                    lado = _texto_de_lado(img)
                    if lado is None:
                        lado = recortado and img.height > 1.15 * img.width
                    if lado:
                        em_pe = True
                    elif not recortado and img.height > 1.3 * img.width:
                        # Scan de folha em pé (CamScanner, WhatsApp): o
                        # documento ocupa a folha toda e as letras ficam
                        # pequenas. Vai em 2 metades (cima/baixo).
                        dividir = True
                pages.append(img)
                de_lado.append(lado)
            if len(pages) > 1:
                em_pe = False
                dividir = False
                # Frente e verso fotografados de lado (cartão em pé na foto):
                # vão girados para a posição de leitura (ver _imagens_para_ia).
                if any(de_lado):
                    paginas_de_lado = (pages, de_lado)

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
        img, info = _recortar_documento(img)
        em_pe = _texto_de_lado(img)
        if em_pe is None:
            em_pe = info.startswith("recortado") and img.height > 1.15 * img.width

    max_side = 3200
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    modo = "em_pe" if em_pe else ("dividir" if dividir else "normal")
    if paginas_de_lado:
        modo = "paginas_de_lado"
        img.info["paginas_de_lado"] = paginas_de_lado
    return img, modo



def _jpeg_para_ia(img):
    """Mesmo tamanho/qualidade usados no envio de 1 arquivo."""
    if img.width > 2 * img.height:
        # Faixa de painéis (PDF digital): limita pela área, não pela largura.
        area_max = 2_600_000
        if img.width * img.height > area_max:
            f = (area_max / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * f), int(img.height * f)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=92)
        return out.getvalue()
    max_w = 1650
    if img.width > max_w:
        img = img.resize(
            (max_w, max(1, int(img.height * max_w / img.width))),
            Image.Resampling.LANCZOS,
        )
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=91)
    return out.getvalue()



def _imagens_para_ia(arquivos):
    """
    Lista de imagens para a IA.
    - Normal: 1 imagem (1 ou 2 arquivos combinados).
    - Documento fotografado de lado (1 arquivo): 2 imagens, giradas para um
      lado e para o outro; a IA lê a que estiver na posição certa.
      Custa 1 imagem a mais (2.048 tokens) só nesses casos.
    """
    if len(arquivos) == 1:
        a = arquivos[0]
        img, modo = _preparar_imagem(a["data"], a["filename"], a["mimetype"])
        if modo == "em_pe":
            print("[LER] documento de lado: enviando 2 orientações", flush=True)
            return [
                _jpeg_para_ia(img.rotate(-90, expand=True)),
                _jpeg_para_ia(img.rotate(90, expand=True)),
            ], NOTA_ROTACOES
        if modo == "dividir":
            w, h = img.size
            cima = img.crop((0, 0, w, int(h * 0.54)))
            baixo = img.crop((0, int(h * 0.46), w, h))
            if w < 1400:
                # Metades pequenas: amplia para as letras chegarem maiores.
                f = 1600 / w
                cima = cima.resize((1600, int(cima.height * f)), Image.Resampling.LANCZOS)
                baixo = baixo.resize((1600, int(baixo.height * f)), Image.Resampling.LANCZOS)
            print("[LER] scan de folha em pé: enviando metade de cima e de baixo", flush=True)
            return [_jpeg_para_ia(cima), _jpeg_para_ia(baixo)], NOTA_METADES
        if modo == "paginas_de_lado":
            paginas, de_lado = img.info["paginas_de_lado"]
            imagens = []
            for angulo in (-90, 90):
                giradas = [
                    p.rotate(angulo, expand=True) if lado else p
                    for p, lado in zip(paginas, de_lado)
                ]
                imagens.append(_jpeg_para_ia(_empilhar_paginas(giradas)))
            print("[LER] frente e verso de lado: enviando 2 orientações", flush=True)
            return imagens, NOTA_ROTACOES_PAGINAS
        return [_jpeg_para_ia(img)], ""

    # 2 arquivos (frente + verso): se algum estiver de lado, vão girados.
    preparados = [_preparar_imagem(a["data"], a["filename"], a["mimetype"]) for a in arquivos[:2]]
    if any(modo in ("em_pe", "paginas_de_lado") for _, modo in preparados):
        paginas, de_lado = [], []
        for img, modo in preparados:
            if modo == "paginas_de_lado":
                ps, ls = img.info["paginas_de_lado"]
                paginas += list(ps)
                de_lado += list(ls)
            else:
                paginas.append(img)
                de_lado.append(modo == "em_pe")
        imagens = []
        for angulo in (-90, 90):
            giradas = [p.rotate(angulo, expand=True) if lado else p for p, lado in zip(paginas, de_lado)]
            imagens.append(_jpeg_para_ia(_empilhar_paginas(giradas)))
        print("[LER] frente e verso (2 arquivos) de lado: enviando 2 orientações", flush=True)
        return imagens, NOTA_ROTACOES_PAGINAS
    return [_combinar_arquivos_identificacao(arquivos)], ""



def _empilhar_paginas(paginas, largura=1650, gap=18):
    """Cartões (já deitados) um embaixo do outro, com a mesma largura."""
    prontas = []
    for p in paginas:
        f = largura / p.width
        prontas.append(p.resize((largura, max(1, int(p.height * f))), Image.Resampling.LANCZOS))
    altura = sum(p.height for p in prontas) + gap * (len(prontas) - 1)
    folha = Image.new("RGB", (largura, altura), "white")
    y = 0
    for p in prontas:
        folha.paste(p, (0, y))
        y += p.height + gap
    return folha



NOTA_ROTACOES_PAGINAS = (
    "ATENÇÃO: as 2 imagens a seguir mostram o MESMO documento (frente e verso), "
    "girado em sentidos opostos: numa delas o texto fica de cabeça para baixo. "
    "Leia os dados na imagem em que o texto está na posição normal de leitura."
)



NOTA_ROTACOES = (
    "ATENÇÃO: as 2 imagens a seguir são o MESMO documento, giradas em "
    "sentidos opostos (uma delas está de cabeça para baixo). Leia os "
    "dados na imagem em que o texto está na posição normal de leitura."
)



NOTA_METADES = (
    "ATENÇÃO: as 2 imagens a seguir são a METADE DE CIMA e a METADE DE BAIXO "
    "da MESMA folha escaneada (com uma pequena faixa repetida entre elas). "
    "Leia as duas como um único documento."
)



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
