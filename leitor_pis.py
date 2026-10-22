"""PIS/PASEP/NIS e a rota /ler-pis."""

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



from comum import (  # noqa: F401
    GroqLimite,
    _groq_json_imagem,
    _tirar_acentos,
    app,
)

from contas import (  # noqa: F401
    _registrar_uso,
)

from imagem import (  # noqa: F401
    PathLikeSuffix,
    _combinar_arquivos_identificacao,
)

from pasta import (  # noqa: F401
    _classes_pdf,
)

from leitor_renda import (  # noqa: F401
    _pdf_texto_holerite,
)



# ============================================================
# PIS / PASEP / NIS / NIT (extrato do FGTS, Carteira de Trabalho Digital,
# Cartão Cidadão ou qualquer documento com o número escrito)
# ============================================================

PROMPT_PIS = """
Encontre no documento o número do PIS/PASEP (também chamado NIS ou NIT).
Ele tem 11 dígitos e costuma aparecer como 123.45678.90-1, perto de
"PIS", "PASEP", "PIS/PASEP", "NIS" ou "NIT".
Não confunda com CPF, CTPS, título de eleitor, conta, matrícula ou CNPJ.
Responda SOMENTE JSON: {"pis": "somente dígitos ou vazio", "nome": "nome do titular, se aparecer"}.
""".strip()



def _pis_valido(pis):
    """Confere o dígito verificador do PIS/PASEP/NIS (pesos 3,2,9,8,7,6,5,4,3,2)."""
    p = re.sub(r"\D", "", str(pis or ""))
    if len(p) != 11 or len(set(p)) == 1:
        return False
    pesos = (3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    resto = 11 - sum(int(p[i]) * pesos[i] for i in range(10)) % 11
    dv = 0 if resto in (10, 11) else resto
    return dv == int(p[10])



def _pis_do_texto(texto):
    """
    Acha o PIS no texto de um PDF digital (ex.: extrato do FGTS) sem usar a IA.
    Só aceita número válido perto das palavras PIS/PASEP/NIS/NIT.
    """
    t = _tirar_acentos(str(texto or "")).upper()
    numero = r"(\d{3}[.\s]?\d{5}[.\s]?\d{2}[\-.\s]?\d)"
    achados = []
    for m in re.finditer(r"\b(PIS|PASEP|NIS|NIT)\b", t):
        trecho = t[m.end(): m.end() + 80]
        n = re.search(numero, trecho)
        if n:
            digitos = re.sub(r"\D", "", n.group(1))
            if _pis_valido(digitos) and digitos not in achados:
                achados.append(digitos)
    # Um único número válido: certeza. Vários diferentes: deixa para a IA.
    return achados[0] if len(achados) == 1 else ""



@app.post("/ler-pis")
def ler_pis():
    arquivo = request.files.get("file") or (request.files.getlist("files") or [None])[0]
    if not arquivo or not (arquivo.filename or "").strip():
        return jsonify({"ok": False, "error": "Arquivo do PIS não enviado."}), 400
    data = arquivo.read()
    if not data:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400
    nome_arq = arquivo.filename or ""
    mime = arquivo.mimetype or ""
    inicio = time.perf_counter()

    try:
        # 1) PDF digital (ex.: extrato do FGTS): lê o texto, sem gastar IA.
        if PathLikeSuffix(nome_arq) == ".pdf" or mime == "application/pdf":
            texto = _pdf_texto_holerite(data)
            pis = _pis_do_texto(texto)
            if pis:
                print("[PIS] encontrado no texto do PDF (sem IA)", flush=True)
                _registrar_uso("pis")
                return jsonify({
                    "ok": True, "pis": pis, "valido": True, "nome": "",
                    "fonte": "texto", "tempo_total": round(time.perf_counter() - inicio, 3),
                })

            # PDF digital tipo "pasta": o PIS pode estar depois das 8
            # primeiras páginas, e a IA só veria as páginas 1 e 2.
            info = _classes_pdf({"data": data, "filename": nome_arq, "mimetype": mime})
            if info and info[0] > 8 and any(info[1][:2]):
                print("[PASTA] PIS não achado nas 8 primeiras páginas", flush=True)
                return jsonify({
                    "ok": False, "code": "PASTA",
                    "error": (f"Não achei o PIS nas 8 primeiras páginas desse PDF ({info[0]} páginas, "
                              "parece uma pasta). Mande só o extrato do FGTS ou a Carteira de Trabalho, "
                              "ou digite o número."),
                }), 422

        # 2) Foto/print ou PDF escaneado: leitura curta da IA.
        imagem = _combinar_arquivos_identificacao(
            [{"data": data, "filename": nome_arq, "mimetype": mime}]
        )
        resultado, tempo_api = _groq_json_imagem(PROMPT_PIS, imagem)
        pis = re.sub(r"\D", "", str(resultado.get("pis") or ""))
        nome = _tirar_acentos(str(resultado.get("nome") or "").strip().upper())
        print(f"[PIS] lido pela IA; valido={_pis_valido(pis)}", flush=True)
        _registrar_uso("pis")
        return jsonify({
            "ok": True, "pis": pis, "valido": _pis_valido(pis), "nome": nome,
            "fonte": "ia", "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - inicio, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
