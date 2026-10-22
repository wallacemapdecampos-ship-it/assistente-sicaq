"""Aviso de PDF 'pasta' (vários documentos juntos), só pelo texto."""

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
    _tirar_acentos,
)

from imagem import (  # noqa: F401
    PathLikeSuffix,
)



# ============================================================
# AVISO DE "PASTA" (PDF com vários documentos juntos)
# A leitura continua igual (só as primeiras páginas). Aqui só se olha o
# TEXTO de cada página, sem IA, para avisar quando o PDF parece uma pasta:
# - o documento pedido está numa página que a leitura não usa -> não gasta
#   IA e pede para mandar só o documento;
# - a parte lida mistura outros documentos -> lê e avisa para conferir.
# PDF escaneado (sem texto) não dá para classificar: só avisa se tiver
# mais páginas do que a leitura usa.
# ============================================================

_SINAIS_PAGINA = [
    ("renda", (r"\bVENCIMENTOS\b|\bPROVENTOS\b", r"\bDESCONTOS\b")),
    ("renda", (r"CONTRACHEQUE|CONTRA-CHEQUE|HOLERITE|HOLERITH|RECIBO DE PAGAMENTO|DEMONSTRATIVO DE PAGAMENTO",)),
    ("renda", (r"\bINSS\b|PREVIDENCIA SOCIAL", r"BENEFICIO", r"COMPETENCIA|CREDITO|VALOR LIQUIDO|MENSALIDADE")),
    ("identidade", (r"SENATRAN|CARTEIRA NACIONAL DE HABILITACAO|PERMISSAO PARA DIRIGIR",)),
    ("identidade", (r"REGISTRO GERAL|CARTEIRA DE IDENTIDADE|IDENTIDADE NACIONAL",)),
    ("certidao", (r"CERTIDAO DE (NASCIMENTO|CASAMENTO|OBITO)|REGISTRO CIVIL DAS PESSOAS NATURAIS",)),
    ("trabalho", (r"CARTEIRA DE TRABALHO|\bFGTS\b",)),
    ("conta", (r"NOTA FISCAL|FATURA|CONTA DE (ENERGIA|LUZ|AGUA|GAS)|CODIGO DE BARRAS",)),
]



_NOME_DOC_PASTA = {
    "renda": ("o holerite / comprovante de renda", "só o holerite (ou só a página dele)"),
    "identidade": ("o documento (CNH, RG ou CIN)", "só o documento (ou só a página dele)"),
    "certidao": ("a certidão", "só a certidão (ou só a página dela)"),
    "conta": ("o comprovante de endereço", "só o comprovante (ou só a página dele)"),
}



def _classe_pagina(texto):
    t = _tirar_acentos(str(texto or "")).upper()
    if len(t.strip()) < 40:
        return ""  # sem texto (escaneada/foto)
    for classe, padroes in _SINAIS_PAGINA:
        if all(re.search(p, t) for p in padroes):
            return classe
    return "outro"



_CLASSES_CONHECIDAS = {"renda", "identidade", "certidao", "trabalho", "conta"}



def _nas_paginas(nums):
    nums = sorted(nums)
    return f"na página {nums[0]}" if len(nums) == 1 else f"nas páginas {_faixa_paginas(nums)}"



def _faixa_paginas(nums):
    nums = sorted(nums)
    if not nums:
        return ""
    grupos, ini, ant = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == ant + 1:
            ant = n
            continue
        if ini == ant:
            grupos.append(f"{ini}")
        else:
            grupos.append(f"{ini} e {ant}" if ant == ini + 1 else f"{ini} a {ant}")
        if n is not None:
            ini = ant = n
    return ", ".join(grupos)



def _classes_pdf(item, maximo=60):
    """(nº de páginas, [classe de cada página]) de um PDF; None se não for PDF."""
    if PathLikeSuffix(item.get("filename", "")) != ".pdf" and item.get("mimetype") != "application/pdf":
        return None
    try:
        doc = pymupdf.open(stream=item["data"], filetype="pdf")
    except Exception:
        return None
    try:
        classes = []
        for i in range(min(len(doc), maximo)):
            try:
                classes.append(_classe_pagina(doc[i].get_text("text") or ""))
            except Exception:
                classes.append("")
        return len(doc), classes
    finally:
        doc.close()



def _avaliar_pasta(arquivos, quer, lidas=None):
    """
    quer: 'renda', 'identidade' ou 'certidao'. lidas: quantas páginas a
    leitura usa (None = holerite: 8 com texto, 3 escaneado).
    Devolve (bloqueio, aviso): bloqueio = mensagem para NÃO ler (o documento
    está fora das páginas lidas); aviso = ler e pedir conferência.
    """
    nome, mandar = _NOME_DOC_PASTA.get(quer, ("o documento", "só o documento"))
    bloqueio, aviso = "", ""
    for item in arquivos:
        info = _classes_pdf(item)
        if not info:
            continue
        n, classes = info
        usa = lidas if lidas else (8 if any(classes[:2]) else 3)
        alvo = [i + 1 for i, c in enumerate(classes) if c == quer]
        # Outros documentos reconhecidos (CNH, conta, CTPS...) nas páginas lidas.
        outros = [i + 1 for i, c in enumerate(classes) if c in _CLASSES_CONHECIDAS and c != quer]
        misturado = any(i <= usa for i in outros)
        if n <= usa and not misturado:
            continue  # PDF comum (frente/verso, vários meses de holerite...)
        so_usa = "a página 1" if usa == 1 else f"as {usa} primeiras páginas"
        nas_lidas = "na página 1" if usa == 1 else f"nas {usa} primeiras páginas"
        if alvo and not any(i <= usa for i in alvo):
            bloqueio = (
                f"Esse PDF parece uma pasta com {n} páginas: {nome} está {_nas_paginas(alvo)}, "
                f"mas a leitura só usa {so_usa}. Mande {mandar}."
            )
            break
        dentro = [i for i in alvo if i <= usa]
        if dentro:
            aviso = (f"Esse PDF tem {n} páginas e parece uma pasta. Li {nome} {_nas_paginas(dentro)}: "
                     f"confira os dados. Se algo vier errado, mande {mandar}.")
        elif misturado:
            aviso = (f"Esse PDF tem {n} páginas e parece uma pasta: não reconheci {nome} {nas_lidas}, "
                     f"que são as que eu leio. Confira o resultado; se vier errado, mande {mandar}.")
        else:
            aviso = (f"Esse PDF tem {n} páginas e eu leio só {so_usa}. "
                     f"Se {nome} estiver depois, mande {mandar}.")
    return bloqueio, aviso
