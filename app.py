"""Junta todos os módulos (o Render roda 'gunicorn app:app')."""

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
    _holerite_competencias_validas,
    _ia_reserva,
    _reserva_configurada,
    app,
)

from contas import (  # noqa: F401
    _login_ligado,
)

import comum, contas, imagem, pasta, leitor_cnh, leitor_identidade, leitor_irpf, leitor_renda, leitor_pis, leitor_certidao, leitor_endereco  # noqa: F401,E401 (registra as rotas)

# Reexporta tudo (compatibilidade com testes/ferramentas que usam app.<nome>).

for _m in (comum, contas, imagem, pasta, leitor_cnh, leitor_identidade, leitor_irpf, leitor_renda, leitor_pis, leitor_certidao, leitor_endereco):
    globals().update({k: v for k, v in vars(_m).items() if not k.startswith('__') and k not in globals()})



@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Assistente SICAQ Web",
        "groq_configurada": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "login": _login_ligado(),
        "holerite_competencias": _holerite_competencias_validas(),
    })



# ============================================================
# LEITURA REPETIDA: mesmo arquivo lido de novo em até 30 min devolve o
# resultado anterior, sem chamar a IA (não gasta o limite da Groq).
# Só o RESULTADO fica na memória do servidor, pela "impressão digital"
# (hash) do arquivo; o documento não é guardado. Some sozinho em 30 min
# e quando o servidor reinicia.
# ============================================================
CACHE_LEITURA_SEG = 30 * 60



CACHE_LEITURA_MAX = 300



_CACHE_LEITURA = {}  # chave -> (validade, resposta)



def _chave_leitura():
    h = hashlib.sha256()
    h.update(request.path.encode())
    # Holerite: o período aceito muda com o mês; certidão: o nome do cliente.
    if request.path == "/ler-holerite":
        h.update("|".join(_holerite_competencias_validas()).encode())
    for k in sorted(request.form.keys()):
        h.update(f"|{k}={request.form.get(k)}".encode())
    arquivos = request.files.getlist("files") or request.files.getlist("file")
    if not arquivos:
        return None
    for f in arquivos:
        try:
            f.stream.seek(0)
            h.update(hashlib.sha256(f.stream.read()).digest())
            f.stream.seek(0)
        except Exception:
            return None
    return h.hexdigest()



def _com_cache_de_leitura(funcao):
    import functools

    @functools.wraps(funcao)
    def envolta(*args, **kwargs):
        chave = None
        try:
            chave = _chave_leitura()
        except Exception:
            chave = None
        agora = time.time()
        if chave:
            item = _CACHE_LEITURA.get(chave)
            if item and item[0] > agora:
                print(f"[CACHE] {request.path}: leitura repetida, sem IA", flush=True)
                return jsonify({**item[1], "repetida": True, "tempo_api": 0, "tempo_total": 0})
        resp = funcao(*args, **kwargs)
        reserva = bool(getattr(g, "ia_reserva", False))
        try:
            corpo = resp.get_json(silent=True) if hasattr(resp, "get_json") else None
            if reserva and isinstance(corpo, dict):
                # Avisa o site; leitura da reserva não fica no cache, para a
                # próxima tentativa poder usar a Groq de novo.
                corpo = {**corpo, "ia_reserva": True}
                resp = (jsonify(corpo), getattr(resp, "status_code", 200))
                return resp
            if chave and getattr(resp, "status_code", 200) == 200 and isinstance(corpo, dict) and corpo.get("ok"):
                if len(_CACHE_LEITURA) >= CACHE_LEITURA_MAX:
                    for k in [k for k, v in _CACHE_LEITURA.items() if v[0] <= agora] or list(_CACHE_LEITURA)[:50]:
                        _CACHE_LEITURA.pop(k, None)
                _CACHE_LEITURA[chave] = (agora + CACHE_LEITURA_SEG, corpo)
        except Exception:
            pass
        return resp
    return envolta



for _rota in ("ler", "ler_holerite", "ler_pis", "ler_certidao", "ler_endereco"):
    if _rota in app.view_functions:
        app.view_functions[_rota] = _com_cache_de_leitura(app.view_functions[_rota])



def _testar_reserva_no_inicio():
    """
    Ao subir o servidor, faz 1 leitura de teste na IA reserva com uma imagem
    gerada aqui (só o texto TESTE 4821, nenhum dado de cliente) e escreve o
    resultado no log. Roda uma vez por deploy.
    """
    if not _reserva_configurada():
        print("[RESERVA] não configurada (CF_ACCOUNT_ID/CF_API_TOKEN)", flush=True)
        return
    marca = "/tmp/.teste_reserva_feito2"
    try:
        fd = os.open(marca, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    except Exception:
        pass

    def rodar():
        try:
            from PIL import ImageDraw
            im = Image.new("RGB", (360, 120), "white")
            ImageDraw.Draw(im).text((30, 50), "CODIGO: TESTE 4821", fill="black")
            buf = io.BytesIO()
            im.resize((720, 240)).save(buf, "JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            payload = {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": 'Leia o codigo da imagem. Responda SOMENTE JSON: {"codigo": "..."}'},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "temperature": 0.0, "max_completion_tokens": 60,
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request("https://invalido.local", data=json.dumps(payload).encode("utf-8"))
            body = _ia_reserva(req, 60)
            if body is None:
                print("[RESERVA] teste FALHOU (veja a linha acima)", flush=True)
            else:
                print(f"[RESERVA] teste ok: {body['choices'][0]['message']['content'][:80]}", flush=True)
        except Exception as e:
            print(f"[RESERVA] teste com erro ({type(e).__name__}: {e})", flush=True)

    import threading
    threading.Thread(target=rodar, daemon=True).start()



_testar_reserva_no_inicio()



if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
