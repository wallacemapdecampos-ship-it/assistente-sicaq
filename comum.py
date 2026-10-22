"""Base comum: Flask, IA (Groq + reserva), limites e funções usadas por vários leitores."""

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



def _espera_groq(detalhe):
    """Segundos de espera da mensagem da Groq ("try again in 1m12.5s"), ou None."""
    m = re.search(r"try again in\s+([0-9hms.]+)", str(detalhe or ""), re.I)
    if not m:
        return None
    total, achou = 0.0, False
    for valor, unidade in re.findall(r"([0-9.]+)(ms|h|m|s)", m.group(1)):
        try:
            v = float(valor)
        except ValueError:
            continue
        achou = True
        total += {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unidade] * v
    return total if achou else None



def _texto_espera(segundos):
    if segundos is None:
        return "Aguarde um minuto e clique de novo."
    if segundos < 90:
        return f"Aguarde cerca de {max(1, round(segundos))} segundos e clique de novo."
    minutos = round(segundos / 60)
    if minutos < 90:
        return f"Libera de novo em cerca de {minutos} minutos."
    return f"Libera de novo em cerca de {round(minutos / 60)} horas."



def _contar_tokens(body):
    """Soma os tokens que a IA usou nesta leitura (para o custo simulado)."""
    try:
        from flask import has_request_context
        if not has_request_context():
            return
        u = (body or {}).get("usage") or {}
        g.tokens_entrada = getattr(g, "tokens_entrada", 0) + int(u.get("prompt_tokens") or 0)
        g.tokens_saida = getattr(g, "tokens_saida", 0) + int(u.get("completion_tokens") or 0)
        g.chamadas_ia = getattr(g, "chamadas_ia", 0) + 1
    except Exception:
        pass



_GROQ_PAUSADA_ATE = 0.0  # enquanto a Groq estiver sem limite, usa só a reserva



def _groq_abrir(req, timeout=35, esperar_limite=True):
    """
    Envia o pedido à Groq. No limite POR MINUTO do plano gratuito, se a espera
    for curta (até 45 s), aguarda e tenta de novo sozinho; assim o analista
    não vê erro. Limite diário ou espera longa: avisa quanto tempo falta.
    """
    global _GROQ_PAUSADA_ATE
    if time.time() < _GROQ_PAUSADA_ATE:
        # Groq sem limite: vai direto na reserva, sem perder tempo tentando.
        if not esperar_limite:
            raise GroqLimite("Groq no limite; 2ª leitura pulada.")
        reserva = _ia_reserva(req, timeout)
        if reserva is not None:
            return reserva
    for tentativa in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            _contar_tokens(body)
            if _GROQ_PAUSADA_ATE:
                _GROQ_PAUSADA_ATE = 0.0
                print("[GROQ] voltou; reserva desligada", flush=True)
            return body
        except urllib.error.HTTPError as e:
            if e.code >= 500 and esperar_limite:
                print(f"[GROQ] fora do ar (HTTP {e.code}); tentando a IA reserva", flush=True)
                reserva = _ia_reserva(req, timeout)
                if reserva is not None:
                    return reserva
            if e.code != 429:
                raise
            detalhe = e.read().decode("utf-8", errors="replace")
            baixo = detalhe.lower()
            diario = "per day" in baixo or "(tpd)" in baixo or "(rpd)" in baixo
            espera = _espera_groq(detalhe)
            try:
                cabecalho = float((e.headers or {}).get("retry-after") or "")
            except (TypeError, ValueError):
                cabecalho = None
            if espera is None and cabecalho is not None:
                espera = cabecalho
            # A mensagem da Groq só fala de limites/tokens (sem dados do cliente).
            try:
                msg = json.loads(detalhe).get("error", {}).get("message", "")
            except Exception:
                msg = detalhe
            print(f"[GROQ] 429 {'diário' if diario else 'por minuto'}; espera={espera}; "
                  f"retry-after={cabecalho}; msg={msg[:300]!r}", flush=True)
            if esperar_limite and tentativa == 0 and not diario and espera is not None and espera <= 45:
                time.sleep(espera + 0.7)
                continue
            # Limite diário (ou espera longa): pula a Groq até ela liberar.
            if _reserva_configurada() and (diario or (espera or 0) > 45):
                _GROQ_PAUSADA_ATE = time.time() + (espera if espera is not None else 600) + 5
                print(f"[GROQ] pausada por {(_GROQ_PAUSADA_ATE - time.time()) / 60:.0f} min; usando só a reserva", flush=True)
            # Groq no limite: tenta a IA reserva (Cloudflare) com o mesmo pedido.
            # A 2ª leitura do holerite (esperar_limite=False) só é pulada.
            if esperar_limite:
                reserva = _ia_reserva(req, timeout)
                if reserva is not None:
                    return reserva
            if diario:
                raise GroqLimite("Limite diário gratuito da Groq atingido. " + _texto_espera(espera))
            raise GroqLimite("Limite por minuto da Groq atingido. " + _texto_espera(espera))
    if esperar_limite:
        reserva = _ia_reserva(req, timeout)
        if reserva is not None:
            return reserva
    raise GroqLimite("Limite por minuto da Groq atingido. " + _texto_espera(None))



CF_MODELO_QWEN = "@cf/qwen/qwen3.8-27b"            # o mesmo modelo da Groq



CF_MODELO_LLAMA = "@cf/meta/llama-4-scout-17b-16e-instruct"  # último recurso



def _reserva_configurada():
    return bool(os.getenv("CF_ACCOUNT_ID", "").strip() and os.getenv("CF_API_TOKEN", "").strip())



def _json_do_texto(texto):
    """Tira raciocínio <think>, cercas ```json e texto em volta; devolve o JSON limpo."""
    t = str(texto or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        json.loads(t)
        return t
    except Exception:
        pass
    ini, fim = t.find("{"), t.rfind("}")
    if ini >= 0 and fim > ini:
        trecho = t[ini:fim + 1]
        json.loads(trecho)
        return trecho
    raise ValueError("resposta sem JSON")



def _modelos_reserva():
    escolhido = os.getenv("CF_MODEL", "").strip()
    lista = [escolhido, CF_MODELO_QWEN, CF_MODELO_LLAMA]
    return [m for i, m in enumerate(lista) if m and m not in lista[:i]]



def _ia_reserva(req, timeout=35):
    """
    IA reserva (Cloudflare Workers AI), usada só quando a Groq bate no limite
    ou cai. Primeiro a MESMA Qwen da Groq (mesma qualidade, mesmos prompts);
    se ela falhar, a Llama 4 Scout. Recebe o pedido montado para a Groq e
    devolve no mesmo formato, ou None. A Cloudflare não treina com os dados.
    """
    if not _reserva_configurada():
        return None
    try:
        original = json.loads(req.data.decode("utf-8"))
    except Exception:
        return None
    conta = os.getenv("CF_ACCOUNT_ID", "").strip()
    token = os.getenv("CF_API_TOKEN", "").strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{conta}/ai/v1/chat/completions"
    saida = int(original.get("max_completion_tokens") or original.get("max_tokens") or 512)
    for modelo in _modelos_reserva():
        qwen = "qwen" in modelo
        payload = {
            "model": modelo,
            "messages": original.get("messages") or [],
            "temperature": original.get("temperature", 0.0),
            "max_tokens": max(saida, 256) + 200,
            "stream": False,
        }
        if qwen:
            # Sem "pensar" antes de responder: igual à Groq (reasoning none).
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if original.get("response_format"):
            payload["response_format"] = {"type": "json_object"}
        ultimo = None
        for tentativa in range(3):
            pedido = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Assistente-SICAQ-Reserva/1.1",
                },
                method="POST",
            )
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(pedido, timeout=max(timeout, 60)) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                texto = body["choices"][0]["message"]["content"]
                body["choices"][0]["message"]["content"] = _json_do_texto(texto)
                body["model"] = "reserva:" + modelo
                _contar_tokens(body)
                try:
                    from flask import has_request_context
                    if has_request_context():
                        g.ia_reserva = True
                except Exception:
                    pass
                print(f"[RESERVA] leitura feita por {modelo} em {time.perf_counter() - t0:.1f}s", flush=True)
                return body
            except urllib.error.HTTPError as e:
                # A mensagem de erro da Cloudflare não traz dados do cliente.
                detalhe = e.read().decode("utf-8", errors="replace")[:300]
                ultimo = f"HTTP {e.code}: {detalhe!r}"
                if e.code == 400 and "chat_template_kwargs" in payload:
                    payload.pop("chat_template_kwargs", None)
                    payload["reasoning_effort"] = "low"
                    continue
                if e.code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)  # sem modo JSON: tenta sem
                    continue
                break
            except Exception as e:
                ultimo = type(e).__name__
                if tentativa == 0 and isinstance(e, ValueError):
                    continue
                break
        print(f"[RESERVA] {modelo} falhou ({ultimo})", flush=True)
    return None



RATE_WINDOW = 60



RATE_MAX = 20



_rate = defaultdict(deque)



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



def _tirar_acentos(valor):
    texto = str(valor or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )



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



def _holerite_valor(valor):
    return f"{float(valor or 0):.2f}".replace(".", ",")



def _groq_json_imagem(prompt, image_bytes, max_tokens=120):
    """Chamada curta à Groq: uma imagem + instrução, resposta em JSON."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return _groq_json([
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ], max_tokens)



def _groq_json(conteudo, max_tokens=120):
    """Chamada à Groq com texto e/ou imagens já montados; resposta em JSON."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")
    model = os.getenv("GROQ_MODEL", "").strip() or "qwen/qwen3.8-27b"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": conteudo,
        }],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
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
            "User-Agent": "Assistente-SICAQ-PIS/1.0",
        },
        method="POST",
    )
    try:
        t0 = time.perf_counter()
        body = _groq_abrir(req)
        return json.loads(body["choices"][0]["message"]["content"]), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        detalhe = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            espera = ""
            m = re.search(r"try again in\s+([0-9.]+)s", detalhe, re.I)
            if m:
                espera = f" Aguarde cerca de {float(m.group(1)):.0f} segundos e tente novamente."
            raise GroqLimite("Limite gratuito do Groq atingido temporariamente." + espera)
        raise RuntimeError(f"Groq HTTP {e.code}: {detalhe}")



def _sim(valor):
    if isinstance(valor, bool):
        return valor
    return _tirar_acentos(str(valor or "")).strip().upper() in {
        "TRUE", "SIM", "S", "1", "YES", "VERDADEIRO",
    }



def _texto_limpo(valor):
    return re.sub(r"\s+", " ", _tirar_acentos(str(valor or "")).upper()).strip()



def _pdf_pagina_tem_foto(page):
    """Página que é uma foto/scan (imagem cobrindo a página): texto de OCR não é confiável."""
    try:
        area = page.rect.width * page.rect.height
        return any(
            (pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area
            for i in page.get_image_info()
        )
    except Exception:
        return True
