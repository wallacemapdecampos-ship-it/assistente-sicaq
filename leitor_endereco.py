"""Comprovante de endereço, conferência do CEP e as rotas /ler-endereco e /conferir-cep."""

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
    _groq_json,
    _tirar_acentos,
    app,
)

from contas import (  # noqa: F401
    _registrar_uso,
)

from pasta import (  # noqa: F401
    _avaliar_pasta,
)

from leitor_identidade import (  # noqa: F401
    UFS_BR,
    _chave_cidade,
    _municipio_oficial,
)

from leitor_certidao import (  # noqa: F401
    _conteudo_certidao,
)



# ============================================================
# COMPROVANTE DE ENDEREÇO + CONFERÊNCIA DO CEP
# O SICAQ não aceita CEP genérico da cidade (ex.: 18100-000 de Sorocaba)
# quando a cidade tem CEP por rua: dá erro e derruba a sessão. Aqui o CEP
# lido é conferido no ViaCEP; se for genérico, busca o CEP da rua.
# ============================================================

PROMPT_ENDERECO = """
Leia este COMPROVANTE DE ENDEREÇO brasileiro (conta de luz, água, gás,
telefone, internet, TV, fatura de cartão ou banco, IPTU, contrato de aluguel)
e responda SOMENTE JSON válido com as chaves:
tipo_logradouro, logradouro, numero, complemento, bairro, cidade,
uf, cep, referencia, emissor, campos_incertos.

Regras:
- Use o endereço do CLIENTE (instalação, unidade consumidora, entrega ou
  correspondência do titular). NUNCA o endereço da empresa que emitiu a
  conta (sede, agência, CNPJ, "central de atendimento").
- Se houver endereço de instalação/unidade consumidora e outro de
  correspondência, use o de INSTALAÇÃO (onde a pessoa mora).
- tipo_logradouro: RUA, AVENIDA, TRAVESSA, ALAMEDA, ESTRADA, RODOVIA, PRAÇA,
  VIELA etc. (por extenso). logradouro: só o nome, sem o tipo.
- numero: só o número do imóvel (ou "SN" quando a conta disser que não tem número).
- complemento: SÓ apto, bloco, casa, fundos, lote, quadra (vazio se não houver).
  NUNCA ponha o bairro no complemento.
- logradouro e bairro: copie EXATAMENTE como impresso na conta, com as mesmas
  abreviações (se está "JD", devolva "JD"; não troque por "JARDIM"; se está
  "DR", devolva "DR"). Não corrija nem complete nomes.
- bairro: o bairro do imóvel (ex.: JD THOMAZ GUEDES, VL HORTENCIA, CENTRO,
  RESIDENCIAL ...). Nas contas ele costuma vir na linha de baixo da rua ou
  junto com a cidade. NUNCA use o nome da cidade como bairro.
- cep: somente os 8 dígitos. uf: sigla de 2 letras.
- referencia: mês/ano de referência ou vencimento da conta, em MM/AAAA.
- emissor: nome da empresa que emitiu (ex.: CPFL, SABESP, VIVO).
- Não invente. Campo ilegível = vazio e inclua a chave em campos_incertos.
""".strip()



TIPOS_LOGRADOURO = (
    "RUA", "AVENIDA", "TRAVESSA", "ALAMEDA", "ESTRADA", "RODOVIA", "PRACA",
    "VIELA", "VILA", "LARGO", "BECO", "VIA", "PARQUE", "CONDOMINIO", "CAMINHO",
    "PASSAGEM", "LADEIRA", "SERVIDAO", "VIADUTO", "RUELA", "QUADRA", "CHACARA",
    "SITIO", "FAZENDA", "JARDIM", "CONJUNTO", "LOTEAMENTO",
)



_ABREV_LOGRADOURO = {
    "R": "RUA", "AV": "AVENIDA", "AVN": "AVENIDA", "TV": "TRAVESSA", "TRAV": "TRAVESSA",
    "AL": "ALAMEDA", "EST": "ESTRADA", "ESTR": "ESTRADA", "ROD": "RODOVIA",
    "PC": "PRACA", "PCA": "PRACA", "PRC": "PRACA", "VL": "VILA", "LGO": "LARGO",
    "LG": "LARGO", "BC": "BECO", "CAM": "CAMINHO", "PQ": "PARQUE", "COND": "CONDOMINIO",
}



def _separar_tipo_logradouro(tipo, nome):
    """("R.", "SETE DE SETEMBRO") / ("", "R. SETE DE SETEMBRO") -> ("RUA", "SETE DE SETEMBRO")."""
    tipo = _tirar_acentos(str(tipo or "")).upper().strip(" .")
    nome = re.sub(r"\s+", " ", _tirar_acentos(str(nome or "")).upper()).strip(" ,.-")
    tipo = _ABREV_LOGRADOURO.get(tipo, tipo)
    m = re.match(r"^([A-Z]{1,12})\.?\s+(.+)$", nome)
    if m:
        primeiro = _ABREV_LOGRADOURO.get(m.group(1), m.group(1))
        if primeiro in TIPOS_LOGRADOURO:
            tipo = tipo or primeiro
            if primeiro == tipo:
                nome = m.group(2).strip()
    return (tipo if tipo in TIPOS_LOGRADOURO else tipo), nome



def _viacep(caminho, timeout=8):
    """GET no ViaCEP. Devolve o JSON (dict/list) ou None se não respondeu."""
    url = "https://viacep.com.br/ws/" + caminho + "/json/"
    req = urllib.request.Request(url, headers={"User-Agent": "Assistente-SICAQ/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return {"erro": True}
        print(f"[CEP] ViaCEP HTTP {e.code}", flush=True)
    except Exception as e:
        print(f"[CEP] ViaCEP indisponível ({type(e).__name__})", flush=True)
    return None



def _faixa_numero(complemento, numero):
    """
    O número do imóvel cabe na faixa do CEP? ("até 499", "de 500 ao fim",
    "de 2 a 98 - lado par", "lado ímpar"). Devolve True, False ou None (sem faixa).
    """
    c = _tirar_acentos(str(complemento or "")).lower()
    try:
        n = int(re.sub(r"\D", "", str(numero or "")) or "x")
    except ValueError:
        return None
    if not c.strip():
        return None
    ok = True
    if "lado par" in c and n % 2:
        ok = False
    if ("lado impar" in c) and not n % 2:
        ok = False
    m = re.search(r"de\s+(\d+)\s+(?:a|ate)\s+(\d+)", c)
    if m:
        return ok and int(m.group(1)) <= n <= int(m.group(2))
    m = re.search(r"de\s+(\d+)(?:/\d+)?\s+ao\s+fim", c)
    if m:
        return ok and n >= int(m.group(1))
    m = re.search(r"ate\s+(\d+)(?:/(\d+))?", c)
    if m:
        return ok and n <= max(int(m.group(1)), int(m.group(2) or 0))
    return ok if ("lado" in c) else None



def _cep_fmt(cep):
    """O SICAQ só aceita o CEP com os 8 números, sem traço."""
    return re.sub(r"\D", "", str(cep or ""))[:8]



def _conferir_cep(cep, logradouro="", numero="", bairro="", cidade="", uf="", tipo=""):
    """
    Confere o CEP no ViaCEP e, se for genérico ou não bater com a rua,
    procura o CEP da rua. Devolve dict com:
    status: "ok" | "generico" | "inexistente" | "diferente" | "sem_cep" | "indisponivel"
    mensagem, cep, cidade/uf oficiais e sugestoes [{cep, logradouro, complemento, bairro, cabe}].
    """
    cep_d = re.sub(r"\D", "", str(cep or ""))
    tipo, nome_rua = _separar_tipo_logradouro(tipo, logradouro)
    uf = _tirar_acentos(str(uf or "")).upper().strip()[:2]
    cidade = str(cidade or "").strip()
    saida = {"status": "", "mensagem": "", "cep": _cep_fmt(cep_d), "cidade": cidade, "uf": uf,
             "sugestoes": [], "pode_usar": False}

    info = None
    if len(cep_d) == 8:
        info = _viacep(cep_d)
        if info is None:
            saida.update(status="indisponivel", mensagem="Não consegui conferir o CEP agora (serviço de CEP fora do ar). Confira antes de usar.")
            return saida
        if info.get("erro"):
            saida.update(status="inexistente", mensagem="Esse CEP não existe. Confira no comprovante ou use uma das sugestões.")
            info = None
        else:
            saida["cidade"] = info.get("localidade") or cidade
            saida["uf"] = info.get("uf") or uf
            if (info.get("logradouro") or "").strip():
                rua_cep = _chave_cidade(info.get("logradouro"))
                rua_lida = _chave_cidade(nome_rua)
                import difflib
                parecida = (not rua_lida) or rua_lida in rua_cep or rua_cep.endswith(rua_lida) or \
                    difflib.SequenceMatcher(None, rua_lida, rua_cep).ratio() >= 0.75
                if parecida:
                    saida.update(status="ok", pode_usar=True, mensagem="CEP confere com a rua.",
                                 logradouro_cep=info.get("logradouro"), bairro_cep=_sem_pontuacao(info.get("bairro")))
                    return saida
                saida.update(status="diferente",
                             mensagem=f"Esse CEP é da {info.get('logradouro')}, não da rua do comprovante. Confira ou use uma sugestão.")
            else:
                saida.update(status="generico",
                             mensagem="CEP genérico da cidade: o SICAQ não aceita. Use o CEP da rua (sugestões abaixo).")
    else:
        saida.update(status="sem_cep", mensagem="CEP não encontrado no comprovante. Veja as sugestões pela rua.")

    # Busca pela rua (o ViaCEP exige cidade e rua com 3+ letras).
    cidade_busca = saida["cidade"] or cidade
    uf_busca = saida["uf"] or uf
    achados = []
    if uf_busca in UFS_BR and len(cidade_busca) >= 3 and len(nome_rua) >= 3:
        rua_busca = " ".join(nome_rua.split()[:6])
        lista = _viacep(f"{uf_busca}/{urllib.parse.quote(cidade_busca)}/{urllib.parse.quote(rua_busca)}")
        if isinstance(lista, list):
            achados = lista
    chave_bairro = _chave_cidade(bairro)
    sugestoes = []
    for item in achados[:50]:
        cabe = _faixa_numero(item.get("complemento"), numero)
        mesmo_bairro = bool(chave_bairro) and _chave_cidade(item.get("bairro")) == chave_bairro
        nota = (2 if cabe else (0 if cabe is False else 1)) + (1 if mesmo_bairro else 0)
        sugestoes.append({"cep": _cep_fmt(item.get("cep")), "logradouro": item.get("logradouro") or "",
                          "complemento": item.get("complemento") or "", "bairro": item.get("bairro") or "",
                          "cabe": cabe, "nota": nota})
    sugestoes.sort(key=lambda s: -s["nota"])
    for s in sugestoes:
        s.pop("nota", None)
    saida["sugestoes"] = sugestoes[:8]

    if saida["status"] == "generico" and not sugestoes:
        # Cidade pequena com CEP único: aí o genérico é o certo.
        saida.update(pode_usar=True, mensagem=(
            "CEP geral da cidade e não achei CEP por rua. Se a cidade só tem esse CEP, ele é o "
            "correto; se tiver CEP por rua, confira nos Correios antes de usar."))
    elif saida["status"] == "indisponivel":
        pass
    elif sugestoes and saida["status"] in ("generico", "inexistente", "sem_cep"):
        # Troca sozinho pelo CEP REAL da mesma rua (o melhor: faixa do número e
        # bairro). Rua, número e bairro continuam os do comprovante.
        melhor = sugestoes[0]
        saida["sugerido"] = melhor["cep"]
        certeza = melhor["cabe"] is True or len(sugestoes) == 1
        antes = {"generico": "O CEP do comprovante era genérico", "inexistente": "O CEP do comprovante não existe",
                 "sem_cep": "O comprovante não tinha CEP"}[saida["status"]]
        saida.update(
            cep_original=saida["cep"], cep=melhor["cep"], status="trocado", pode_usar=True,
            bairro_cep=_sem_pontuacao(melhor["bairro"]),
            certeza=certeza,
            mensagem=(f"{antes}. Coloquei o CEP da rua: {melhor['cep']} ({melhor['logradouro']}"
                      f"{' — ' + melhor['complemento'] if melhor['complemento'] else ''}"
                      f"{', ' + melhor['bairro'] if melhor['bairro'] else ''}). "
                      + ("" if certeza else "A rua tem mais de um CEP e o número não define qual: confira. ")),
        )
    elif sugestoes and saida["status"] == "diferente":
        melhor = sugestoes[0]
        if melhor["cabe"] or len(sugestoes) == 1:
            saida["sugerido"] = melhor["cep"]
    return saida



_INICIO_BAIRRO = re.compile(
    r"^(JD|JARDIM|VL|VILA|PQ|PRQ|PARQUE|RES|RESID|RESIDENCIAL|CJ|CONJ|CONJUNTO|NUCLEO|NUC|"
    r"BAIRRO|CENTRO|CHACARA|CH|LOTEAMENTO|LOT|PORTAL|RECANTO|CONDOMINIO|COND|HAB|HABITACIONAL|"
    r"SITIO|ESTANCIA|JDIM|BOSQUE|CIDADE|DISTRITO|ALTO)\b")



_COMPLEMENTO_DE_VERDADE = re.compile(r"\b(AP|APT|APTO|APARTAMENTO|BL|BLOCO|CASA|FUNDOS|FDS|LT|LOTE|QD|QUADRA|SALA|ANDAR|TORRE|BOX|LOJA)\b")



def _sem_pontuacao(v):
    """O SICAQ não aceita ponto, vírgula, traço, barra etc.: só letras, números e espaço."""
    t = _tirar_acentos(str(v or "")).upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()



def _normalizar_endereco(r):
    """Limpa o que a IA leu e confere a cidade com a lista do IBGE."""
    limpo = _sem_pontuacao
    tipo, rua = _separar_tipo_logradouro(r.get("tipo_logradouro"), r.get("logradouro"))
    numero = limpo(r.get("numero"))
    # Número grudado na rua ("ARMANDO PANNUNZIO, 1.200"): vai para o Número.
    m = re.search(r",\s*(\d[\d.]*)\s*$", rua)
    if m:
        if not numero or re.fullmatch(r"S ?N[O]?|SEM NUMERO|SEM N|S", numero):
            numero = re.sub(r"\D", "", m.group(1))
        rua = rua[:m.start()]
    rua = limpo(rua)
    uf = limpo(r.get("uf"))[:2]
    cidade = limpo(r.get("cidade"))
    incertos = [str(x) for x in (r.get("campos_incertos") or [])]
    # SICAQ: Número só aceita dígitos; sem número = 99999 (regra da equipe).
    if re.fullmatch(r"S ?N[O]?|SEM NUMERO|SEM N|S", numero):
        numero = "99999"
    numero = re.sub(r"\D", "", numero)[:7]
    if not numero:
        numero = "99999"  # não achei o número na conta: confira
        incertos.append("numero")
    if cidade:
        oficial, uf_of, certeza = _municipio_oficial(cidade, uf if uf in UFS_BR else "")
        cidade, uf = (oficial or cidade), (uf_of or uf)
        if not certeza:
            incertos.append("cidade")
    bairro = limpo(r.get("bairro"))
    complemento = limpo(r.get("complemento"))
    # Bairro foi parar no complemento e a cidade no bairro (ex.: complemento
    # "JD THOMAZ GUEDES", bairro "TATUI"): desfaz a troca.
    cidade_ch = _chave_cidade(cidade)
    bairro_e_cidade = not bairro or _chave_cidade(bairro) == cidade_ch
    if bairro_e_cidade and complemento and _INICIO_BAIRRO.match(complemento) \
            and not _COMPLEMENTO_DE_VERDADE.search(complemento):
        bairro, complemento = complemento, ""
        incertos.append("bairro")
    elif bairro and _chave_cidade(bairro) == cidade_ch:
        bairro = ""  # cidade não é bairro: o CEP da rua pode trazer o certo
        incertos.append("bairro")
    return {
        "tipo_logradouro": tipo,
        "logradouro": rua,
        "numero": numero,
        "complemento": complemento,
        "bairro": bairro,
        "cidade": _sem_pontuacao(cidade),
        "uf": uf if uf in UFS_BR else "",
        "cep": re.sub(r"\D", "", str(r.get("cep") or ""))[:8],
        "referencia": str(r.get("referencia") or "").strip()[:7],
        "emissor": limpo(r.get("emissor")),
        "campos_incertos": sorted(set(incertos)),
    }



@app.post("/ler-endereco")
def ler_endereco():
    arquivos = []
    for f in (request.files.getlist("files") or request.files.getlist("file"))[:2]:
        if not f or not (f.filename or "").strip():
            continue
        data = f.read()
        if data:
            arquivos.append({"data": data, "filename": f.filename or "", "mimetype": f.mimetype or ""})
    if not arquivos:
        return jsonify({"ok": False, "error": "Comprovante de endereço não enviado."}), 400
    inicio = time.perf_counter()
    bloqueio, aviso_pasta = _avaliar_pasta(arquivos, "conta", 2)
    if bloqueio:
        return jsonify({"ok": False, "code": "PASTA", "error": bloqueio}), 422
    try:
        conteudo, texto_pdf, n_imagens = _conteudo_certidao(arquivos, PROMPT_ENDERECO, "COMPROVANTE")
        lido, tempo_api = _groq_json(conteudo, 300)
        dados = _normalizar_endereco(lido)
        conf = _conferir_cep(dados["cep"], dados["logradouro"], dados["numero"], dados["bairro"],
                             dados["cidade"], dados["uf"], dados["tipo_logradouro"])
        # Rua e bairro ficam escritos como na conta (JD continua JD). Só se a
        # conta não tiver bairro, usa o do CEP, marcado para conferir.
        if not dados["bairro"] and conf.get("bairro_cep"):
            dados["bairro"] = conf["bairro_cep"]
            dados["campos_incertos"] = sorted(set(dados["campos_incertos"] + ["bairro"]))
        print(f"[ENDERECO] cep={conf['status']} sugestoes={len(conf['sugestoes'])} "
              f"texto_pdf={bool(texto_pdf)} imagens={n_imagens}", flush=True)
        _registrar_uso("endereco")
        return jsonify({
            "ok": True, "dados": dados, "cep": conf, "aviso_pasta": aviso_pasta,
            "tempo_api": round(tempo_api, 3), "tempo_total": round(time.perf_counter() - inicio, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



@app.post("/conferir-cep")
def conferir_cep():
    """O analista mudou o CEP ou a rua no site: confere de novo (sem IA)."""
    d = request.get_json(silent=True) or {}
    conf = _conferir_cep(d.get("cep"), d.get("logradouro"), d.get("numero"), d.get("bairro"),
                         d.get("cidade"), d.get("uf"), d.get("tipo_logradouro"))
    return jsonify({"ok": True, "cep": conf})
