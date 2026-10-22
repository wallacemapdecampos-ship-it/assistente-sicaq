"""Certidão -> estado civil e a rota /ler-certidao."""

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
    _parse_data_br,
    _pdf_pagina_tem_foto,
    _sim,
    _texto_limpo,
    app,
)

from contas import (  # noqa: F401
    _registrar_uso,
)

from imagem import (  # noqa: F401
    NOTA_METADES,
    PathLikeSuffix,
    _abrir_imagem,
    _recortar_documento,
)

from pasta import (  # noqa: F401
    _avaliar_pasta,
)



# ============================================================
# CERTIDÃO (nascimento, casamento ou óbito) -> ESTADO CIVIL do SICAQ
# A IA só transcreve os fatos (tipo, regime de bens, averbações);
# a escolha da opção do SICAQ é feita aqui, por regra fixa.
# ============================================================

EC_SOLTEIRO = "SOLTEIRO (A)"



EC_DIVORCIADO = "DIVORCIADO (A)"



EC_SEPARADO = "SEPARADO (A) JUDICIALMENTE"



EC_VIUVO = "VIÚVO (A)"



EC_CASADO_TOTAL = "CASADO (A) COM COMUNHÃO TOTAL DE BENS"



EC_CASADO_SEPARACAO = "CASADO (A) COM SEPARAÇÃO DE BENS"



EC_CASADO_PARCIAL = "CASADO (A) COM COMUNHÃO PARCIAL DE BENS"



ESTADOS_CIVIS_SICAQ = (
    EC_SOLTEIRO, EC_DIVORCIADO, EC_SEPARADO, EC_VIUVO,
    EC_CASADO_TOTAL, EC_CASADO_SEPARACAO, EC_CASADO_PARCIAL,
)



# Lei do Divórcio (6.515/77, em vigor desde 27/12/1977): sem pacto
# antenupcial, o regime legal passou de comunhão universal para parcial.
DATA_LEI_DIVORCIO = date(1977, 12, 27)



PROMPT_CERTIDAO = """
Leia esta CERTIDÃO do Registro Civil brasileiro (nascimento, casamento ou
óbito) para descobrir o ESTADO CIVIL atual da pessoa. Leia com atenção o
campo "AVERBAÇÕES / ANOTAÇÕES" (pode estar no fim da folha ou no verso):
é ali que aparecem divórcio, separação, óbito e casamento.

Responda SOMENTE JSON com estas chaves:
- tipo_certidao: "NASCIMENTO", "CASAMENTO", "OBITO" ou "OUTRO".
- nomes: lista com o nome completo das pessoas registradas. Nascimento: o
  registrado. Casamento: os dois cônjuges (inclua também o nome que cada um
  passou a usar, se mudou). Óbito: o falecido.
- conjuge_do_falecido: só na de óbito, o nome do cônjuge/viúvo(a) informado.
- estado_civil_falecido: só na de óbito, o estado civil escrito do falecido.
- data_casamento: DD/MM/AAAA (só na de casamento).
- regime_bens: copie o regime de bens exatamente como está escrito.
- averbacoes: resumo curto (até 30 palavras) do que consta em averbações e
  anotações; vazio se nada consta.
- divorcio: true se houver averbação/anotação de DIVÓRCIO (inclusive
  conversão de separação em divórcio).
- separacao: true se houver averbação de SEPARAÇÃO (judicial, consensual ou
  extrajudicial).
- restabelecimento: true se houver RESTABELECIMENTO da sociedade conjugal.
- obito_conjuge: true se houver anotação/averbação de ÓBITO de um dos cônjuges.
- casamento_anotado: true se for certidão de NASCIMENTO com anotação ou
  averbação de CASAMENTO.
- nulidade: true se houver NULIDADE ou ANULAÇÃO do casamento.

Regras: só marque true se estiver escrito na certidão (não deduza); use false
quando não houver. Campo que não existir: "". Copie os nomes letra por letra.
""".strip()



NOTA_PAGINAS_CERTIDAO = (
    "ATENÇÃO: as imagens a seguir são páginas (frente e verso) da MESMA "
    "certidão. As averbações costumam ficar no verso ou no fim da folha."
)



_PARTICULAS_NOME = {"DE", "DA", "DO", "DAS", "DOS", "E", "D"}



def _tokens_nome(nome):
    t = re.sub(r"[^A-Z ]", " ", _texto_limpo(nome))
    return [p for p in t.split() if p not in _PARTICULAS_NOME]



def _mesma_pessoa(nome_a, nome_b):
    """
    Mesmo 1º nome e ao menos um sobrenome em comum (o sobrenome muda com
    casamento/divórcio, por isso não exige o último nome igual).
    None quando um dos nomes está vazio.
    """
    a, b = _tokens_nome(nome_a), _tokens_nome(nome_b)
    if not a or not b:
        return None
    if a[0] != b[0]:
        return False
    if len(a) == 1 or len(b) == 1:
        return True
    return bool(set(a[1:]) & set(b[1:]))



def _regime_sicaq(regime):
    """Devolve (opção do SICAQ ou "", tem_certeza, observação)."""
    r = _texto_limpo(regime)
    if not r or r in {"-", "NAO CONSTA", "NADA CONSTA", "NAO INFORMADO"}:
        return "", True, ""
    if "AQUESTO" in r or "PARTICIPACAO FINAL" in r:
        return "", False, "AQUESTOS"
    if "SEPARA" in r:  # separação de bens / total / convencional / obrigatória / legal
        return EC_CASADO_SEPARACAO, True, ""
    if "PARCIAL" in r:
        return EC_CASADO_PARCIAL, True, ""
    if "UNIVERSAL" in r or "TOTAL" in r:
        return EC_CASADO_TOTAL, True, ""
    if "COMUNH" in r:  # "comunhão de bens" (certidão antiga): em geral universal
        return EC_CASADO_TOTAL, False, ""
    return "", False, "DESCONHECIDO"



def _tipo_certidao(valor):
    t = _texto_limpo(valor)
    if "OBITO" in t:
        return "OBITO"
    if "CASAMENTO" in t:
        return "CASAMENTO"
    if "NASCIMENTO" in t:
        return "NASCIMENTO"
    return "OUTRO"



def _estado_civil_certidao(d, nome_doc="", texto_pdf=""):
    """Regras fixas: fatos lidos da certidão -> opção exata do SICAQ."""
    d = d if isinstance(d, dict) else {}
    tipo = _tipo_certidao(d.get("tipo_certidao"))
    nomes_brutos = d.get("nomes")
    if isinstance(nomes_brutos, str):
        nomes_brutos = re.split(r"\s*[;\n]\s*", nomes_brutos)
    nomes = []
    for n in nomes_brutos or []:
        n = _texto_limpo(n)
        if n and n not in nomes:
            nomes.append(n)
    conjuge = _texto_limpo(d.get("conjuge_do_falecido"))
    estado_falecido = _texto_limpo(d.get("estado_civil_falecido"))
    regime = _texto_limpo(d.get("regime_bens"))
    data_cas = _parse_data_br(d.get("data_casamento"))
    averbacoes = re.sub(r"\s+", " ", str(d.get("averbacoes") or "")).strip()
    divorcio = _sim(d.get("divorcio"))
    separacao = _sim(d.get("separacao"))
    restabelecimento = _sim(d.get("restabelecimento"))
    obito = _sim(d.get("obito_conjuge"))
    casamento_anotado = _sim(d.get("casamento_anotado"))
    nulidade = _sim(d.get("nulidade"))

    estado, motivo, incerto, avisos = "", "", False, []

    # PDF digital: a palavra escrita que a IA não marcou vira aviso.
    if texto_pdf:
        t = _texto_limpo(texto_pdf)
        if not divorcio and re.search(r"(?<!LEI DO )DIVORCI", t):
            incerto = True
            avisos.append("A certidão menciona DIVÓRCIO: confira as averbações.")
        if not separacao and re.search(
            r"SEPARACAO (JUDICIAL|CONSENSUAL|EXTRAJUDICIAL|LITIGIOSA)", t
        ):
            incerto = True
            avisos.append("A certidão menciona SEPARAÇÃO: confira as averbações.")
        if tipo == "CASAMENTO" and not obito and re.search(r"\bOBITO\b|FALEC", t):
            incerto = True
            avisos.append("A certidão menciona ÓBITO: confira as averbações.")
        if tipo == "NASCIMENTO" and not casamento_anotado and re.search(
            r"\bCASOU|CONTRAIU (O )?CASAMENTO|CASAMENTO (COM|CELEBRADO|REALIZADO)", t
        ):
            incerto = True
            avisos.append("A certidão menciona CASAMENTO: confira as anotações.")

    if tipo == "NASCIMENTO":
        if casamento_anotado and divorcio:
            estado, incerto = EC_DIVORCIADO, True
            motivo = "Certidão de nascimento com anotação de casamento e divórcio"
            avisos.append("Se possível, confirme com a certidão de casamento.")
        elif casamento_anotado and obito:
            estado, incerto = EC_VIUVO, True
            motivo = "Certidão de nascimento com anotação de casamento e óbito do cônjuge"
            avisos.append("Se possível, confirme com a certidão de casamento.")
        elif casamento_anotado:
            motivo = "Certidão de nascimento com anotação de CASAMENTO"
            incerto = True
            avisos.append("A pessoa é casada: peça a certidão de casamento para ver o regime de bens.")
        else:
            estado = EC_SOLTEIRO
            motivo = "Certidão de nascimento sem anotação de casamento"
    elif tipo == "CASAMENTO":
        if nulidade:
            incerto = True
            motivo = "Certidão de casamento com averbação de nulidade/anulação"
            avisos.append("Casamento anulado: confirme o estado civil com o cliente.")
        elif divorcio:
            estado = EC_DIVORCIADO
            motivo = "Certidão de casamento com averbação de divórcio"
        elif obito:
            estado = EC_VIUVO
            motivo = "Certidão de casamento com anotação de óbito do cônjuge"
        elif separacao and not restabelecimento:
            estado = EC_SEPARADO
            motivo = "Certidão de casamento com averbação de separação"
        else:
            opcao, certeza, obs = _regime_sicaq(regime)
            motivo = "Certidão de casamento sem divórcio/separação averbados"
            if restabelecimento:
                motivo += " (separação desfeita por restabelecimento)"
            if opcao:
                estado = opcao
                if not certeza:
                    incerto = True
                    avisos.append(f"Regime escrito: {regime}. Confira na certidão.")
            elif obs == "AQUESTOS":
                incerto = True
                avisos.append("Regime de participação final nos aquestos não existe no SICAQ: escolha manualmente.")
            elif obs == "DESCONHECIDO":
                incerto = True
                avisos.append(f"Regime de bens não reconhecido ({regime}): escolha manualmente.")
            elif data_cas:
                estado = EC_CASADO_TOTAL if data_cas < DATA_LEI_DIVORCIO else EC_CASADO_PARCIAL
                incerto = True
                legal = "comunhão universal" if data_cas < DATA_LEI_DIVORCIO else "comunhão parcial"
                avisos.append(
                    f"Não achei o regime de bens; pela data do casamento "
                    f"({data_cas.strftime('%d/%m/%Y')}) o regime legal é {legal}. Confira."
                )
            else:
                incerto = True
                avisos.append("Não achei o regime de bens: escolha manualmente.")
    elif tipo == "OBITO":
        estado = EC_VIUVO
        motivo = "Certidão de óbito do cônjuge"
        if estado_falecido and "CASAD" not in estado_falecido:
            incerto = True
            if any(x in estado_falecido for x in ("DIVORCI", "SOLTEIR", "VIUV")):
                estado = ""
                avisos.append(
                    f"O falecido era {estado_falecido}: por esta certidão o cliente não fica viúvo."
                )
            else:
                avisos.append(f"Estado civil do falecido: {estado_falecido}. Confira.")
        if not conjuge:
            incerto = True
            avisos.append("A certidão não mostra o nome do cônjuge: confira se o cliente era casado com o falecido.")
    else:
        incerto = True
        motivo = "Não reconheci como certidão de nascimento, casamento ou óbito"

    nome_confere = None
    if nome_doc:
        if tipo == "OBITO":
            nome_confere = _mesma_pessoa(conjuge, nome_doc) if conjuge else None
        elif nomes:
            nome_confere = any(_mesma_pessoa(n, nome_doc) for n in nomes)
        if nome_confere is False:
            incerto = True
            de_quem = conjuge if tipo == "OBITO" else ", ".join(nomes)
            avisos.append(f"A certidão é de {de_quem}: nome diferente do documento lido.")

    return {
        "estado_civil": estado,
        "tipo_certidao": tipo,
        "motivo": motivo,
        "incerto": bool(incerto or not estado),
        "avisos": avisos,
        "nomes": nomes,
        "conjuge_do_falecido": conjuge,
        "regime_bens": regime,
        "data_casamento": data_cas.strftime("%d/%m/%Y") if data_cas else "",
        "averbacoes": averbacoes,
        "nome_confere": nome_confere,
    }



def _certidao_arquivo(item):
    """
    Devolve ("texto", str) para PDF digital (e-certidão) ou
    ("imagens", [PIL...]) para foto, print ou PDF escaneado (até 2 páginas).
    """
    data, nome, mime = item["data"], item["filename"], item["mimetype"]
    if PathLikeSuffix(nome) == ".pdf" or mime == "application/pdf":
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            paginas = [doc[i] for i in range(min(2, len(doc)))]
            if not paginas:
                raise ValueError("PDF sem páginas válidas.")
            if not any(_pdf_pagina_tem_foto(p) for p in paginas):
                texto = "\n".join((p.get_text("text") or "") for p in paginas).strip()
                if len(texto) >= 150:
                    return "texto", texto
            imagens = []
            for p in paginas:
                foto = _pdf_pagina_tem_foto(p)
                zoom = 3.0 if foto else 2.2
                pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").copy()
                if foto:
                    img, _ = _recortar_documento(img, frac_min=0.25)
                imagens.append(img)
            return "imagens", imagens
        finally:
            doc.close()
    img = _abrir_imagem(data)
    img, _ = _recortar_documento(img, frac_min=0.25)
    return "imagens", [img]



def _jpeg_certidao(img, max_lado=2000):
    if max(img.size) > max_lado:
        f = max_lado / max(img.size)
        img = img.resize(
            (max(1, int(img.width * f)), max(1, int(img.height * f))),
            Image.Resampling.LANCZOS,
        )
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue()



def _conteudo_certidao(arquivos, prompt=None, rotulo="CERTIDÃO"):
    """Monta a mensagem para a IA. Devolve (conteudo, texto_pdf, n_imagens)."""
    textos, imagens = [], []
    for item in arquivos[:2]:
        tipo, valor = _certidao_arquivo(item)
        if tipo == "texto":
            textos.append(valor)
        else:
            imagens.extend(valor)
    imagens = imagens[:2]

    jpegs, nota = [], ""
    if len(imagens) == 1:
        img = imagens[0]
        w, h = img.size
        if h > 1.25 * w:
            # Folha em pé: letras pequenas. Metade de cima e de baixo
            # (com uma faixa repetida) dobram a resolução para a IA.
            jpegs = [
                _jpeg_certidao(img.crop((0, 0, w, int(h * 0.54)))),
                _jpeg_certidao(img.crop((0, int(h * 0.46), w, h))),
            ]
            nota = NOTA_METADES
        else:
            jpegs = [_jpeg_certidao(img)]
    elif imagens:
        jpegs = [_jpeg_certidao(i) for i in imagens]
        nota = NOTA_PAGINAS_CERTIDAO

    conteudo = [{"type": "text", "text": prompt or PROMPT_CERTIDAO}]
    for i, texto in enumerate(textos, start=1):
        conteudo.append({
            "type": "text",
            "text": f"\n===== TEXTO DO {rotulo} (arquivo {i}) =====\n{texto[:12000]}",
        })
    if len(jpegs) > 1:
        conteudo.append({"type": "text", "text": nota})
    for j in jpegs:
        conteudo.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(j).decode("ascii")
        }})
    return conteudo, "\n".join(textos), len(jpegs)



@app.post("/ler-certidao")
def ler_certidao():
    arquivos = []
    for f in (request.files.getlist("files") or request.files.getlist("file"))[:2]:
        if not f or not (f.filename or "").strip():
            continue
        data = f.read()
        if data:
            arquivos.append({"data": data, "filename": f.filename or "", "mimetype": f.mimetype or ""})
    if not arquivos:
        return jsonify({"ok": False, "error": "Arquivo da certidão não enviado."}), 400
    nome_doc = str(request.form.get("nome") or "").strip()
    inicio = time.perf_counter()
    bloqueio, aviso_pasta = _avaliar_pasta(arquivos, "certidao", 2)
    if bloqueio:
        print("[PASTA] certidão fora das páginas lidas", flush=True)
        return jsonify({"ok": False, "code": "PASTA", "error": bloqueio}), 422
    try:
        conteudo, texto_pdf, n_imagens = _conteudo_certidao(arquivos)
        lido, tempo_api = _groq_json(conteudo, 320)
        r = _estado_civil_certidao(lido, nome_doc, texto_pdf)
        print(
            f"[CERTIDAO] tipo={r['tipo_certidao']} estado={r['estado_civil'] or '-'} "
            f"incerto={r['incerto']} imagens={n_imagens} texto_pdf={bool(texto_pdf)}",
            flush=True,
        )
        _registrar_uso("certidao")
        return jsonify({
            "ok": True, **r,
            "aviso_pasta": aviso_pasta,
            "fonte": "texto" if texto_pdf and not n_imagens else "imagem",
            "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - inicio, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
