"""Declaração de Imposto de Renda (texto do PDF ou imagem)."""

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
    _groq_json_imagem,
    _holerite_competencias_validas,
    _holerite_valor,
    _tirar_acentos,
)

from contas import (  # noqa: F401
    FUSO_BR,
    _registrar_uso,
)

from imagem import (  # noqa: F401
    PathLikeSuffix,
    _abrir_imagem,
)



# ============================================================
# DECLARAÇÃO DE IMPOSTO DE RENDA (PDF da Receita) — sem IA
# Regra da equipe: a renda é a soma de "Rendimentos tributáveis recebidos de
# pessoa jurídica pelo titular" de cada fonte, dividida por 12. Na declaração
# bruta = líquida. Tipo de fonte pelo documento (CNPJ = JURÍDICA). Data do
# comprovante = mês anterior ao atual. Só vale a declaração do exercício do
# ano atual (ex.: em 2026, exercício 2026 / ano-calendário 2025). O SICAQ
# bloqueia o IR retido nesse tipo de comprovante, então ele não é enviado.
# ============================================================

def _mes_anterior():
    agora = datetime.now(FUSO_BR)
    m, a = agora.month - 1, agora.year
    if m == 0:
        m, a = 12, a - 1
    return f"{m:02d}/{a:04d}"



def _linhas_por_posicao(palavras):
    """Linhas do PDF montadas pela posição das palavras (texto, valores BR)."""
    ws = sorted((w for w in palavras if str(w[4]).strip()), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    linhas = []
    for w in ws:
        yc, alt = (w[1] + w[3]) / 2, max(1.0, w[3] - w[1])
        if linhas and abs(linhas[-1]["y"] - yc) <= 0.45 * alt:
            linhas[-1]["ws"].append(w)
        else:
            linhas.append({"y": yc, "ws": [w]})
    return [" ".join(str(w[4]) for w in sorted(l["ws"], key=lambda w: w[0])) for l in linhas]



def _valor_br(txt):
    return float(str(txt).replace(".", "").replace(",", "."))



def _irpf_do_pdf(arquivos):
    """
    PDF da Declaração de Ajuste Anual -> dados da renda, ou None se não for.
    Lê as fontes de "Rendimentos tributáveis recebidos de pessoa jurídica pelo
    titular": nome, CNPJ/CPF e rendimentos do ano.
    """
    for item in arquivos:
        if PathLikeSuffix(item.get("filename", "")) != ".pdf" and item.get("mimetype") != "application/pdf":
            continue
        try:
            doc = pymupdf.open(stream=item["data"], filetype="pdf")
        except Exception:
            continue
        try:
            textos, linhas = [], []
            for i in range(min(len(doc), 20)):
                pagina = doc[i]
                txt = pagina.get_text("text") or ""
                textos.append(txt)
                try:
                    ls = _linhas_por_posicao(pagina.get_text("words") or [])
                except Exception:
                    ls = []
                linhas += ls or [l for l in txt.splitlines()]
        finally:
            doc.close()
        tudo = _tirar_acentos("\n".join(textos)).upper()
        if "DECLARACAO DE AJUSTE ANUAL" not in tudo or "IMPOSTO SOBRE A RENDA" not in tudo:
            continue
        m = re.search(r"EXERCICIO\s+(\d{4})\s+ANO-CALENDARIO\s+(\d{4})", tudo)
        exercicio, ano_cal = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        m = re.search(r"CPF:\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", tudo)
        cpf_titular = re.sub(r"\D", "", m.group(1)) if m else ""

        fontes, total, dentro, atual = [], None, False, None
        for bruto in linhas:
            ln = _tirar_acentos(str(bruto)).upper().strip()
            if not ln:
                continue
            if ln.startswith("RENDIMENTOS TRIBUTAVEIS RECEBIDOS DE PESSOA JURIDICA PELO TITULAR"):
                # Só a seção principal (não a de exigibilidade suspensa).
                dentro = "SUSPENSA" not in ln and "ACUMULAD" not in ln
                atual = None
                continue
            if not dentro:
                continue
            if ln.startswith("RENDIMENTOS ") or ln.startswith("SEM INFORMA"):
                dentro = False
                continue
            valores = re.findall(r"-?\d{1,3}(?:\.\d{3})*,\d{2}", ln)
            if ln.startswith("TOTAL"):
                if valores:
                    total = _valor_br(valores[0])
                dentro = False
                continue
            if ln.startswith("CNPJ/CPF") or ln.startswith("CPF/CNPJ"):
                if atual is not None:
                    atual["documento"] = re.sub(r"\D", "", ln.split(":", 1)[-1])
                continue
            if len(valores) >= 5:
                nome = ln[: ln.find(valores[0])].strip() if valores[0] in ln else ""
                atual = {"nome": nome, "documento": "", "anual": _valor_br(valores[0]),
                         "previdencia": _valor_br(valores[1]),
                         "irrf_anual": _valor_br(valores[2]) if len(valores) > 2 else 0.0,
                         "decimo": _valor_br(valores[3]) if len(valores) > 3 else 0.0}
                fontes.append(atual)
                continue
            if atual is not None and not valores and not any(
                    k in ln for k in ("NOME DA FONTE", "REND. RECEBIDOS", "DE PES. JURIDICA", "OFICIAL", "SALARIO")):
                atual["nome"] = (atual["nome"] + " " + str(bruto).strip()).strip()
        return {"exercicio": exercicio, "ano_calendario": ano_cal, "cpf_titular": cpf_titular,
                "fontes": fontes, "total": total}
    return None



PROMPT_IRPF = """
Esta é a 1ª página da DECLARAÇÃO DE AJUSTE ANUAL do Imposto de Renda
(Receita Federal). Responda SOMENTE JSON com as chaves:
exercicio, ano_calendario, cpf_titular, fontes, total.

- exercicio e ano_calendario: os anos do cabeçalho ("EXERCÍCIO 2026",
  "ANO-CALENDÁRIO 2025"), só os 4 dígitos.
- cpf_titular: o CPF do cabeçalho, como está impresso.
- fontes: SOMENTE as linhas da seção "RENDIMENTOS TRIBUTÁVEIS RECEBIDOS DE
  PESSOA JURÍDICA PELO TITULAR" (NÃO a dos dependentes, NÃO a de
  exigibilidade suspensa, NÃO a de recebidos acumuladamente). Para cada
  fonte pagadora: {"nome": nome completo da fonte (pode ter 2 linhas),
  "documento": o CNPJ/CPF dela como está impresso, "rendimentos": a coluna
  "REND. RECEBIDOS DE PES. JURÍDICA", "previdencia": a coluna "CONTR.
  PREVID. OFICIAL", "irrf": a coluna "IMPOSTO RETIDO NA FONTE", "decimo":
  a coluna "13º SALÁRIO"}. Valores como estão impressos (ex.: "180.094,44").
- total: o valor da coluna "REND. RECEBIDOS DE PES. JURÍDICA" na linha TOTAL
  dessa seção.
- Se a seção estiver "Sem Informações", fontes = [] e total = "".
""".strip()



def _numero_ir(valor):
    if isinstance(valor, (int, float)):
        return float(valor)
    t = re.sub(r"[^\d,.-]", "", str(valor or ""))
    if not t:
        return 0.0
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return 0.0



def _irpf_primeira_pagina(arquivos):
    """JPEG da 1ª página da declaração (PDF sem texto ou foto), em boa resolução."""
    for item in arquivos:
        data, nome, mime = item["data"], item.get("filename", ""), item.get("mimetype", "")
        if PathLikeSuffix(nome) == ".pdf" or mime == "application/pdf":
            doc = pymupdf.open(stream=data, filetype="pdf")
            try:
                if not len(doc):
                    continue
                pag = doc[0]
                escala = 1500 / max(1.0, pag.rect.width)
                pix = pag.get_pixmap(matrix=pymupdf.Matrix(escala, escala), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            finally:
                doc.close()
        else:
            img = ImageOps.exif_transpose(_abrir_imagem(data)).convert("RGB")
            if img.width > 1500:
                img = img.resize((1500, max(1, int(img.height * 1500 / img.width))))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()
    return None



def _parece_irpf_sem_texto(arquivos):
    """
    PDF da declaração "impresso como imagem" (sem texto): reconhece pelo
    título/nome do arquivo (ex.: "...-IRPF-2026-2025-...", "Declaracao_IR").
    """
    for item in arquivos:
        nome = _tirar_acentos(item.get("filename", "")).upper()
        titulo = ""
        if PathLikeSuffix(item.get("filename", "")) == ".pdf" or item.get("mimetype") == "application/pdf":
            try:
                doc = pymupdf.open(stream=item["data"], filetype="pdf")
                try:
                    if any((doc[i].get_text("text") or "").strip() for i in range(min(len(doc), 2))):
                        continue  # tem texto: quem decide é a leitura pelo texto
                    titulo = _tirar_acentos(str((doc.metadata or {}).get("title") or "")).upper()
                finally:
                    doc.close()
            except Exception:
                continue
        alvo = nome + " " + titulo
        if re.search(r"IRPF|DECLARACAO.{0,3}(DE.{0,3})?(IR|IMPOSTO)\b|IMPOSTO.{0,3}DE.{0,3}RENDA|(^|[^A-Z])IR([^A-Z]|$)", alvo):
            return True
    return False



def _irpf_por_ia(arquivos):
    """Declaração sem texto (imagem): a IA lê a 1ª página. Devolve (info, tempo)."""
    jpeg = _irpf_primeira_pagina(arquivos)
    if not jpeg:
        raise RuntimeError("Não consegui abrir a declaração.")
    bruto, tempo = _groq_json_imagem(PROMPT_IRPF, jpeg, max_tokens=700)
    bruto = bruto if isinstance(bruto, dict) else {}
    fontes = []
    for f in bruto.get("fontes") or []:
        if not isinstance(f, dict):
            continue
        fontes.append({
            "nome": str(f.get("nome") or "").upper().strip(),
            "documento": re.sub(r"\D", "", str(f.get("documento") or "")),
            "anual": _numero_ir(f.get("rendimentos")),
            "previdencia": _numero_ir(f.get("previdencia")),
            "irrf_anual": _numero_ir(f.get("irrf")),
            "decimo": _numero_ir(f.get("decimo")),
        })
    def ano(v):
        m = re.search(r"(20\d{2})", str(v or ""))
        return int(m.group(1)) if m else 0
    total = _numero_ir(bruto.get("total")) or None
    info = {"exercicio": ano(bruto.get("exercicio")), "ano_calendario": ano(bruto.get("ano_calendario")),
            "cpf_titular": re.sub(r"\D", "", str(bruto.get("cpf_titular") or "")),
            "fontes": fontes, "total": total, "lido_por_ia": True}
    return info, tempo



def _resposta_irpf(info, arquivos, inicio, modelo, tempo_api=0.0):
    try:
        dados = _normalizar_irpf(info, fora_prazo=request.form.get("fora_prazo") == "1")
    except ValueError as e:
        txt = str(e)
        code = "IRPF_FORA_PERIODO" if txt.startswith("IRPF_FORA_PERIODO|") else ""
        print(f"[IRPF] recusada: {txt.split('|', 1)[-1]}", flush=True)
        return jsonify({"ok": False, "code": code, "error": txt.split("|", 1)[-1]}), 422
    if info.get("lido_por_ia"):
        # Confere: a soma das fontes tem que bater com o TOTAL da seção.
        soma = round(sum(f.get("anual", 0) for f in info.get("fontes") or []), 2)
        if not info.get("total") or abs(soma - float(info["total"])) > 0.05:
            dados["campos_incertos"] = list(dict.fromkeys((dados.get("campos_incertos") or []) + ["bruta", "liquido_original"]))
    _registrar_uso("renda")
    print(f"[IRPF] exercício {info['exercicio']}, {len(dados['fontes_ir'])} fonte(s) PJ ({modelo})", flush=True)
    return jsonify({
        "ok": True, "dados": dados, "modelo": modelo, "arquivos_lidos": len(arquivos),
        "tempo_api": round(tempo_api, 3), "tempo_total": round(time.perf_counter() - inicio, 3),
        "competencias_validas": _holerite_competencias_validas(),
    })



def _exercicios_irpf_aceitos(hoje=None):
    """Exercício do ano atual; de janeiro a maio (prazo de entrega) também o anterior."""
    hoje = hoje or datetime.now(FUSO_BR).date()
    aceitos = [hoje.year]
    if hoje.month <= 5:
        aceitos.append(hoje.year - 1)
    return aceitos



def _normalizar_irpf(info, hoje=None, fora_prazo=False):
    aceitos = _exercicios_irpf_aceitos(hoje)
    usou_fora_prazo = info.get("exercicio") not in aceitos and fora_prazo
    if info.get("exercicio") not in aceitos and not fora_prazo:
        ex = info.get("exercicio") or "não identificado"
        lista = " ou ".join(f"{a} (ano-calendário {a - 1})" for a in aceitos)
        raise ValueError(f"IRPF_FORA_PERIODO|Declaração do exercício {ex}. Agora vale a do exercício {lista}.")
    fontes = [f for f in info.get("fontes") or [] if f.get("anual", 0) > 0]
    if not fontes:
        raise ValueError(
            "A declaração não tem rendimentos tributáveis recebidos de pessoa jurídica pelo titular."
        )
    # Renda válida = salário: fonte com contribuição ao INSS ou 13º salário.
    # Outras (ex.: rendimento de cooperativa de crédito) não contam.
    de_salario = [f for f in fontes if f.get("previdencia", 0) > 0 or f.get("decimo", 0) > 0]
    if de_salario:
        fontes = de_salario
    fontes.sort(key=lambda f: f["anual"], reverse=True)
    lista = []
    for f in fontes:
        doc = f.get("documento", "")
        lista.append({
            "nome": re.sub(r"\s+", " ", f.get("nome", "")).strip(),
            "documento": doc,
            "tipo_fonte": "FÍSICA" if len(doc) == 11 else "JURÍDICA",
            "rotulo": "CPF" if len(doc) == 11 else "CNPJ",
            "anual": _holerite_valor(f["anual"]),
            "mensal": _holerite_valor(round(f["anual"] / 12, 2)),
        })
    p = lista[0]
    incertos = ["data_admissao"]
    if not p["documento"]:
        incertos.append("documento_fonte")
    return {
        "caracteristica_renda": "COMPROVADA",
        "tipo_renda": "IRPF",
        "tipo_renda_nome": f"Declaração de IR {info['exercicio']}",
        "tipo_fonte": p["tipo_fonte"],
        "documento_fonte": p["documento"],
        "documento_fonte_rotulo": p["rotulo"],
        "documento": "DECLARAÇÃO DE IMPOSTO DE RENDA",
        "cpf_funcionario": info.get("cpf_titular", ""),
        "cnpj": p["documento"] if p["rotulo"] == "CNPJ" else "",
        "numero_beneficio": "",
        "especie_beneficio": "",
        "data_admissao": "",
        "competencia": _mes_anterior(),
        "bruta": p["mensal"],
        "liquida": p["mensal"],
        "irrf": "",
        "liquido_original": p["mensal"],
        "adiantamento": "0,00",
        "descricao_adiantamento": "",
        "fontes_ir": lista,
        "exercicio": info["exercicio"],
        "ano_calendario": info["ano_calendario"],
        "campos_incertos": incertos,
        "fora_prazo": usou_fora_prazo,
    }
