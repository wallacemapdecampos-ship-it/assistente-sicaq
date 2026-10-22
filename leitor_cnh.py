"""CNH: CNH Digital do gov.br, CNH antiga, MRZ (sem IA) e conferências da CNH."""

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
    _groq_abrir,
    _parse_data_br,
    _tirar_acentos,
)

from imagem import (  # noqa: F401
    PathLikeSuffix,
    _pdf_paineis,
)



# ============================================================
# CNH DIGITAL DO GOV.BR (PDF "CNH Digital" da Senatran)
# Modelo mais comum: sempre o mesmo layout, com 3 painéis (frente, verso e
# MRZ). A MRZ é lida aqui mesmo, sem IA (letra OCR-B fixa, conferida pelos
# dígitos verificadores), e a IA recebe só os campos da frente + a MRZ,
# com as letras bem maiores.
# ============================================================

NOTA_CNH_DIGITAL = (
    "ATENÇÃO: esta imagem é uma CNH Digital (PDF do gov.br). Em cima estão os "
    "campos da frente do cartão; embaixo, a MRZ (3 linhas com <<<). Leia cada "
    "campo pelo seu rótulo."
)



NOTA_CNH_DIGITAL_ANTIGA = (
    "ATENÇÃO: esta imagem é uma CNH Digital do MODELO ANTIGO (PDF do gov.br). "
    "Em cima está a frente do cartão, ampliada; embaixo, a faixa do verso com "
    "\"LOCAL\" e \"DATA EMISSÃO\". Esse modelo NÃO tem naturalidade (local de "
    "nascimento): devolva naturalidade_municipio e naturalidade_uf VAZIOS. "
    "\"LOCAL\" é onde a CNH foi emitida, não é naturalidade. data_emissao = "
    "\"DATA EMISSÃO\" do verso. A frente não tem MRZ: devolva mrz vazio. "
    "FILIAÇÃO: 1ª linha = pai, linhas seguintes = mãe (o nome pode quebrar no "
    "meio de uma palavra; junte as partes)."
)



NOTA_CNH_DIGITAL_CAMPOS = (
    "ATENÇÃO: esta imagem é a frente de uma CNH Digital (PDF do gov.br), "
    "ampliada. Leia cada campo pelo seu rótulo. A MRZ já foi lida: devolva mrz vazio."
)



_MRZ_MODELOS = None



def _mrz_modelos():
    """Desenhos médios de cada caractere OCR-B (arquivo mrz_ocrb.json)."""
    global _MRZ_MODELOS
    if _MRZ_MODELOS is None:
        try:
            caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrz_ocrb.json")
            with open(caminho, encoding="utf-8") as f:
                bruto = json.load(f)
            chaves = list(bruto["modelos"])
            matriz = np.array([bruto["modelos"][k] for k in chaves], dtype=np.float32)
            matriz -= matriz.mean(axis=1, keepdims=True)
            matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
            _MRZ_MODELOS = (chaves, matriz, int(bruto["largura"]), int(bruto["altura"]))
        except Exception as e:
            print(f"[MRZ] modelos OCR-B indisponíveis ({type(e).__name__})", flush=True)
            _MRZ_MODELOS = ()
    return _MRZ_MODELOS



def _ler_mrz_ocrb(painel):
    """
    Lê as 2 primeiras linhas da MRZ do painel 3 da CNH Digital, sem IA.
    Devolve "linha1\\nlinha2" só se os dígitos verificadores conferirem;
    senão "".
    """
    modelos = _mrz_modelos()
    if not modelos or cv2 is None or np is None:
        return ""
    chaves, matriz, tw, th = modelos
    try:
        g = np.asarray(painel.convert("L"), dtype=np.uint8)
        H, W = g.shape
        y0 = int(H * 0.45)
        _, tinta = cv2.threshold(g[y0:], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        por_linha = (tinta > 0).sum(axis=1)

        # Faixas horizontais com texto (as linhas da MRZ).
        faixas, ini = [], None
        for y, t in enumerate(list(por_linha) + [0]):
            if t >= 3 and ini is None:
                ini = y
            elif t < 3 and ini is not None:
                faixas.append((ini, y))
                ini = None

        linhas = []
        for a, b in faixas:
            if not (0.018 * H <= b - a <= 0.07 * H):
                continue
            cols = np.where(tinta[a:b].sum(axis=0) > 0)[0]
            if len(cols) == 0 or cols[-1] - cols[0] < 0.4 * W:
                continue
            n, _, stats, _ = cv2.connectedComponentsWithStats(tinta[a:b], connectivity=8)
            comps = sorted(
                [int(stats[i][0]), int(stats[i][0] + stats[i][2])]
                for i in range(1, n)
                if stats[i][4] >= max(6, 0.01 * (b - a) ** 2)
            )
            juntos = []
            for c in comps:
                if juntos and c[0] < juntos[-1][1] - 0.5 * (c[1] - c[0]):
                    juntos[-1][1] = max(juntos[-1][1], c[1])
                else:
                    juntos.append(c)
            if len(juntos) != 30:
                continue
            centros = np.array([(c[0] + c[1]) / 2 for c in juntos])
            passo = float(np.median(np.diff(centros)))
            m = max(1, int(0.08 * (b - a)))
            topo, base = max(0, y0 + a - m), min(H, y0 + b + m)
            texto = ""
            for cx in centros:
                x1 = max(0, int(round(cx - passo / 2)))
                x2 = min(W, int(round(cx + passo / 2)))
                cel = cv2.resize(g[topo:base, x1:x2], (tw, th), interpolation=cv2.INTER_AREA)
                v = 255.0 - cel.astype(np.float32).ravel()
                v -= v.mean()
                norma = float(np.linalg.norm(v))
                if norma == 0:
                    texto += "?"
                    continue
                texto += chaves[int(np.argmax(matriz @ (v / norma)))]
            linhas.append(texto)
            if len(linhas) == 2:
                break

        if len(linhas) < 2:
            return ""
        mrz = "\n".join(linhas)
        if _numero_cnh_mrz(mrz) or _datas_mrz(mrz):
            return mrz
    except Exception as e:
        print(f"[MRZ] leitura OCR-B falhou ({type(e).__name__})", flush=True)
    return ""



def _cnh_digital_senatran(arquivos):
    """
    1 PDF "CNH Digital" do gov.br -> {"imagem": jpeg, "mrz": texto ou ""}.
    Qualquer outro arquivo -> None (segue o caminho normal).
    """
    if len(arquivos) != 1:
        return None
    a = arquivos[0]
    if PathLikeSuffix(a["filename"]) != ".pdf" and a["mimetype"] != "application/pdf":
        return None
    try:
        doc = pymupdf.open(stream=a["data"], filetype="pdf")
    except Exception:
        return None
    try:
        if len(doc) < 1:
            return None
        page = doc[0]
        if "SENATRAN" not in _tirar_acentos(page.get_text("text") or "").upper():
            return None
        paineis = _pdf_paineis(page)
        if paineis and len(paineis) == 2:
            return _cnh_digital_antiga(*paineis)
        if not paineis or len(paineis) != 3:
            return None
        frente, _verso, painel_mrz = paineis

        mrz = _ler_mrz_ocrb(painel_mrz)

        # Frente sem a faixa do mapa/número lateral. A MRZ só vai para a IA
        # quando não foi lida aqui (assim os campos ficam ainda maiores).
        W, H = frente.size
        campos = frente.crop((int(W * 0.16), int(H * 0.18), int(W * 0.99), int(H * 0.99)))
        if mrz:
            folha, nota = campos, NOTA_CNH_DIGITAL_CAMPOS
        else:
            w3, h3 = painel_mrz.size
            linhas = painel_mrz.crop((int(w3 * 0.15), int(h3 * 0.56), int(w3 * 0.85), int(h3 * 0.83)))
            gap = 14
            folha = Image.new("RGB", (max(campos.width, linhas.width), campos.height + gap + linhas.height), "white")
            folha.paste(campos, (0, 0))
            folha.paste(linhas, ((folha.width - linhas.width) // 2, campos.height + gap))
            nota = NOTA_CNH_DIGITAL
        largura, altura = folha.size
        # Amplia até ~1,6 MP (o tamanho que a IA usa): letras maiores.
        fator = min(2.0, (1_600_000 / (largura * altura)) ** 0.5)
        if fator > 1.05:
            folha = folha.resize(
                (int(largura * fator), int(altura * fator)), Image.Resampling.LANCZOS
            )
        out = io.BytesIO()
        folha.save(out, format="JPEG", quality=92)
        print(
            f"[LER] CNH Digital gov.br: {'campos' if mrz else 'campos + MRZ'} {folha.size}; "
            f"MRZ sem IA {'conferida' if mrz else 'não lida'}",
            flush=True,
        )
        return {"imagem": out.getvalue(), "mrz": mrz, "nota": nota}
    except Exception as e:
        print(f"[LER] CNH Digital não montada ({type(e).__name__}); caminho normal", flush=True)
        return None
    finally:
        doc.close()



def _cnh_digital_antiga(frente, verso):
    """
    CNH Digital do MODELO ANTIGO (2 painéis: frente e verso, sem MRZ).
    Esse modelo não tem naturalidade; a data de emissão fica no verso.
    A IA recebe a frente ampliada + só a faixa "LOCAL / DATA EMISSÃO" do verso.
    """
    W, H = frente.size
    campos = frente.crop((int(W * 0.14), int(H * 0.225), int(W * 0.99), int(H * 0.99)))
    w2, h2 = verso.size
    faixa = verso.crop((int(w2 * 0.14), int(h2 * 0.545), int(w2 * 0.97), int(h2 * 0.665)))
    gap = 18
    folha = Image.new("RGB", (max(campos.width, faixa.width), campos.height + gap + faixa.height), "white")
    folha.paste(campos, (0, 0))
    folha.paste(faixa, (0, campos.height + gap))
    largura, altura = folha.size
    fator = min(2.0, (1_600_000 / (largura * altura)) ** 0.5)
    if fator > 1.05:
        folha = folha.resize((int(largura * fator), int(altura * fator)), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    folha.save(out, format="JPEG", quality=92)
    print(f"[LER] CNH Digital gov.br MODELO ANTIGO: frente + faixa da emissão {folha.size}", flush=True)
    return {"imagem": out.getvalue(), "mrz": "", "nota": NOTA_CNH_DIGITAL_ANTIGA, "sem_naturalidade": True}



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
            body = _groq_abrir(req)
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
