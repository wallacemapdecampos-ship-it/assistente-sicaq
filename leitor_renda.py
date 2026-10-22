"""Holerite, eSocial, INSS, holerite do Estado de SP e a rota /ler-holerite."""

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
    _groq_abrir,
    _holerite_competencias_validas,
    _holerite_valor,
    _parse_data_br,
    _pdf_pagina_tem_foto,
    _sim,
    _texto_limpo,
    _tirar_acentos,
    app,
)

from contas import (  # noqa: F401
    _registrar_uso,
)

from imagem import (  # noqa: F401
    PathLikeSuffix,
    _abrir_imagem,
    _recortar_documento,
    _texto_de_lado,
)

from pasta import (  # noqa: F401
    _avaliar_pasta,
)

from leitor_cnh import (  # noqa: F401
    _texto_pdf_arquivos,
)

from leitor_identidade import (  # noqa: F401
    _cpf_valido,
)

from leitor_irpf import (  # noqa: F401
    _irpf_do_pdf,
    _irpf_por_ia,
    _parece_irpf_sem_texto,
    _resposta_irpf,
)



# ============================================================
# HOLERITE / RENDA FORMAL
# ============================================================

def _sem_acentos(txt):
    txt = str(txt or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", txt)
        if unicodedata.category(c) != "Mn"
    )



def _holerite_prompt():
    atual, anterior, retrasado = _holerite_competencias_validas()
    return f"""
Leia este comprovante de renda brasileiro (HOLERITE / CONTRACHEQUE, recibo
do eSocial de empregado doméstico ou EXTRATO DE PAGAMENTO DE BENEFÍCIO DO
INSS) e devolva SOMENTE JSON válido.
A imagem pode ser FOTO (papel torto/dobrado): siga cada linha da tabela com
cuidado, sem pular para a linha de cima ou de baixo.

Chaves, NESTA ORDEM:
origem, competencia, periodo, empregador_tipo, empregador_documento, cpf_funcionario,
numero_beneficio, especie_beneficio, data_admissao, itens, total_proventos,
total_descontos, liquido_original, folha_estado_sp, campos_incertos.

Regras:
- origem: "IRPF" se for DECLARAÇÃO DE AJUSTE ANUAL do Imposto de Renda
  (Receita Federal); nesse caso as outras chaves podem ficar vazias.
  "INSS" se for extrato/demonstrativo de pagamento de benefício do
  INSS (aposentadoria, pensão por morte, auxílio etc.); "ESOCIAL" se for
  recibo de empregado DOMÉSTICO do eSocial (empregador pessoa física);
  "HOLERITE" nos demais casos.
- competencia: mês/ano do comprovante em MM/AAAA. Muitos holerites não
  escrevem "competência": mostram só o PERÍODO (ex.: "01/09/2026 a
  30/09/2026" -> 09/2026) ou "Mês/Ano", "Referência", "Folha de SETEMBRO/2026".
  NUNCA use a data de admissão nem a data de pagamento/assinatura.
- periodo: o texto do período/mês de referência exatamente como impresso
  (ex.: "01/09/2026 a 30/09/2026"); vazio se não houver.
- empregador_tipo: o rótulo do número de inscrição do EMPREGADOR como está
  escrito: "CNPJ", "CEI", "CAEPF", "CNO" ou "CPF". No INSS, deixe vazio.
- empregador_documento: o número de inscrição do EMPREGADOR exatamente como
  está impresso, com pontos, barra e traço. No INSS, deixe vazio.
- cpf_funcionario: CPF do funcionário/beneficiário, se aparecer.
- numero_beneficio: só no INSS, o número do benefício (NB).
- especie_beneficio: só no INSS, a espécie (ex.: "21 - PENSÃO POR MORTE").
- data_admissao: campo "ADMISSÃO", em DD/MM/AAAA com ano de 4 dígitos.
  No INSS, use a data de início do benefício (DIB), se aparecer.
  Não confunda com data de pagamento, CEP, CTPS ou outras datas.
- itens: TODAS as linhas de eventos da tabela, na ordem, cada uma como uma
  lista curta [descrição, referência, provento, desconto], ex.:
  [["SALARIO BASE", "30,00", "2000,00", "0,00"],
   ["IMPOSTO DE RENDA", "27,50", "0,00", "1928,51"]].
  referência = coluna REFERÊNCIA/QTDE (dias, horas, %); NÃO é valor em
  dinheiro. provento = coluna PROVENTOS/VENCIMENTOS/CRÉDITOS; desconto =
  coluna DESCONTOS/DÉBITOS. Atenção no IR e no INSS: a % (ex.: 27,50; 7,50)
  é a referência; o valor descontado é o número da coluna DESCONTOS.
  Alguns holerites REPETEM o valor na Referência e na coluna de descontos (ex.:
  "DESC.ADIANT.SALARIAL  1.106,19 ... 1.106,19"): nesse caso o desconto é esse
  valor; não deixe desconto 0,00 só porque o número também está na Referência.
  Se o texto trouxer marcações [REF: ...], [PROVENTO: ...] e [DESCONTO: ...],
  use exatamente essas colunas. Copie a descrição completa (ex.:
  ADIANTAMENTO SALARIAL, I.R.R.F., EMPRÉSTIMO CONSIGNADO).
  No INSS, cada rubrica é uma linha: crédito (ex.: VALOR TOTAL DE MR DO
  PERÍODO) em provento e débito (consignação, empréstimo, IR) em desconto.
- total_proventos: linha TOTAL da coluna de proventos (TOTAL DE VENCIMENTOS).
  No INSS, o total de créditos (valor bruto).
- total_descontos: linha TOTAL da coluna de descontos.
- liquido_original: TOTAL LÍQUIDO / LÍQUIDO A RECEBER (no INSS, VALOR LÍQUIDO).
  Não use salário base, base INSS, base FGTS ou "Salário Contribuição" como
  total.
- Valores monetários no formato brasileiro, ex.: "2304,21".
- folha_estado_sp: true SOMENTE se o cabeçalho for "GOVERNO DO ESTADO DE SÃO
  PAULO" com "Departamento de Despesa de Pessoal do Estado" (Demonstrativo
  de Pagamento do servidor estadual de SP); senão false. Esse holerite não
  tem CNPJ: deixe empregador_tipo e empregador_documento vazios (Reg.
  Sistema, PV, Reg. Geral e U.C.D. NÃO são CNPJ).
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



_VALOR_BR = re.compile(r"^-?\d{1,3}(?:\.\d{3})*,\d{2}-?$")



def _linhas_holerite_anotadas(palavras):
    """
    Monta as linhas do PDF digital pela posição das palavras e marca cada
    valor com a coluna do cabeçalho mais próxima: [REF: 27,50] [DESCONTO:
    1.928,51]. Assim a IA não confunde a coluna Referência (%, horas) com
    Vencimentos/Descontos. palavras: (x0, y0, x1, y1, texto, ...).
    """
    ws = sorted((w for w in palavras if str(w[4]).strip()), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    linhas = []
    for w in ws:
        yc, alt = (w[1] + w[3]) / 2, max(1.0, w[3] - w[1])
        if linhas and abs(linhas[-1]["y"] - yc) <= 0.45 * alt:
            linhas[-1]["ws"].append(w)
        else:
            linhas.append({"y": yc, "ws": [w]})

    def centro(w):
        return (w[0] + w[2]) / 2

    colunas, saida = None, []
    secoes = None  # layout lado a lado: (x de DESCONTOS, x de BASES)
    for ln in linhas:
        ws = sorted(ln["ws"], key=lambda w: w[0])
        textos = [_tirar_acentos(str(w[4])).upper() for w in ws]

        # Layout "R E M U N E R A Ç Õ E S | D E S C O N T O S | B A S E S":
        # proventos e descontos ficam LADO A LADO na mesma linha. Separa cada
        # linha pela posição dos títulos (senão a IA mistura as colunas).
        junto = "".join(textos)
        if sum(len(x) == 1 for x in textos) >= 10 and "DESCONTOS" in junto and \
                any(k in junto for k in ("REMUNERAC", "PROVENTOS", "VENCIMENTOS")):
            def x_de(palavra):
                for i in range(len(textos)):
                    if "".join(textos[i:i + len(palavra)]) == palavra:
                        return ws[i][0]
                return None
            x_desc = x_de("DESCONTOS")
            x_base = x_de("BASES") or float("inf")
            if x_desc is not None:
                secoes = (x_desc, x_base)
                colunas = None
                saida.append("REMUNERAÇÕES (proventos) | DESCONTOS | BASES  -- colunas lado a lado; cada linha abaixo já vem separada")
                continue
        if secoes:
            if any(k in x for x in textos for k in ("TOTAL", "LIQUIDO")):
                secoes = None
            else:
                x_desc, x_base = secoes
                g_prov = [w for w in ws if w[0] < x_desc - 2]
                g_desc = [w for w in ws if x_desc - 2 <= w[0] < x_base - 2]
                g_base = [w for w in ws if w[0] >= x_base - 2]
                # o código da verba de desconto (ex.: "401") às vezes começa um
                # pouco antes do título DESCONTOS: devolve para a coluna certa
                while g_desc and g_prov and re.fullmatch(r"\d{2,4}", str(g_prov[-1][4])) and \
                        (len(g_prov) == 1 or any(_VALOR_BR.match(str(w[4])) for w in g_prov[:-1])):
                    g_desc.insert(0, g_prov.pop())
                partes_linha = []
                for rotulo, grupo in (("PROVENTO", g_prov), ("DESCONTO", g_desc), ("BASE", g_base)):
                    if not grupo:
                        continue
                    valores = [w for w in grupo if _VALOR_BR.match(str(w[4]))]
                    txt = []
                    for w in grupo:
                        if rotulo != "BASE" and valores and w is valores[-1]:
                            txt.append(f"[{rotulo}: {w[4]}]")
                        elif rotulo != "BASE" and _VALOR_BR.match(str(w[4])):
                            txt.append(f"[REF: {w[4]}]")
                        else:
                            txt.append(str(w[4]))
                    partes_linha.append((rotulo, " ".join(txt)))
                for rotulo, txt in partes_linha:
                    saida.append(txt if rotulo != "BASE" else "(base) " + txt)
                continue
        prov = next((w for w, x in zip(ws, textos) if x.startswith(("VENCIMENTO", "PROVENTO", "CREDITO"))), None)
        desc = next((w for w, x in zip(ws, textos) if x.startswith(("DESCONTO", "DEBITO"))), None)
        if prov and desc:
            ref = next((w for w, x in zip(ws, textos) if x.startswith(("REFER", "REF.", "QTD", "QUANT"))), None)
            colunas = [("PROVENTO", centro(prov)), ("DESCONTO", centro(desc))]
            if ref:
                colunas.append(("REF", centro(ref)))
            saida.append(" ".join(str(w[4]) for w in ws))
            continue
        if any("LIQUIDO" in x for x in textos):
            colunas = None  # depois do líquido vêm bases (FGTS, IRRF...), não valores
        valores = [w for w in ws if _VALOR_BR.match(str(w[4]))]
        if not colunas or len(valores) > 3:
            saida.append(" ".join(str(w[4]) for w in ws))
            continue
        partes = []
        for w in ws:
            if _VALOR_BR.match(str(w[4])):
                nome = min(colunas, key=lambda c: abs(c[1] - centro(w)))[0]
                partes.append(f"[{nome}: {w[4]}]")
            else:
                partes.append(str(w[4]))
        saida.append(" ".join(partes))
    return "\n".join(saida)



def _pdf_texto_holerite(data):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            partes = []
            for i in range(min(len(doc), 8)):
                try:
                    texto = _linhas_holerite_anotadas(doc[i].get_text("words") or [])
                except Exception:
                    texto = ""
                if not texto.strip():
                    texto = doc[i].get_text("text") or ""
                if texto.strip():
                    partes.append(f"\n--- PÁGINA {i+1} ---\n{texto}")
            return "".join(partes).strip()
        finally:
            doc.close()
    except Exception:
        return ""



NOTA_ROTACOES_HOLERITE = (
    "ATENÇÃO: as 2 imagens a seguir são o MESMO holerite (foto tirada de lado), "
    "giradas em sentidos opostos. Leia SÓ a imagem em que o texto está na posição "
    "normal de leitura e siga cada linha da tabela até as colunas de valores."
)



def _jpeg_holerite(img, max_lado=1600, qualidade=90):
    if max(img.size) > max_lado:
        escala = max_lado / max(img.size)
        img = img.resize(
            (max(1, int(img.width * escala)), max(1, int(img.height * escala))),
            Image.Resampling.LANCZOS,
        )
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=qualidade, optimize=True)
    return out.getvalue()



def _foto_holerite(img):
    """Foto do holerite: recorta o papel; de lado, vai nas 2 orientações."""
    img, _ = _recortar_documento(img)
    lado = _texto_de_lado(img)
    if lado:
        print("[HOLERITE] foto de lado: enviando 2 orientações", flush=True)
        return [_jpeg_holerite(img.rotate(-90, expand=True)), _jpeg_holerite(img.rotate(90, expand=True))], NOTA_ROTACOES_HOLERITE
    return [_jpeg_holerite(img)], ""



def _imagem_holerite(data, filename, mimetype):
    """Devolve (lista de JPEGs, nota para a IA)."""
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if not is_pdf:
        # Foto: corrige a rotação do celular e recorta o papel.
        return _foto_holerite(_abrir_imagem(data))

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        # PDF que é só uma foto (ex.: foto do celular salva em PDF): trata
        # como foto (recorte do papel e correção se estiver de lado).
        if len(doc) == 1 and _pdf_pagina_tem_foto(doc[0]):
            pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(3.0, 3.0), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").copy()
            return _foto_holerite(img)

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
        return [out.getvalue()], ""
    finally:
        doc.close()



PROMPT_HOLERITE_TABELA = """
SEGUNDA CONFERÊNCIA deste comprovante de renda: as contas da primeira leitura
NÃO fecharam (total de vencimentos - total de descontos tem que dar o líquido).
Releia SÓ a tabela de eventos e os totais, com muita atenção às colunas, e
devolva SOMENTE JSON: {"itens": [[descrição, referência, provento, desconto], ...],
"total_proventos": "...", "total_descontos": "...", "liquido_original": "..."}.

Regras:
- Inclua TODAS as linhas da tabela, uma por evento, na ordem, sem pular
  nenhuma (atenção às últimas linhas, ex.: contribuições, adiantamento,
  empréstimo, pensão).
- referência = coluna REFERÊNCIA/QTDE (dias, horas, %). provento = coluna
  VENCIMENTOS/PROVENTOS. desconto = coluna DESCONTOS.
- Alguns holerites REPETEM o mesmo número na Referência e na coluna de
  descontos (ex.: "DESC.ADIANT.SALARIAL 1.106,19 ... 1.106,19"): o desconto é
  esse valor, não 0,00.
- Confira: a soma dos descontos tem que dar o TOTAL DE DESCONTOS impresso, e a
  soma dos proventos o TOTAL DE VENCIMENTOS. Se não der, procure a linha que
  faltou ou o valor que caiu na coluna errada.
- Valores no formato brasileiro, ex.: "1106,19".
""".strip()



def _groq_holerite_call(arquivos, prompt=None, esperar_limite=True):
    """
    Lê 1 ou 2 arquivos de holerite na mesma chamada.
    Se houver dois meses diferentes, o prompt escolhe a competência
    válida mais recente. Nunca soma valores de meses distintos.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    conteudo = [
        {"type": "text", "text": prompt or _holerite_prompt()}
    ]

    for idx, item in enumerate(arquivos[:2], start=1):
        data = item["data"]
        filename = item["filename"]
        mimetype = item["mimetype"]

        texto_pdf = ""
        if PathLikeSuffix(filename) == ".pdf" or mimetype == "application/pdf":
            texto_pdf = _pdf_texto_holerite(data)

        texto_pdf = _compactar_texto_holerite(texto_pdf, 20000)
        # PDF "impresso" do navegador/CamScanner: tem só cabeçalho/rodapé em
        # texto (data, URL) e o holerite é imagem. Só usa o texto se ele
        # tiver os valores do holerite; senão manda a imagem.
        valores_txt = len(re.findall(r"\d{1,3}(?:\.\d{3})*,\d{2}", texto_pdf))
        if len(texto_pdf) >= 120 and valores_txt >= 3:
            # Limita cada arquivo para não desperdiçar tokens no plano gratuito.
            conteudo.append({
                "type": "text",
                "text": (
                    f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====\n"
                    + _compactar_texto_holerite(texto_pdf)
                ),
            })
        else:
            imagens, nota_img = _imagem_holerite(data, filename, mimetype)
            conteudo.append({
                "type": "text",
                "text": f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====" + (f"\n{nota_img}" if nota_img else ""),
            })
            for imagem in imagens:
                b64 = base64.b64encode(imagem).decode("ascii")
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
            # Plano gratuito: no máximo 1.000 tokens de resposta por minuto
            # (OTPM). As linhas vão em formato curto para caber.
            "max_completion_tokens": 950,
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
            body = _groq_abrir(req, esperar_limite=esperar_limite)
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



def _cnpj_valido(cnpj):
    """Confere os 2 dígitos verificadores do CNPJ."""
    c = re.sub(r"\D", "", str(cnpj or ""))
    if len(c) != 14 or len(set(c)) == 1:
        return False
    def dv(base, pesos):
        resto = sum(int(a) * b for a, b in zip(base, pesos)) % 11
        return "0" if resto < 2 else str(11 - resto)
    p1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = dv(c[:12], p1)
    d2 = dv(c[:12] + d1, [6] + p1)
    return c[12:] == d1 + d2



def _holerite_itens(itens):
    """
    Normaliza as linhas transcritas. Aceita o formato curto
    [descricao, provento, desconto] e o antigo {descricao, provento, desconto}.
    """
    saida = []
    for i in itens if isinstance(itens, list) else []:
        if isinstance(i, (list, tuple)):
            if len(i) >= 4:  # [descrição, referência, provento, desconto]
                i = {"descricao": i[0], "referencia": i[1], "provento": i[2], "desconto": i[3]}
            else:            # formato antigo [descrição, provento, desconto]
                i = {"descricao": i[0] if len(i) > 0 else "",
                     "provento": i[1] if len(i) > 1 else 0,
                     "desconto": i[2] if len(i) > 2 else 0}
        if not isinstance(i, dict):
            continue
        desc = _tirar_acentos(str(i.get("descricao") or "")).upper().strip()
        prov = _holerite_numero(i.get("provento"))
        descn = _holerite_numero(i.get("desconto"))
        if desc or prov or descn:
            saida.append({"descricao": desc, "provento": prov, "desconto": descn,
                          "referencia": _holerite_numero(i.get("referencia"))})
    return saida



def _eh_adiantamento_salarial(descricao):
    d = _tirar_acentos(str(descricao or "")).upper()
    if not re.search(r"ADIANT|ANTECIP|VALE\s*SAL|ADTO|ADT\.", d):
        return False
    # Não é adiantamento do salário mensal:
    if re.search(r"FERIAS|13|DECIMO|EMPREST|CONSIG|TRANSP|REFEI|ALIMENT|PENS|INSS|IRRF", d):
        return False
    # Sobras e acertos do adiantamento não entram (ex.: "DESCONTO TROCO
    # ADTO. SAL." 0,78): só o desconto do adiantamento em si.
    if re.search(r"TROCO|DIFEREN|\bDIF\b|ARREDOND|SALDO|ESTORNO|DEVOL|SOBRA|REEMB|RESTITU", d):
        return False
    return True



def _eh_irrf(descricao):
    d = _tirar_acentos(str(descricao or "")).upper()
    return bool(re.search(r"\bIRRF\b|I\.R\.R\.F|IMPOSTO\s+DE\s+RENDA|\bI\.?R\.?\s*FONTE", d))



CNPJ_INSS = "29979036000140"



# Demonstrativo de Pagamento do Governo do Estado de São Paulo
# (Departamento de Despesa de Pessoal do Estado): não traz CNPJ impresso,
# então o SICAQ recebe este CNPJ fixo.
CNPJ_ESTADO_SP = "46379400000150"



TIPOS_RENDA = {
    "HOLERITE_CNPJ": "Holerite (CNPJ)",
    "HOLERITE_ESTADO_SP": "Holerite Governo do Estado de SP (CNPJ fixo)",
    "HOLERITE_CEI": "Holerite (CEI)",
    "HOLERITE_CAEPF": "Holerite (CAEPF)",
    "HOLERITE_CNO": "Holerite (CNO)",
    "ESOCIAL": "eSocial doméstico (CPF do patrão)",
    "INSS": "Benefício do INSS",
}



def _fonte_renda(r):
    """
    Decide, por regra fixa, como a renda entra no SICAQ:
    - holerite com CNPJ: JURÍDICA + CNPJ do empregador;
    - holerite com CEI, CAEPF ou CNO: FÍSICA + CPF do próprio cliente;
    - eSocial doméstico: FÍSICA + CPF do patrão;
    - benefício do INSS: JURÍDICA + CNPJ do INSS.
    Devolve dict com tipo, tipo_fonte, documento_fonte, rotulo, documento,
    cpf_funcionario e incertos.
    """
    origem = _texto_limpo(r.get("origem"))
    rotulo = _texto_limpo(r.get("empregador_tipo"))
    impresso = str(r.get("empregador_documento") or r.get("cnpj") or "").strip()
    dig = re.sub(r"\D", "", impresso)
    cpf_func = re.sub(r"\D", "", str(r.get("cpf_funcionario") or ""))
    cpf_func = cpf_func if _cpf_valido(cpf_func) else ""

    def saida(tipo, fonte, documento, rot, comprovante, incertos=()):
        return {
            "tipo": tipo, "tipo_fonte": fonte, "documento_fonte": documento,
            "rotulo": rot, "documento": comprovante,
            "cpf_funcionario": cpf_func, "incertos": list(incertos),
        }

    if "INSS" in origem:
        return saida("INSS", "JURÍDICA", CNPJ_INSS, "CNPJ", "EXTRATO DE BENEFÍCIO PREVIDENCIÁRIO")

    # Holerite do Governo do Estado de SP: sem CNPJ impresso. Usa o CNPJ fixo
    # e ignora qualquer número que a IA tenha pego (ex.: Reg. Sistema / PV).
    # Para o site e a extensão continua sendo um holerite com CNPJ.
    if _sim(r.get("folha_estado_sp")):
        fonte = saida("HOLERITE_CNPJ", "JURÍDICA", CNPJ_ESTADO_SP, "CNPJ", "CONTRACHEQUE/HOLLERITH")
        fonte["nome"] = TIPOS_RENDA["HOLERITE_ESTADO_SP"]
        return fonte

    formato_caepf = re.fullmatch(r"\d{3}\.\d{3}\.\d{3}/\d{3}-\d{2}", impresso)
    if "CAEPF" in rotulo or (len(dig) == 14 and formato_caepf and not _cnpj_valido(dig)):
        tipo = "HOLERITE_CAEPF"
    elif "CNO" in rotulo:
        tipo = "HOLERITE_CNO"
    elif "CEI" in rotulo or len(dig) == 12:
        tipo = "HOLERITE_CEI"
    elif "ESOCIAL" in origem or "CPF" in rotulo or len(dig) == 11:
        tipo = "ESOCIAL"
    else:
        tipo = "HOLERITE_CNPJ"

    if tipo == "HOLERITE_CNPJ":
        return saida(tipo, "JURÍDICA", dig, "CNPJ", "CONTRACHEQUE/HOLLERITH",
                     [] if _cnpj_valido(dig) else ["documento_fonte"])
    if tipo == "ESOCIAL":
        # CPF do patrão: precisa ser válido e diferente do CPF do funcionário.
        ok = _cpf_valido(dig) and dig != cpf_func
        return saida(tipo, "FÍSICA", dig, "CPF", "CONTRACHEQUE/HOLLERITH",
                     [] if ok else ["documento_fonte"])
    # CEI / CAEPF / CNO: o SICAQ recebe o CPF do próprio cliente.
    return saida(tipo, "FÍSICA", cpf_func, "CPF", "CONTRACHEQUE/HOLLERITH",
                 [] if cpf_func else ["documento_fonte"])



def _holerite_estado_sp_pdf(arquivos):
    """
    True quando o PDF é o Demonstrativo de Pagamento do Governo do Estado de
    SP (cabeçalho "GOVERNO DO ESTADO DE SÃO PAULO" + "Departamento de Despesa
    de Pessoal do Estado"). Foto ou scan sem texto: decide a IA.
    """
    try:
        txt = _tirar_acentos(_texto_pdf_arquivos(arquivos)).upper()
    except Exception:
        return False
    txt = re.sub(r"\s+", " ", txt)
    return "GOVERNO DO ESTADO DE SAO PAULO" in txt and "DESPESA DE PESSOAL DO ESTADO" in txt



def _subconjunto_que_soma(itens, alvo, chave="referencia", max_itens=16):
    """
    Linhas cujo valor de `chave` soma exatamente `alvo` (±0,05). Devolve a
    combinação com MENOS linhas, só se for a única desse tamanho; senão None.
    """
    itens = itens[:max_itens]
    from itertools import combinations
    for tam in range(1, len(itens) + 1):
        achados = [c for c in combinations(itens, tam)
                   if abs(round(sum(i[chave] for i in c), 2) - alvo) <= 0.05]
        if len(achados) == 1:
            return list(achados[0])
        if len(achados) > 1:
            return None
    return None



def _normalizar_holerite(resultado, fora_prazo=False):
    resultado = dict(resultado or {})
    fonte = _fonte_renda(resultado)
    competencia = _holerite_competencia_mm_aaaa(resultado.get("competencia"))
    validas = _holerite_competencias_validas()
    if competencia not in validas:
        # Holerite que só mostra o período ("01/09/2026 a 30/09/2026"):
        # a competência é o mês da data final do período.
        datas = re.findall(r"\b\d{1,2}\s*/\s*(\d{1,2})\s*/\s*((?:19|20)\d{2})\b",
                           str(resultado.get("periodo") or "") + " " + str(resultado.get("competencia") or ""))
        if datas:
            mes, ano = datas[-1]
            if 1 <= int(mes) <= 12:
                pelo_periodo = f"{int(mes):02d}/{ano}"
                if pelo_periodo in validas or not competencia:
                    competencia = pelo_periodo

    usou_fora_prazo = False
    if competencia not in validas and fora_prazo:
        # O analista escolheu usar mesmo fora do prazo.
        usou_fora_prazo = True
        if not re.fullmatch(r"\d{2}/\d{4}", competencia or ""):
            competencia = ""
    elif competencia not in validas:
        identificada = competencia or "não identificada"
        raise ValueError(
            f"HOLERITE_FORA_PERIODO|Competência identificada: {identificada}. "
            f"Período aceito: {validas[0]}, {validas[1]} e {validas[2]}."
        )

    incertos = [str(x) for x in (resultado.get("campos_incertos") or [])]
    bruta = _holerite_numero(resultado.get("total_proventos") or resultado.get("bruta"))
    total_desc = _holerite_numero(resultado.get("total_descontos"))
    liquido_original = _holerite_numero(
        resultado.get("liquido_original") or resultado.get("liquido") or resultado.get("valor_liquido")
        or resultado.get("total_liquido") or resultado.get("liquido_a_receber"))
    faltou = [k for k in ("total_proventos", "total_descontos", "liquido_original") if not resultado.get(k)]
    if faltou:
        print(f"[HOLERITE] IA não devolveu: {', '.join(faltou)}", flush=True)
    adiantamento = _holerite_numero(resultado.get("adiantamento_salarial_total"))
    descricao_ad = str(resultado.get("descricao_adiantamento") or "").strip().upper()
    irrf = _holerite_numero(resultado.get("irrf"))

    # --- Conferências a partir das linhas transcritas (sem gastar IA) ---
    itens = _holerite_itens(resultado.get("itens"))
    if itens:
        soma_prov = round(sum(i["provento"] for i in itens), 2)
        soma_desc = round(sum(i["desconto"] for i in itens), 2)
        # Totais: se não vieram, usa as somas das linhas.
        if not bruta:
            bruta = soma_prov
        if not total_desc:
            total_desc = soma_desc

        # Bruto lido do número errado do rodapé (ex.: "Salário Contribuição"
        # no lugar do "Total Vencimentos"): se as contas não fecham com o
        # bruto lido, mas fecham com a soma dos proventos, usa a soma.
        if bruta and total_desc and liquido_original and soma_prov and \
                abs(round(bruta - total_desc, 2) - liquido_original) > 0.05 and \
                abs(round(soma_prov - total_desc, 2) - liquido_original) <= 0.05:
            print("[HOLERITE] bruto trocado pela soma dos proventos (contas fecham)", flush=True)
            bruta = soma_prov

        # Valor lido só na Referência (alguns holerites repetem o valor nas
        # duas colunas e a IA fica com a primeira): se os descontos (ou
        # proventos) não fecham com o total, procura as linhas sem valor cuja
        # referência cobre EXATAMENTE a diferença, sem ambiguidade.
        for coluna, total in (("desconto", total_desc), ("provento", bruta)):
            soma = round(sum(i[coluna] for i in itens), 2)
            falta = round(total - soma, 2)
            if not total or falta <= 0.05:
                continue
            cands = [i for i in itens if not i["provento"] and not i["desconto"] and i["referencia"] > 0.05]
            achou = _subconjunto_que_soma(cands, falta)
            if achou:
                for i in achou:
                    i[coluna] = i["referencia"]
                print(f"[HOLERITE] {len(achou)} valor(es) da Referência movido(s) para {coluna} (contas fecham)", flush=True)
        soma_prov = round(sum(i["provento"] for i in itens), 2)
        soma_desc = round(sum(i["desconto"] for i in itens), 2)

        # Linha de desconto lida com a % da coluna Referência no lugar do
        # valor (ex.: IR 27,50 no lugar de 1.928,51). Se os totais do
        # holerite fecham (bruto - descontos = líquido) e UMA só linha
        # suspeita explica a diferença, corrige essa linha e marca para conferir.
        dif = round(total_desc - soma_desc, 2)
        totais_fecham = bool(bruta and total_desc and liquido_original) and \
            abs(round(bruta - total_desc, 2) - liquido_original) <= 0.05
        if abs(dif) > 0.05 and totais_fecham:
            # Suspeita = valor com cara de % (até 30) numa linha de IR/INSS ou
            # igual à própria referência.
            suspeitas = [
                i for i in itens if 0 < i["desconto"] <= 30 and (
                    _eh_irrf(i["descricao"]) or "INSS" in i["descricao"].replace(".", "")
                    or (i["referencia"] and abs(i["desconto"] - i["referencia"]) < 0.005)
                )
            ]
            if len(suspeitas) == 1 and suspeitas[0]["desconto"] + dif > 0:
                linha = suspeitas[0]
                linha["desconto"] = round(linha["desconto"] + dif, 2)
                soma_desc = round(sum(i["desconto"] for i in itens), 2)
                print("[HOLERITE] linha de desconto corrigida pela soma dos descontos", flush=True)
                if _eh_irrf(linha["descricao"]):
                    incertos += ["irrf"]

        # Adiantamento e IRRF calculados pelas linhas (coluna DESCONTOS).
        # Só a coluna DESCONTOS: o adiantamento volta ao líquido porque foi
        # descontado; linha de provento com "adiant." não é esse desconto.
        linhas_ad = [i for i in itens if i["desconto"] > 0 and _eh_adiantamento_salarial(i["descricao"])]
        if linhas_ad:
            adiantamento = round(sum(i["desconto"] for i in linhas_ad), 2)
            descricao_ad = " + ".join(i["descricao"] for i in linhas_ad)
        else:
            adiantamento, descricao_ad = 0.0, ""
        linhas_ir = [i for i in itens if _eh_irrf(i["descricao"])]
        if linhas_ir:
            irrf = round(sum(i["desconto"] for i in linhas_ir), 2)
        elif not _holerite_numero(resultado.get("irrf")):
            irrf = 0.0
        if abs(soma_prov - bruta) > 0.05:
            incertos += ["bruta"]
        if abs(soma_desc - total_desc) > 0.05:
            # Alguma linha de desconto foi lida errado: o IR e o adiantamento
            # (que vêm das linhas) também precisam ser conferidos.
            incertos += ["liquido_original", "irrf", "adiantamento"]
    else:
        incertos += ["adiantamento"]

    # Líquido não veio: calcula pelos totais (bruto - descontos) e marca para conferir.
    if not liquido_original and bruta and total_desc and bruta > total_desc:
        liquido_original = round(bruta - total_desc, 2)
        incertos += ["liquido_original"]
        print("[HOLERITE] líquido calculado pelos totais (a IA não leu)", flush=True)

    # Bruto - descontos tem de dar o líquido.
    # Sem os 3 valores não dá para conferir: marca para revisar.
    if bruta and total_desc and liquido_original:
        if abs(round(bruta - total_desc, 2) - liquido_original) > 0.05:
            incertos += ["bruta", "liquido_original"]
    else:
        incertos += ["bruta", "liquido_original"]

    # CNPJ/CPF da fonte pagadora conferido pelos dígitos verificadores.
    incertos += fonte["incertos"]

    # Admissão: data válida, ano de 4 dígitos e antes da competência.
    admissao_txt = str(resultado.get("data_admissao") or "").strip()
    admissao = _parse_data_br(admissao_txt)
    try:
        mes_c, ano_c = (int(x) for x in competencia.split("/"))
        fim_comp = date(ano_c + (mes_c == 12), (mes_c % 12) + 1, 1)
    except Exception:
        fim_comp = None
    if not admissao or admissao.year < 1950 or (fim_comp and admissao >= fim_comp):
        incertos.append("data_admissao")
    else:
        admissao_txt = f"{admissao.day:02d}/{admissao.month:02d}/{admissao.year:04d}"

    # Regra SICAQ: somente adiantamento salarial volta ao líquido.
    liquida_sicaq = liquido_original + adiantamento

    return {
        "caracteristica_renda": "COMPROVADA",
        "tipo_renda": fonte["tipo"],
        "tipo_renda_nome": fonte.get("nome") or TIPOS_RENDA.get(fonte["tipo"], ""),
        "tipo_fonte": fonte["tipo_fonte"],
        "documento_fonte": fonte["documento_fonte"],
        "documento_fonte_rotulo": fonte["rotulo"],
        "documento": fonte["documento"],
        "cpf_funcionario": fonte["cpf_funcionario"],
        "cnpj": fonte["documento_fonte"] if fonte["rotulo"] == "CNPJ" else "",
        "numero_beneficio": re.sub(r"\D", "", str(resultado.get("numero_beneficio") or ""))
        if fonte["tipo"] == "INSS" else "",
        "especie_beneficio": str(resultado.get("especie_beneficio") or "").strip().upper()
        if fonte["tipo"] == "INSS" else "",
        "data_admissao": admissao_txt,
        "competencia": competencia,
        "bruta": _holerite_valor(bruta),
        "liquida": _holerite_valor(liquida_sicaq),
        "irrf": _holerite_valor(irrf),
        "liquido_original": _holerite_valor(liquido_original),
        "adiantamento": _holerite_valor(adiantamento),
        "descricao_adiantamento": descricao_ad,
        "campos_incertos": sorted(set(incertos + (["competencia"] if usou_fora_prazo else []))),
        "fora_prazo": usou_fora_prazo,
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
    # Botão "usar mesmo fora do prazo" do site.
    fora_prazo = request.form.get("fora_prazo") == "1"

    # Declaração de Imposto de Renda (PDF da Receita): lida pelo texto, sem IA.
    irpf = None
    try:
        irpf = _irpf_do_pdf(arquivos)
    except Exception as e:
        print(f"[IRPF] leitura falhou ({type(e).__name__})", flush=True)
    if irpf is not None:
        return _resposta_irpf(irpf, arquivos, inicio, "texto do PDF")

    # Declaração "impressa como imagem" (sem texto): a IA lê a 1ª página.
    try:
        if _parece_irpf_sem_texto(arquivos):
            info, tempo_ir = _irpf_por_ia(arquivos)
            if info.get("exercicio") or info.get("fontes"):
                return _resposta_irpf(info, arquivos, inicio, "IA (declaração em imagem)", tempo_ir)
            print("[IRPF] nome parecia IR, mas não é declaração: lendo como holerite", flush=True)
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        print(f"[IRPF] leitura por imagem falhou ({type(e).__name__}: {e})", flush=True)

    # PDF tipo "pasta" com o holerite fora das páginas lidas: não gasta IA.
    bloqueio, aviso_pasta = _avaliar_pasta(arquivos, "renda")
    if bloqueio:
        print("[PASTA] holerite fora das páginas lidas", flush=True)
        return jsonify({"ok": False, "code": "PASTA", "error": bloqueio}), 422

    try:
        resultado, modelo, tempo_api = _groq_holerite_call(arquivos)
        # PDF com texto: confere o cabeçalho do Estado de SP sem depender da IA.
        if str((resultado or {}).get("origem") or "").upper().strip() == "IRPF":
            # A IA viu que é declaração de IR: lê com a regra própria dela.
            info, tempo_ir = _irpf_por_ia(arquivos)
            return _resposta_irpf(info, arquivos, inicio, "IA (declaração em imagem)", tempo_api + tempo_ir)
        if _holerite_estado_sp_pdf(arquivos):
            resultado = {**(resultado or {}), "folha_estado_sp": True}
        dados = _normalizar_holerite(resultado, fora_prazo=fora_prazo)

        # Segunda olhada: as contas não fecharam -> relê só a tabela de valores
        # e fica com a leitura que fecha (só nesses casos; 1 leitura a mais).
        if any(c in dados.get("campos_incertos", []) for c in ("bruta", "liquido_original", "adiantamento", "irrf")):
            try:
                tabela, _, tempo2 = _groq_holerite_call(arquivos, PROMPT_HOLERITE_TABELA, esperar_limite=False)
                tempo_api += tempo2
                if isinstance(tabela, dict) and tabela.get("itens"):
                    nova = {**resultado, **{k: v for k, v in tabela.items()
                                            if k in ("itens", "total_proventos", "total_descontos", "liquido_original") and v}}
                    dados2 = _normalizar_holerite(nova, fora_prazo=fora_prazo)
                    ruins = lambda d: sum(c in d.get("campos_incertos", []) for c in ("bruta", "liquido_original", "adiantamento", "irrf"))
                    if ruins(dados2) < ruins(dados):
                        print("[HOLERITE] segunda olhada fechou as contas: usando a releitura", flush=True)
                        dados = dados2
                    else:
                        print("[HOLERITE] segunda olhada não melhorou: mantendo a 1ª leitura", flush=True)
            except GroqLimite:
                print("[HOLERITE] segunda olhada pulada (limite da Groq)", flush=True)
            except Exception as e:
                print(f"[HOLERITE] segunda olhada falhou ({type(e).__name__})", flush=True)
        _registrar_uso("renda")

        return jsonify({
            "ok": True,
            "dados": dados,
            "aviso_pasta": aviso_pasta,
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
