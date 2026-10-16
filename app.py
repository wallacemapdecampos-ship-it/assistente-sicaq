import base64
import io
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

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import pymupdf

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


RATE_WINDOW = 60
RATE_MAX = 20
_rate = defaultdict(deque)

PROMPT = """
Leia o documento brasileiro e responda SOMENTE JSON válido com estas chaves:
tipo_documento, modelo_documento, nome, cpf, rg, data_nascimento, numero_cnh,
orgao_emissor, uf_emissao, data_primeira_habilitacao, data_emissao, validade,
sexo, nacionalidade, naturalidade_municipio, naturalidade_uf, filiacao_1,
filiacao_2, campos_incertos.

Regras:
- tipo_documento: CNH, RG ou CIN. modelo_documento: ANTIGO, NOVO ou vazio.
- Não invente dados.
- Leia todas as páginas/imagens como partes do MESMO documento.
- Nome: copie exatamente.
- RG/CIN: procure CPF, REGISTRO GERAL/RG, FILIAÇÃO, NASCIMENTO, SEXO,
  ÓRGÃO EXPEDIDOR e DATA DE EXPEDIÇÃO/EMISSÃO.
- Para RG/CIN: filiacao_1 = PAI e filiacao_2 = MÃE.
  Se houver duas linhas sem rótulo, identifique pai/mãe apenas quando estiver claro.
  Se ficar ambíguo, deixe os campos vazios e inclua-os em campos_incertos.
- cpf: somente dígitos.
- rg: letras/números sem pontuação.
- RG antigo: mantenha o número do Registro Geral em rg.
- RG novo/CIN: CPF é o identificador principal.
- data_emissao: use DATA DE EXPEDIÇÃO ou DATA DE EMISSÃO; nunca nascimento.
- validade: somente se houver VALIDADE explicitamente impressa.
- naturalidade_municipio: somente o nome COMPLETO da cidade, sem abreviações e sem UF; naturalidade_uf: SIGLA de 2 letras da UF de NASCIMENTO impressa junto da cidade (não use a UF de emissão do documento). Exemplos: 'S.ROQUE - SP' => cidade 'SAO ROQUE', UF 'SP'; 'S PAULO' ou 'S. PAULO' => 'SAO PAULO'. Nunca devolva 'S.ROQUE', 'S PAULO' ou 'CIDADE - SP' como nome final da cidade.
- sexo: MASCULINO ou FEMININO quando explicitamente identificável.
- órgão/UF: SSP-SP => orgao_emissor SSP e uf_emissao SP.
- Se aparecer IIRGD/IRGD em SP, mantenha a sigla em orgao_emissor; não converta aqui.
- CNH: leia SEMPRE pelos RÓTULOS OFICIAIS, nunca apenas pela posição:
  * data_nascimento = somente a DATA do campo 3 "DATA, LOCAL E UF DE NASCIMENTO";
  * data_emissao = somente o campo 4a "DATA EMISSÃO";
  * validade = somente o campo 4b "VALIDADE";
  * numero_cnh = somente o campo 5 "Nº REGISTRO", normalmente com 11 dígitos.
    IGNORE número vertical, número do espelho, código de segurança, RENACH e outros números;
  * data_primeira_habilitacao = somente o valor ao lado do texto explícito
    "1ª HABILITAÇÃO" / "PRIMEIRA HABILITAÇÃO".
    NUNCA use a validade nem datas da tabela de categorias como 1ª habilitação;
  * NÃO TROQUE 1ª HABILITAÇÃO, DATA EMISSÃO e VALIDADE entre si;
  * antes de responder, confira novamente Nº REGISTRO, 1ª HABILITAÇÃO,
    DATA EMISSÃO e VALIDADE pelos respectivos rótulos;
  * se um desses campos não estiver claramente legível, deixe vazio em vez de adivinhar.
- Se não estiver legível, deixe vazio e inclua a chave em campos_incertos.
""".strip()


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


@app.before_request
def _guard():
    if request.method == "OPTIONS":
        return None
    if not _origin_ok():
        return jsonify({"ok": False, "error": "Origem não autorizada."}), 403
    if request.path != "/health" and not _rate_ok():
        return jsonify({"ok": False, "error": "Muitas requisições. Aguarde um minuto."}), 429
    return None


def _to_jpeg(data: bytes, filename: str, mimetype: str) -> bytes:
    """
    Prioriza precisão de leitura para CNH/RG/CIN.
    O Render continua comprimindo o resultado, mas sem reduzir demais
    os números pequenos da CNH.
    """
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            for i in range(min(2, len(doc))):
                page = doc[i]
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2.2, 2.2),
                    alpha=False
                )
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append(img.copy())

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
        img = Image.open(io.BytesIO(data)).convert("RGB")

    max_side = 2800
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=91)
    return out.getvalue()


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


def _groq_call(image_bytes: bytes):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
    ]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]

    last_error = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_completion_tokens": 420,
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
                "User-Agent": "Assistente-SICAQ-Web/1.0",
            },
            method="POST",
        )
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            content = body["choices"][0]["message"]["content"]
            dados = json.loads(content)
            return dados, model, elapsed
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                espera = ""
                m = re.search(r"try again in\s+([0-9.]+)s", detail, re.I)
                if m:
                    espera = f" Aguarde cerca de {float(m.group(1)):.0f} segundos e tente novamente."
                last_error = GroqLimite(
                    "Limite gratuito do Groq atingido temporariamente." + espera
                )
                break

            last_error = RuntimeError(f"Groq HTTP {e.code}: {detail}")
            # Tenta o próximo modelo apenas em erros de modelo/404/400.
            if e.code not in (400, 404):
                break
        except Exception as e:
            last_error = e
            break

    if isinstance(last_error, GroqLimite):
        raise last_error
    raise RuntimeError(str(last_error or "Falha na leitura Groq/Qwen."))




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


def _tirar_acentos(valor):
    texto = str(valor or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


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
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
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


def _parse_data_br(valor):
    """Aceita DD/MM/AAAA, DD-MM-AAAA e DD.MM.AAAA. Devolve date ou None."""
    valor = str(valor or "").strip()
    m = re.fullmatch(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})", valor)
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except Exception:
        return None


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



UFS_BR = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

ESTADOS_POR_NOME = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "GOIAS": "GO", "MARANHAO": "MA",
    "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "MINAS GERAIS": "MG",
    "PARA": "PA", "PARAIBA": "PB", "PARANA": "PR", "PERNAMBUCO": "PE",
    "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO", "RORAIMA": "RR",
    "SANTA CATARINA": "SC", "SAO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}


def _uf_naturalidade(uf_valor, municipio_bruto):
    """
    Devolve a sigla da UF de nascimento (ex.: 'SP'), sempre uma das 27 UFs.
    1) usa naturalidade_uf lida pela IA (sigla ou nome do estado);
    2) se vier vazia/inválida, tenta a sigla no fim do texto da cidade
       ('S.ROQUE - SP', 'SAO PAULO/SP', 'SOROCABA (SP)').
    Nunca usa a UF de emissão do documento, que pode ser de outro estado.
    """
    uf = _tirar_acentos(str(uf_valor or "")).upper()
    uf = re.sub(r"[^A-Z ]", " ", uf)
    uf = re.sub(r"\s+", " ", uf).strip()
    if uf in UFS_BR:
        return uf
    if uf in ESTADOS_POR_NOME:
        return ESTADOS_POR_NOME[uf]

    cidade = _tirar_acentos(str(municipio_bruto or "")).upper().strip()
    m = re.search(r"[\s\-/(]+([A-Z]{2})\)?\s*$", cidade)
    if m and m.group(1) in UFS_BR and len(cidade) > 4:
        return m.group(1)
    return ""


def _normalizar_municipio_naturalidade(valor):
    """Expande abreviações comuns de naturalidade e devolve somente o município."""
    texto = _tirar_acentos(str(valor or "").strip().upper())

    # Remove sufixo de UF quando o OCR mistura a UF no campo da cidade.
    texto = re.sub(r"\s*[-/]\s*[A-Z]{2}\s*$", "", texto)
    texto = re.sub(r"\s*\(\s*[A-Z]{2}\s*\)\s*$", "", texto)
    m = re.search(r"^(.{3,}?)\s+([A-Z]{2})$", texto)
    if m and m.group(2) in UFS_BR:
        texto = m.group(1)
    texto = re.sub(r"^(S|STA|STO)\s*\.\s*", r"\1 ", texto)
    texto = re.sub(r"\s+", " ", texto).strip(" .,-")
    texto = re.sub(r"\s+", " ", texto)

    equivalencias = {
        "S PAULO": "SAO PAULO",
        "S ROQUE": "SAO ROQUE",
        "S JOSE DOS CAMPOS": "SAO JOSE DOS CAMPOS",
        "S JOSE DO RIO PRETO": "SAO JOSE DO RIO PRETO",
        "S JOSE DO RIO PARDO": "SAO JOSE DO RIO PARDO",
        "S BERNARDO DO CAMPO": "SAO BERNARDO DO CAMPO",
        "S CAETANO DO SUL": "SAO CAETANO DO SUL",
        "S CARLOS": "SAO CARLOS",
        "S VICENTE": "SAO VICENTE",
        "S SEBASTIAO": "SAO SEBASTIAO",
        "S JOAQUIM DA BARRA": "SAO JOAQUIM DA BARRA",
        "S PEDRO": "SAO PEDRO",
        "S MANUEL": "SAO MANUEL",
        "S MIGUEL ARCANJO": "SAO MIGUEL ARCANJO",
        "S SIMAO": "SAO SIMAO",
        "S BENTO DO SAPUCAI": "SAO BENTO DO SAPUCAI",
        "S JOAO DA BOA VISTA": "SAO JOAO DA BOA VISTA",
        "S JOSE": "SAO JOSE",
    }

    if texto in equivalencias:
        return equivalencias[texto]

    # Regras gerais para abreviações no início do nome do município.
    texto = re.sub(r"^S\s+(?=[A-Z])", "SAO ", texto)
    texto = re.sub(r"^STA\.?\s+(?=[A-Z])", "SANTA ", texto)
    texto = re.sub(r"^STO\.?\s+(?=[A-Z])", "SANTO ", texto)
    return texto

def _normalizar_saida(d):
    d = _corrigir_datas_cnh(d)
    d = dict(d or {})
    tipo = str(d.get("tipo_documento") or "").strip().upper()
    nacionalidade = str(d.get("nacionalidade") or "").strip().upper()
    if nacionalidade.startswith("BRASILEIR"):
        nacionalidade = "BRASILEIRA"

    nat_mun = _normalizar_municipio_naturalidade(
        d.get("naturalidade_municipio") or ""
    )
    nat_uf = _uf_naturalidade(
        d.get("naturalidade_uf"),
        d.get("naturalidade_municipio"),
    )

    sexo = _tirar_acentos(str(d.get("sexo") or "").strip().upper())
    if sexo in ("M", "MASC", "MASCULINO"):
        sexo = "MASCULINO"
    elif sexo in ("F", "FEM", "FEMININO"):
        sexo = "FEMININO"

    # Exibição no site: CIDADE/UF (ex.: SOROCABA/SP).
    # O site separa pela barra e envia à extensão o município (sem UF)
    # e a UF em campos próprios, como o SICAQ exige.
    incertos = list(d.get("campos_incertos") or [])
    if nat_mun and nat_uf:
        naturalidade = f"{nat_mun}/{nat_uf}"
    else:
        naturalidade = nat_mun
        if nat_mun and "naturalidade_uf" not in incertos:
            incertos.append("naturalidade_uf")

    return {
        "documento": tipo,
        "nome": _tirar_acentos(str(d.get("nome") or "").strip().upper()),
        "cpf": "".join(c for c in str(d.get("cpf") or "") if c.isdigit()),
        "rg": "".join(c for c in str(d.get("rg") or "").upper() if c.isalnum()),
        "numero_cnh": "".join(c for c in str(d.get("numero_cnh") or "") if c.isdigit()),
        "orgao_emissor": _tirar_acentos(str(d.get("orgao_emissor") or "").strip().upper()),
        "uf": _tirar_acentos(str(d.get("uf_emissao") or "").strip().upper()),
        "nascimento": _formatar_data_br(d.get("data_nascimento")),
        "primeira": _formatar_data_br(d.get("data_primeira_habilitacao")),
        "emissao": _formatar_data_br(d.get("data_emissao")),
        "validade": _formatar_data_br(d.get("validade")),
        "sexo": sexo,
        "nacionalidade": nacionalidade or "BRASILEIRA",
        "naturalidade": naturalidade,
        "naturalidade_uf": nat_uf,
        "filiacao1": _tirar_acentos(str(d.get("filiacao_1") or "").strip().upper()),
        "filiacao2": _tirar_acentos(str(d.get("filiacao_2") or "").strip().upper()),
        "modelo_documento": str(d.get("modelo_documento") or "").strip().upper(),
        "campos_incertos": incertos,
    }



# ============================================================
# HOLERITE / RENDA FORMAL
# ============================================================

def _sem_acentos(txt):
    txt = str(txt or "")
    return "".join(
        c for c in unicodedata.normalize("NFD", txt)
        if unicodedata.category(c) != "Mn"
    )


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


def _holerite_prompt():
    atual, anterior, retrasado = _holerite_competencias_validas()
    return f"""
Leia este HOLERITE / CONTRACHEQUE brasileiro e devolva SOMENTE JSON válido.

Chaves obrigatórias:
cnpj, data_admissao, competencia, bruta, liquido_original,
adiantamento_salarial_total, descricao_adiantamento, irrf, campos_incertos.

Regras:
- cnpj: CNPJ do empregador, somente dígitos.
- data_admissao: DD/MM/AAAA. Não confunda com data de pagamento.
- competencia: mês/ano do holerite em MM/AAAA.
- bruta: TOTAL DE PROVENTOS / TOTAL BRUTO. Não use salário base, base INSS,
  base FGTS ou líquido.
- liquido_original: somente LÍQUIDO A RECEBER / LÍQUIDO DO HOLERITE.
- adiantamento_salarial_total: some SOMENTE antecipação do salário mensal.
  Exemplos válidos: ADIANTAMENTO DE SALÁRIO, ADIANTAMENTO SALARIAL,
  ADIANT. SALÁRIO, ADIANTAMENTO QUINZENAL, VALE SALÁRIO, ANTECIPAÇÃO SALARIAL.
- NÃO trate como adiantamento salarial: empréstimo/consignado, férias,
  adiantamento de férias, 13º, vale transporte, vale refeição, vale alimentação,
  INSS, IRRF, pensão ou contribuições.
- descricao_adiantamento: copie a descrição usada para formar o adiantamento;
  se não houver, deixe vazio.
- irrf: valor DESCONTADO de IRRF / I.R.R.F. / IMPOSTO DE RENDA RETIDO.
  Se não houver, devolva "0,00".
- Valores monetários no formato brasileiro, ex.: "2304,21".
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


def _holerite_valor(valor):
    return f"{float(valor or 0):.2f}".replace(".", ",")


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


def _pdf_texto_holerite(data):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            partes = []
            for i in range(min(len(doc), 8)):
                texto = doc[i].get_text("text") or ""
                if texto.strip():
                    partes.append(f"\n--- PÁGINA {i+1} ---\n{texto}")
            return "".join(partes).strip()
        finally:
            doc.close()
    except Exception:
        return ""


def _imagem_holerite(data, filename, mimetype):
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if not is_pdf:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        max_w = 1500
        if img.width > max_w:
            escala = max_w / img.width
            img = img.resize((max_w, max(1, int(img.height * escala))))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
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
        return out.getvalue()
    finally:
        doc.close()


def _groq_holerite_call(arquivos):
    """
    Lê 1 ou 2 arquivos de holerite na mesma chamada.
    Se houver dois meses diferentes, o prompt escolhe a competência
    válida mais recente. Nunca soma valores de meses distintos.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    conteudo = [
        {"type": "text", "text": _holerite_prompt()}
    ]

    for idx, item in enumerate(arquivos[:2], start=1):
        data = item["data"]
        filename = item["filename"]
        mimetype = item["mimetype"]

        texto_pdf = ""
        if PathLikeSuffix(filename) == ".pdf" or mimetype == "application/pdf":
            texto_pdf = _pdf_texto_holerite(data)

        texto_pdf = _compactar_texto_holerite(texto_pdf, 20000)
        if len(texto_pdf) >= 120:
            # Limita cada arquivo para não desperdiçar tokens no plano gratuito.
            conteudo.append({
                "type": "text",
                "text": (
                    f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====\n"
                    + _compactar_texto_holerite(texto_pdf)
                ),
            })
        else:
            imagem = _imagem_holerite(data, filename, mimetype)
            b64 = base64.b64encode(imagem).decode("ascii")
            conteudo.append({
                "type": "text",
                "text": f"\n\n===== HOLERITE / ARQUIVO {idx}: {filename} =====",
            })
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
            "max_completion_tokens": 650,
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
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read().decode("utf-8"))
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


def _normalizar_holerite(resultado):
    resultado = dict(resultado or {})
    cnpj = re.sub(r"\D", "", str(resultado.get("cnpj") or ""))
    competencia = _holerite_competencia_mm_aaaa(resultado.get("competencia"))
    validas = _holerite_competencias_validas()

    if competencia not in validas:
        identificada = competencia or "não identificada"
        raise ValueError(
            f"HOLERITE_FORA_PERIODO|Competência identificada: {identificada}. "
            f"Período aceito: {validas[0]}, {validas[1]} e {validas[2]}."
        )

    bruta = _holerite_numero(resultado.get("bruta"))
    liquido_original = _holerite_numero(resultado.get("liquido_original"))
    adiantamento = _holerite_numero(resultado.get("adiantamento_salarial_total"))
    irrf = _holerite_numero(resultado.get("irrf"))

    # Regra SICAQ: somente adiantamento salarial volta ao líquido.
    liquida_sicaq = liquido_original + adiantamento

    return {
        "caracteristica_renda": "COMPROVADA",
        "tipo_fonte": "JURIDICA",
        "documento": "CONTRACHEQUE/HOLLERITH",
        "cnpj": cnpj,
        "data_admissao": str(resultado.get("data_admissao") or "").strip(),
        "competencia": competencia,
        "bruta": _holerite_valor(bruta),
        "liquida": _holerite_valor(liquida_sicaq),
        "irrf": _holerite_valor(irrf),
        "liquido_original": _holerite_valor(liquido_original),
        "adiantamento": _holerite_valor(adiantamento),
        "descricao_adiantamento": str(resultado.get("descricao_adiantamento") or "").strip().upper(),
        "campos_incertos": list(resultado.get("campos_incertos") or []),
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

    try:
        resultado, modelo, tempo_api = _groq_holerite_call(arquivos)
        dados = _normalizar_holerite(resultado)

        return jsonify({
            "ok": True,
            "dados": dados,
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


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Assistente SICAQ Web",
        "groq_configurada": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "holerite_competencias": _holerite_competencias_validas(),
    })


@app.post("/ler")
def ler():
    # Novo formato: "files" pode conter 1 ou 2 arquivos.
    recebidos = request.files.getlist("files")

    # Compatibilidade com a versão anterior do site.
    if not recebidos and "file" in request.files:
        recebidos = [request.files["file"]]

    recebidos = [f for f in recebidos if f and (f.filename or "").strip()]

    if not recebidos:
        return jsonify({"ok": False, "error": "Arquivo não enviado."}), 400

    if len(recebidos) > 2:
        return jsonify({
            "ok": False,
            "error": "Selecione no máximo 2 arquivos do mesmo documento."
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

    start = time.perf_counter()
    try:
        t_preparo = time.perf_counter()
        image_bytes = _combinar_arquivos_identificacao(arquivos)
        tempo_preparo = time.perf_counter() - t_preparo

        dados_raw, modelo, tempo_api = _groq_call(image_bytes)

        # CNH: uma única chamada ao Groq para não estourar o limite gratuito.
        # A conferência final é feita localmente, sem consumir tokens.
        if str(dados_raw.get("tipo_documento") or "").strip().upper() == "CNH":
            dados_raw = _validar_cnh_final(dados_raw)

        dados = _normalizar_saida(dados_raw)

        return jsonify({
            "ok": True,
            "dados": dados,
            "modelo": modelo,
            "arquivos_lidos": len(arquivos),
            "tempo_preparo": round(tempo_preparo, 3),
            "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - start, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
