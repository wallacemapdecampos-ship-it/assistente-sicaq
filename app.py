import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timedelta
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
- naturalidade_municipio: somente cidade; naturalidade_uf: UF.
- sexo: MASCULINO ou FEMININO quando explicitamente identificável.
- órgão/UF: SSP-SP => orgao_emissor SSP e uf_emissao SP.
- Se aparecer IIRGD/IRGD em SP, mantenha a sigla em orgao_emissor; não converta aqui.
- CNH:
  * data_nascimento = campo de nascimento;
  * data_emissao = SOMENTE o campo 4a DATA EMISSÃO;
  * validade = SOMENTE o campo 4b VALIDADE;
  * ATENÇÃO: 4a DATA EMISSÃO e 4b VALIDADE ficam lado a lado. NÃO TROQUE OS DOIS.
  * A DATA DE EMISSÃO deve ser anterior ou igual à VALIDADE.
  * numero_cnh = campo 5 Nº REGISTRO;
  * data_primeira_habilitacao = somente campo 1ª HABILITAÇÃO.
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
    """Modo rápido para CNH/RG/CIN."""
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            for i in range(min(2, len(doc))):
                page = doc[i]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(1.65, 1.65), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg"))).convert("RGB")

                max_w = 1250
                if img.width > max_w:
                    scale = max_w / img.width
                    img = img.resize(
                        (max_w, max(1, int(img.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                pages.append(img.copy())

            if not pages:
                raise ValueError("PDF sem páginas válidas.")

            if len(pages) == 1:
                img = pages[0]
            else:
                gap = 18
                total_w = pages[0].width + pages[1].width + gap
                max_h = max(p.height for p in pages)
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

    max_side = 1900
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=84)
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
        max_w = 1450
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
        imagens[0].save(out, format="JPEG", quality=84)
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
    canvas.save(out, format="JPEG", quality=84)
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
            last_error = RuntimeError(f"Groq HTTP {e.code}: {detail}")
            # Tenta o próximo modelo apenas em erros de modelo/404/400.
            if e.code not in (400, 404):
                break
        except Exception as e:
            last_error = e
            break

    raise RuntimeError(str(last_error or "Falha na leitura Groq/Qwen."))



def _parse_data_br(valor):
    valor = str(valor or "").strip()
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", valor)
    if not m:
        return None
    try:
        from datetime import date
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except Exception:
        return None


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


def _normalizar_saida(d):
    d = _corrigir_datas_cnh(d)
    d = dict(d or {})
    tipo = str(d.get("tipo_documento") or "").strip().upper()
    nacionalidade = str(d.get("nacionalidade") or "").strip().upper()
    if nacionalidade.startswith("BRASILEIR"):
        nacionalidade = "BRASILEIRA"

    nat_mun = str(d.get("naturalidade_municipio") or "").strip().upper()
    nat_uf = str(d.get("naturalidade_uf") or "").strip().upper()
    naturalidade = f"{nat_mun}/{nat_uf}" if nat_mun and nat_uf else nat_mun

    return {
        "documento": tipo,
        "nome": str(d.get("nome") or "").strip().upper(),
        "cpf": "".join(c for c in str(d.get("cpf") or "") if c.isdigit()),
        "rg": "".join(c for c in str(d.get("rg") or "").upper() if c.isalnum()),
        "numero_cnh": "".join(c for c in str(d.get("numero_cnh") or "") if c.isdigit()),
        "orgao_emissor": str(d.get("orgao_emissor") or "").strip().upper(),
        "uf": str(d.get("uf_emissao") or "").strip().upper(),
        "nascimento": str(d.get("data_nascimento") or "").strip(),
        "primeira": str(d.get("data_primeira_habilitacao") or "").strip(),
        "emissao": str(d.get("data_emissao") or "").strip(),
        "validade": str(d.get("validade") or "").strip(),
        "sexo": str(d.get("sexo") or "").strip().upper(),
        "nacionalidade": nacionalidade or "BRASILEIRA",
        "naturalidade": naturalidade,
        "filiacao1": str(d.get("filiacao_1") or "").strip().upper(),
        "filiacao2": str(d.get("filiacao_2") or "").strip().upper(),
        "modelo_documento": str(d.get("modelo_documento") or "").strip().upper(),
        "campos_incertos": d.get("campos_incertos") or [],
    }



# ============================================================
# HOLERITE / RENDA FORMAL
# ============================================================

def _sem_acentos(txt):
    import unicodedata
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


def _groq_holerite_call(data, filename, mimetype):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    texto_pdf = ""
    if PathLikeSuffix(filename) == ".pdf" or mimetype == "application/pdf":
        texto_pdf = _pdf_texto_holerite(data)

    if len(texto_pdf) >= 120:
        conteudo = _holerite_prompt() + "\n\nTEXTO EXTRAÍDO DO HOLERITE:\n" + texto_pdf[:18000]
    else:
        imagem = _imagem_holerite(data, filename, mimetype)
        b64 = base64.b64encode(imagem).decode("ascii")
        conteudo = [
            {"type": "text", "text": _holerite_prompt()},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]

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
                "User-Agent": "Assistente-SICAQ-Holerite-Web/1.0",
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
            ultimo = RuntimeError(f"Groq HTTP {e.code}: {detalhe}")
            if e.code not in (400, 404):
                break
        except Exception as e:
            ultimo = e
            break

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
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Holerite não enviado."}), 400

    f = request.files["file"]
    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400

    inicio = time.perf_counter()
    try:
        resultado, modelo, tempo_api = _groq_holerite_call(
            data,
            f.filename or "",
            f.mimetype or "",
        )
        dados = _normalizar_holerite(resultado)
        return jsonify({
            "ok": True,
            "dados": dados,
            "modelo": modelo,
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
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
