import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import pymupdf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

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
  * data_emissao = campo 4a DATA EMISSÃO;
  * validade = campo 4b VALIDADE;
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
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            for i in range(min(2, len(doc))):
                page = doc[i]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append(img.copy())
            if not pages:
                raise ValueError("PDF sem páginas válidas.")
            max_w = max(im.width for im in pages)
            resized = []
            total_h = 0
            for im in pages:
                if im.width != max_w:
                    scale = max_w / im.width
                    im = im.resize((max_w, int(im.height * scale)))
                resized.append(im)
                total_h += im.height
            canvas = Image.new("RGB", (max_w, total_h), "white")
            y = 0
            for im in resized:
                canvas.paste(im, (0, y))
                y += im.height
            img = canvas
        finally:
            doc.close()
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")

    max_side = 2200
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90, optimize=True)
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
        "qwen/qwen3.6-27b",
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
            "temperature": 0.1,
            "top_p": 0.8,
            "max_completion_tokens": 600,
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


def _normalizar_saida(d):
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


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Assistente SICAQ Web",
        "groq_configurada": bool(os.getenv("GROQ_API_KEY", "").strip()),
    })


@app.post("/ler")
def ler():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Arquivo não enviado."}), 400

    f = request.files["file"]
    data = f.read()
    if not data:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400

    start = time.perf_counter()
    try:
        image_bytes = _to_jpeg(data, f.filename or "", f.mimetype or "")
        dados_raw, modelo, tempo_api = _groq_call(image_bytes)
        dados = _normalizar_saida(dados_raw)
        return jsonify({
            "ok": True,
            "dados": dados,
            "modelo": modelo,
            "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - start, 3),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
