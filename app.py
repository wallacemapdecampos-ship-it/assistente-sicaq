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


def _groq_abrir(req, timeout=35):
    """
    Envia o pedido à Groq. No limite POR MINUTO do plano gratuito, se a espera
    for curta (até 45 s), aguarda e tenta de novo sozinho; assim o analista
    não vê erro. Limite diário ou espera longa: avisa quanto tempo falta.
    """
    for tentativa in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
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
            if tentativa == 0 and not diario and espera is not None and espera <= 45:
                time.sleep(espera + 0.7)
                continue
            if diario:
                raise GroqLimite("Limite diário gratuito da Groq atingido. " + _texto_espera(espera))
            raise GroqLimite("Limite por minuto da Groq atingido. " + _texto_espera(espera))
    raise GroqLimite("Limite por minuto da Groq atingido. " + _texto_espera(None))


RATE_WINDOW = 60
RATE_MAX = 20
_rate = defaultdict(deque)

PROMPT = """
Leia o documento brasileiro e responda SOMENTE JSON válido com estas chaves:
tipo_documento, modelo_documento, nome, cpf, rg, data_nascimento, numero_cnh,
orgao_emissor, uf_emissao, data_primeira_habilitacao, data_emissao, validade,
sexo, nacionalidade, naturalidade_municipio, naturalidade_uf, filiacao_1,
filiacao_2, mrz, campos_incertos.

Regras:
- tipo_documento: CNH, RG ou CIN. modelo_documento: ANTIGO, NOVO ou vazio.
- CIN = Carteira de Identidade Nacional (modelo novo): também tem o título
  "CARTEIRA DE IDENTIDADE", mas o número do documento é o CPF, tem VALIDADE,
  rótulos em português e inglês (NOME/NAME, NACIONALIDADE/NATIONALITY BRA),
  QR Code e zona '<<<' no verso. Nesse caso responda CIN, nunca RG.
  RG = modelo antigo, com número de REGISTRO GERAL próprio (diferente do CPF)
  e normalmente sem validade.
- A imagem pode ser FOTO de celular: o documento pode estar pequeno, inclinado,
  girado ou com reflexo. Leia só o documento e ignore o fundo.
- Não invente dados.
- mrz: se existir a zona de leitura mecânica (linhas com muitos '<', no verso
  da CNH nova ou da CIN), copie essas linhas exatamente, separadas por \\n.
  Se não existir, devolva vazio.
- Leia todas as páginas/imagens como partes do MESMO documento.
- Nome, filiacao_1 e filiacao_2: copie LETRA POR LETRA exatamente como está
  impresso. Não corrija a grafia, não acrescente, não duplique e não remova
  letras (ex.: se está impresso MARIO, devolva MARIO, nunca MARRIO).
- RG/CIN: procure CPF, REGISTRO GERAL/RG, FILIAÇÃO, NASCIMENTO, SEXO,
  ÓRGÃO EXPEDIDOR e DATA DE EXPEDIÇÃO/EMISSÃO.
- Para RG/CIN: filiacao_1 = PAI e filiacao_2 = MÃE.
  Muitas vezes a FILIAÇÃO vem corrida, "PAI E MÃE", e a quebra de linha cai
  no meio do nome da mãe (ex.: "JOSE DA SILVA E MARIA" / "APARECIDA SOUZA").
  Separe pelo " E ", não pela linha: pai "JOSE DA SILVA", mãe "MARIA
  APARECIDA SOUZA". Não deixe o " E " nem parte da mãe no nome do pai.
  Se houver duas linhas sem rótulo, identifique pai/mãe apenas quando estiver claro.
  Se ficar ambíguo, deixe os campos vazios e inclua-os em campos_incertos.
- cpf: somente dígitos.
- rg: letras/números sem pontuação, SOMENTE do campo "REGISTRO GERAL"
  (no RG antigo fica no lado dos dados, ao lado do nome, geralmente em
  VERMELHO e com dígito no fim, ex.: 58.569.906-9 => 585699069).
  NUNCA use em rg o número pequeno impresso no lado da digital/assinatura
  (número do espelho/formulário), o número ao lado da foto (ex.: 8274-3),
  números verticais, nem números do campo DOC. ORIGEM / certidão.
- RG novo/CIN: CPF é o identificador principal.
- data_emissao: use DATA DE EXPEDIÇÃO ou DATA DE EMISSÃO; nunca nascimento.
- Todas as datas em DD/MM/AAAA, com o mês em número (ex.: 04/MAR/1994 => 04/03/1994).
- validade: somente se houver VALIDADE explicitamente impressa.
- naturalidade_municipio: somente o nome COMPLETO da cidade, sem abreviações e sem UF; naturalidade_uf: SIGLA de 2 letras da UF de NASCIMENTO impressa junto da cidade (não use a UF de emissão do documento). Exemplos: 'S.ROQUE - SP' => cidade 'SAO ROQUE', UF 'SP'; 'S PAULO' ou 'S. PAULO' => 'SAO PAULO'. Nunca devolva 'S.ROQUE', 'S PAULO' ou 'CIDADE - SP' como nome final da cidade.
- sexo: MASCULINO ou FEMININO quando explicitamente identificável.
- órgão/UF: SSP-SP => orgao_emissor SSP e uf_emissao SP.
- Se aparecer IIRGD/IRGD em SP, mantenha a sigla em orgao_emissor; não converta aqui.
- orgao_emissor: copie a sigla/nome do órgão exatamente como impresso
  (ex.: SSP, IIRGD, SESP, SDS, SEDS, PC, PCMG, IFP, DETRAN, IGP, CASA CIVIL).
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
  * se um desses campos não estiver claramente legível, deixe vazio em vez de adivinhar;
  * CNH modelo ANTIGO (campo "DATA NASCIMENTO" sozinho, sem "LOCAL E UF DE
    NASCIMENTO"): esse modelo NÃO informa a naturalidade. Devolva
    naturalidade_municipio e naturalidade_uf VAZIOS. O campo "LOCAL" do verso é
    o LOCAL DE EMISSÃO da CNH, nunca a naturalidade. A data de emissão desse
    modelo fica no verso, em "DATA EMISSÃO".
- naturalidade_municipio é SEMPRE nome de CIDADE. Nunca coloque nela nome de
  pessoa (nome do titular, do pai ou da mãe). Se não houver cidade de
  nascimento impressa, deixe vazio.
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


# ============================================================
# LOGIN (um usuário por analista) — sem banco de dados
# Usuários na variável USUARIOS do Render, em JSON:
#   {"ana": {"nome": "Ana", "senha": "pbkdf2$...", "admin": false}, ...}
# Pessoas novas podem ir numa lista extra USUARIOS_2, USUARIOS_3... (mesmo formato).
# A sessão é um "crachá" assinado (HMAC com SEGREDO_SESSAO) que vale até
# 23h59 do dia (horário de Brasília). Tirar o usuário da lista bloqueia na hora.
# Sem USUARIOS configurado, o login fica desligado (funciona como antes).
# ============================================================

FUSO_BR = ZoneInfo("America/Sao_Paulo")
_SEGREDO_PADRAO = secrets.token_hex(32)
_falhas_login = defaultdict(deque)


def _segredo():
    return (os.getenv("SEGREDO_SESSAO", "").strip() or _SEGREDO_PADRAO).encode()


def _usuarios():
    """Usuários que podem entrar: os da lista do Render menos os excluídos pelo dono."""
    todos = _usuarios_todos()
    fora = _excluidos()
    return {u: c for u, c in todos.items() if u not in fora or u in DONOS_SISTEMA}


def _usuarios_todos():
    """
    Junta a lista USUARIOS com as listas extras USUARIOS_<algo> (ex.:
    USUARIOS_2). Assim dá para incluir alguém novo sem reescrever a lista
    principal. Uma lista com JSON quebrado é ignorada sem derrubar as outras.
    """
    nomes = ["USUARIOS"] + sorted(k for k in os.environ if k.startswith("USUARIOS_"))
    todos = {}
    for nome in nomes:
        try:
            bruto = json.loads(os.getenv(nome, "").strip() or "{}")
            todos.update({str(k).strip().lower(): v for k, v in bruto.items() if isinstance(v, dict)})
        except Exception:
            print(f"[LOGIN] variável {nome} inválida (JSON)", flush=True)
    # Cadastrados pelo painel (banco). Quem está na lista do Render tem prioridade.
    for u, conta in _usuarios_do_banco().items():
        todos.setdefault(u, conta)
    # ADM ligado/desligado pelo dono no painel (o dono é sempre ADM).
    for u, adm in _papeis().items():
        if u in todos and u not in DONOS_SISTEMA:
            todos[u] = {**todos[u], "admin": adm}
    # Donos do sistema são sempre ADM.
    for u in DONOS_SISTEMA:
        if u in todos:
            todos[u] = {**todos[u], "admin": True}
    return todos


_PAPEIS = {"dados": None, "quando": 0.0}


def _papeis(atualizar=False):
    """{usuario: é ADM?} definido pelo dono no painel."""
    if not os.getenv("DATABASE_URL", "").strip():
        return {}
    agora = time.time()
    if atualizar or _PAPEIS["dados"] is None or agora - _PAPEIS["quando"] > 60:
        try:
            linhas = _no_banco(lambda con: con.run("SELECT usuario, admin FROM papeis_usuarios")) or []
            _PAPEIS["dados"] = {str(u): bool(a) for u, a in linhas}
        except Exception as e:
            print(f"[PAPEL] banco indisponível ({type(e).__name__})", flush=True)
        _PAPEIS["quando"] = agora
    return _PAPEIS["dados"] or {}


# ---------- Usuários cadastrados pelo painel ----------
# O dono cadastra nome, login, senha inicial (guardada só como hash), empresa
# e se é ADM da empresa. No 1º acesso a pessoa cria a própria senha.
_CADASTRADOS = {"dados": None, "quando": 0.0}
LOGIN_VALIDO = re.compile(r"^[a-z0-9][a-z0-9._-]{2,29}$")


def _usuarios_do_banco(atualizar=False):
    if not os.getenv("DATABASE_URL", "").strip():
        return {}
    agora = time.time()
    if atualizar or _CADASTRADOS["dados"] is None or agora - _CADASTRADOS["quando"] > 60:
        try:
            linhas = _no_banco(lambda con: con.run(
                "SELECT usuario, nome, senha_inicial, admin FROM usuarios_cadastrados")) or []
            _CADASTRADOS["dados"] = {
                str(u): {"nome": str(n or u), "senha": str(s), "admin": bool(a), "painel": True}
                for u, n, s, a in linhas}
        except Exception as e:
            print(f"[CADASTRO] banco indisponível ({type(e).__name__})", flush=True)
        _CADASTRADOS["quando"] = agora
    return _CADASTRADOS["dados"] or {}


def _login_ligado():
    return bool(_usuarios_todos())


# ---------- Logins excluídos pelo dono ----------
# Os logins nascem na lista USUARIOS do Render. "Excluir" pelo painel guarda o
# usuário na tabela usuarios_excluidos: ele não entra mais e some do painel
# (dá para Restaurar). Sem banco, ninguém fica excluído.
_EXCLUIDOS = {"dados": None, "quando": 0.0}


def _excluidos(atualizar=False):
    if not os.getenv("DATABASE_URL", "").strip():
        return set()
    agora = time.time()
    if atualizar or _EXCLUIDOS["dados"] is None or agora - _EXCLUIDOS["quando"] > 60:
        try:
            linhas = _no_banco(lambda con: con.run("SELECT usuario FROM usuarios_excluidos")) or []
            _EXCLUIDOS["dados"] = {str(r[0]) for r in linhas}
            _EXCLUIDOS["quando"] = agora
        except Exception as e:
            # Banco fora do ar: usa a última lista conhecida.
            print(f"[EXCLUIR] banco indisponível ({type(e).__name__})", flush=True)
            _EXCLUIDOS["quando"] = agora
    return _EXCLUIDOS["dados"] or set()


def _hash_senha(senha, sal=None, voltas=200_000):
    sal = sal or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", senha.encode("utf-8"), bytes.fromhex(sal), voltas)
    return f"pbkdf2${voltas}${sal}${h.hex()}"


def _senha_confere(senha, guardado):
    try:
        tipo, voltas, sal, esperado = str(guardado).split("$")
        if tipo != "pbkdf2":
            return False
        h = hashlib.pbkdf2_hmac("sha256", str(senha).encode("utf-8"), bytes.fromhex(sal), int(voltas))
        return hmac.compare_digest(h.hex(), esperado)
    except Exception:
        return False


def _b64(dados):
    return base64.urlsafe_b64encode(dados).decode().rstrip("=")


def _de_b64(texto):
    return base64.urlsafe_b64decode(texto + "=" * (-len(texto) % 4))


def _fim_do_dia():
    agora = datetime.now(FUSO_BR)
    return int(agora.replace(hour=23, minute=59, second=59, microsecond=0).timestamp())


def _token_novo(usuario, trocar=False, sid=""):
    dados = {"u": usuario, "exp": _fim_do_dia()}
    if sid:
        dados["s"] = sid  # sessão (computador) que fez o login
    if trocar:
        dados["t"] = 1  # ainda com a senha inicial: só pode criar a senha própria
    corpo = _b64(json.dumps(dados).encode())
    assinatura = _b64(hmac.new(_segredo(), corpo.encode(), hashlib.sha256).digest())
    return f"{corpo}.{assinatura}"


def _dados_do_token(token):
    """(usuario, precisa_trocar_senha, sid) ou (None, False, "")."""
    try:
        corpo, assinatura = str(token or "").split(".", 1)
        certa = _b64(hmac.new(_segredo(), corpo.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(certa, assinatura):
            return None, False, ""
        dados = json.loads(_de_b64(corpo))
        usuario = str(dados.get("u") or "").lower()
        if int(dados.get("exp") or 0) < time.time():
            return None, False, ""
        if usuario not in _usuarios():  # removido = bloqueado
            return None, False, ""
        return usuario, bool(dados.get("t")), str(dados.get("s") or "")
    except Exception:
        return None, False, ""


def _usuario_do_token(token):
    return _dados_do_token(token)[0]


# ---------- Um computador por login ----------
# Cada login novo vira a sessão ativa daquele usuário; o computador que
# estava usando antes é derrubado no próximo acesso (ou na checagem que o
# site faz a cada 2 minutos). Fica na memória e no banco (sobrevive ao
# servidor reiniciar). Banco fora do ar: não derruba ninguém.
_SESSOES = {}


def _sessao_atual(usuario):
    """sid da sessão ativa ("" = nenhuma registrada; None = não deu para saber)."""
    if usuario in _SESSOES:
        return _SESSOES[usuario]
    if not os.getenv("DATABASE_URL", "").strip():
        return ""
    try:
        linhas = _no_banco(lambda con: con.run(
            "SELECT sid FROM sessoes_ativas WHERE usuario = :usuario", usuario=usuario)) or []
        _SESSOES[usuario] = str(linhas[0][0]) if linhas else ""
        return _SESSOES[usuario]
    except Exception as e:
        print(f"[SESSAO] banco indisponível ({type(e).__name__})", flush=True)
        return None


def _registrar_sessao(usuario):
    sid = secrets.token_hex(8)
    _SESSOES[usuario] = sid
    if os.getenv("DATABASE_URL", "").strip():
        try:
            _no_banco(lambda con: con.run(
                "INSERT INTO sessoes_ativas (usuario, sid) VALUES (:usuario, :sid)"
                " ON CONFLICT (usuario) DO UPDATE SET sid = EXCLUDED.sid, criada_em = CURRENT_TIMESTAMP",
                usuario=usuario, sid=sid))
        except Exception as e:
            print(f"[SESSAO] não gravou no banco ({type(e).__name__}); só na memória", flush=True)
    return sid


@app.before_request
def _guard():
    if request.method == "OPTIONS":
        return None
    if not _origin_ok():
        return jsonify({"ok": False, "error": "Origem não autorizada."}), 403
    if request.path not in ("/health", "/sessao") and not _rate_ok():
        return jsonify({"ok": False, "error": "Muitas requisições. Aguarde um minuto."}), 429
    g.usuario = ""
    if request.path in ("/health", "/login") or not _login_ligado():
        return None
    cab = request.headers.get("Authorization", "")
    usuario, trocar, sid = _dados_do_token(cab[7:] if cab.startswith("Bearer ") else "")
    if not usuario:
        return jsonify({"ok": False, "code": "LOGIN", "error": "Sua sessão terminou. Entre de novo."}), 401
    atual = _sessao_atual(usuario)
    if atual and sid != atual:
        return jsonify({"ok": False, "code": "OUTRO_PC",
                        "error": "Este usuário entrou em outro computador. Entre de novo."}), 401
    g.usuario = usuario
    g.trocar = trocar
    if _sem_empresa(usuario):
        return jsonify({"ok": False, "code": "SEM_EMPRESA", "error": MSG_SEM_EMPRESA}), 401
    if trocar and request.path != "/trocar-senha":
        return jsonify({"ok": False, "code": "TROCAR_SENHA",
                        "error": "Crie a sua senha para continuar."}), 403
    return None


@app.post("/login")
def login():
    if not _login_ligado():
        return jsonify({"ok": False, "error": "Login não configurado no servidor."}), 400
    dados = request.get_json(silent=True) or {}
    usuario = str(dados.get("usuario") or "").strip().lower()
    senha = str(dados.get("senha") or "")

    # Muitas senhas erradas para o mesmo usuário (no mesmo lugar): pausa de
    # 10 minutos. Por usuário, para um erro não travar o escritório todo.
    agora = time.time()
    falhas = _falhas_login[(_ip(), usuario)]
    while falhas and falhas[0] < agora - 600:
        falhas.popleft()
    if len(falhas) >= 8:
        return jsonify({"ok": False, "error": "Muitas tentativas. Aguarde 10 minutos."}), 429

    conta = _usuarios().get(usuario)
    # Analista bloqueado (3 senhas erradas): não entra nem com a senha certa.
    vigiado = bool(conta) and not conta.get("admin")
    erros_antes = 0
    if vigiado:
        erros_antes, bloqueado = _bloqueio(usuario)
        if bloqueado:
            print(f"[LOGIN] tentativa em usuário bloqueado: {usuario}", flush=True)
            return jsonify({"ok": False, "code": "BLOQUEADO", "error": MSG_BLOQUEADO}), 423

    certo, trocar = _conferir_senha_login(usuario, conta, senha)
    if not certo:
        falhas.append(agora)
        print("[LOGIN] senha ou usuário incorretos", flush=True)
        if vigiado:
            erros, bloqueado = _registrar_erro_login(usuario)
            if bloqueado:
                print(f"[LOGIN] {usuario} bloqueado após {LIMITE_ERROS_LOGIN} senhas erradas", flush=True)
                return jsonify({"ok": False, "code": "BLOQUEADO", "error": MSG_BLOQUEADO}), 423
            restam = LIMITE_ERROS_LOGIN - erros
            if restam == 1:
                return jsonify({"ok": False, "error": "Usuário ou senha incorretos. Atenção: mais 1 erro "
                                "bloqueia o acesso e só o administrador libera."}), 401
        return jsonify({"ok": False, "error": "Usuário ou senha incorretos."}), 401

    if _sem_empresa(usuario):
        print(f"[LOGIN] {usuario} sem empresa: não entra", flush=True)
        return jsonify({"ok": False, "code": "SEM_EMPRESA", "error": MSG_SEM_EMPRESA}), 403

    if vigiado and erros_antes:
        _limpar_bloqueio(usuario)  # acertou: zera os erros
    print(f"[LOGIN] entrou: {usuario}" + (" (senha inicial)" if trocar else ""), flush=True)
    return jsonify(_resposta_login(usuario, conta, trocar))


def _resposta_login(usuario, conta, trocar):
    sid = _registrar_sessao(usuario)
    return {
        "ok": True,
        "token": _token_novo(usuario, trocar, sid),
        "usuario": usuario,
        "nome": str(conta.get("nome") or usuario),
        "admin": bool(conta.get("admin")),
        "empresa": _empresa_de(usuario),
        "dono": _eh_dono(usuario),
        "leme": _pergunta_leme(usuario),
        "expira": _fim_do_dia(),
        "trocar_senha": bool(trocar),
    }


# ---------- Senha própria de cada analista ----------
# A lista USUARIOS (no Render) tem a senha INICIAL. No primeiro acesso cada
# analista cria a sua senha, que fica só como "hash" no banco (Neon), na
# tabela senhas_usuarios. O administrador pode "resetar": a pessoa volta a
# entrar com a senha inicial e cria outra. Sem banco, fica como antes.
SENHA_MINIMO = 6
_SENHAS = {"dados": None, "quando": 0.0}


def _senhas_proprias(atualizar=False):
    """{usuario: hash} das senhas criadas. None quando não há banco."""
    if not os.getenv("DATABASE_URL", "").strip():
        return None
    agora = time.time()
    if atualizar or _SENHAS["dados"] is None or agora - _SENHAS["quando"] > 60:
        try:
            linhas = _no_banco(lambda con: con.run("SELECT usuario, senha FROM senhas_usuarios")) or []
            _SENHAS["dados"] = {str(u): str(s) for u, s in linhas}
            _SENHAS["quando"] = agora
        except Exception as e:
            # Banco fora do ar: usa a última lista conhecida.
            print(f"[SENHA] banco indisponível ({type(e).__name__})", flush=True)
    return _SENHAS["dados"]


def _conferir_senha_login(usuario, conta, senha):
    """(senha certa?, precisa criar a senha própria?)"""
    if not conta:
        return False, False
    proprias = _senhas_proprias(atualizar=True)
    propria = (proprias or {}).get(usuario)
    if propria:
        return _senha_confere(senha, propria), False
    certa = _senha_confere(senha, conta.get("senha"))
    # Só obriga a trocar quando dá para guardar a senha nova.
    return certa, bool(certa and proprias is not None)


def _problema_senha_nova(usuario, conta, nova):
    if len(nova) < SENHA_MINIMO:
        return f"A senha precisa ter pelo menos {SENHA_MINIMO} caracteres."
    if len(nova) > 72:
        return "Senha muito longa."
    if nova.strip().lower() == usuario:
        return "A senha não pode ser igual ao usuário."
    if _senha_confere(nova, conta.get("senha")):
        return "Escolha uma senha diferente da senha inicial."
    return ""


@app.post("/trocar-senha")
def trocar_senha():
    if not _login_ligado():
        return jsonify({"ok": False, "error": "Login não configurado no servidor."}), 400
    if _senhas_proprias(atualizar=True) is None:
        return jsonify({"ok": False, "error": "Não consegui falar com o banco de senhas. Tente de novo em 1 minuto."}), 503
    usuario = g.usuario
    conta = _usuarios().get(usuario) or {}
    # Analista só CRIA a senha no primeiro acesso. Trocar depois: só o ADM.
    if not conta.get("admin") and not getattr(g, "trocar", False):
        return jsonify({"ok": False, "error": "Para trocar a senha, fale com o administrador."}), 403
    dados = request.get_json(silent=True) or {}
    atual = str(dados.get("senha_atual") or "")
    nova = str(dados.get("senha_nova") or "")

    agora = time.time()
    falhas = _falhas_login[(_ip(), usuario)]
    while falhas and falhas[0] < agora - 600:
        falhas.popleft()
    if len(falhas) >= 8:
        return jsonify({"ok": False, "error": "Muitas tentativas. Aguarde 10 minutos."}), 429

    certo, _ = _conferir_senha_login(usuario, conta, atual)
    if not certo:
        falhas.append(agora)
        return jsonify({"ok": False, "error": "A senha atual não confere."}), 400
    problema = _problema_senha_nova(usuario, conta, nova)
    if problema:
        return jsonify({"ok": False, "error": problema}), 400
    if nova == atual:
        return jsonify({"ok": False, "error": "A senha nova é igual à atual."}), 400

    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO senhas_usuarios (usuario, senha) VALUES (:usuario, :senha)"
            " ON CONFLICT (usuario) DO UPDATE SET senha = EXCLUDED.senha, trocada_em = CURRENT_TIMESTAMP",
            usuario=usuario, senha=_hash_senha(nova),
        ))
    except Exception as e:
        print(f"[SENHA] não gravou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui salvar a senha agora. Tente de novo em 1 minuto."}), 503
    _senhas_proprias(atualizar=True)
    print(f"[SENHA] {usuario} criou/trocou a senha", flush=True)
    return jsonify(_resposta_login(usuario, conta, False))


@app.get("/sessao")
def sessao():
    """O site pergunta a cada 2 min se a sessão ainda vale (o guard responde)."""
    usuario = getattr(g, "usuario", "")
    if not usuario:
        return jsonify({"ok": True, "usuario": ""})
    # Perfil atual: se o dono mudou ADM, empresa ou Leme, o site se ajusta
    # sem a pessoa precisar sair e entrar.
    conta = _usuarios().get(usuario, {})
    return jsonify({"ok": True, "usuario": usuario,
                    "nome": str(conta.get("nome") or usuario),
                    "admin": bool(conta.get("admin")),
                    "dono": _eh_dono(usuario),
                    "empresa": _empresa_de(usuario),
                    "leme": _pergunta_leme(usuario)})


@app.post("/resetar-senha")
def resetar_senha():
    conta = _usuarios().get(getattr(g, "usuario", ""), {})
    if not conta.get("admin"):
        return jsonify({"ok": False, "error": "Só o administrador pode resetar senhas."}), 403
    alvo = str((request.get_json(silent=True) or {}).get("usuario") or "").strip().lower()
    if alvo not in _usuarios():
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    if not _pode_gerir(alvo):
        return jsonify({"ok": False, "error": "Você só pode mexer nos analistas da sua empresa."}), 403
    try:
        _no_banco(lambda con: con.run("DELETE FROM senhas_usuarios WHERE usuario = :usuario", usuario=alvo))
    except Exception as e:
        print(f"[SENHA] reset falhou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _senhas_proprias(atualizar=True)
    _limpar_bloqueio(alvo)  # resetar também desbloqueia
    print(f"[SENHA] {g.usuario} resetou a senha de {alvo}", flush=True)
    return jsonify({"ok": True})


@app.post("/definir-senha")
def definir_senha():
    """ADM escolhe a senha de um analista (já vale no próximo login e desbloqueia)."""
    conta = _usuarios().get(getattr(g, "usuario", ""), {})
    if not conta.get("admin"):
        return jsonify({"ok": False, "error": "Só o administrador pode definir senhas."}), 403
    dados = request.get_json(silent=True) or {}
    alvo = str(dados.get("usuario") or "").strip().lower()
    nova = str(dados.get("senha_nova") or "")
    conta_alvo = _usuarios().get(alvo)
    if not conta_alvo:
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    if not _pode_gerir(alvo):
        return jsonify({"ok": False, "error": "Você só pode mexer nos analistas da sua empresa."}), 403
    problema = _problema_senha_nova(alvo, conta_alvo, nova)
    if problema:
        return jsonify({"ok": False, "error": problema}), 400
    if not os.getenv("DATABASE_URL", "").strip():
        return jsonify({"ok": False, "error": "Sem banco de senhas configurado no servidor."}), 503
    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO senhas_usuarios (usuario, senha) VALUES (:usuario, :senha)"
            " ON CONFLICT (usuario) DO UPDATE SET senha = EXCLUDED.senha, trocada_em = CURRENT_TIMESTAMP",
            usuario=alvo, senha=_hash_senha(nova),
        ))
    except Exception as e:
        print(f"[SENHA] definir falhou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui salvar a senha agora. Tente de novo em 1 minuto."}), 503
    _senhas_proprias(atualizar=True)
    _limpar_bloqueio(alvo)
    print(f"[SENHA] {g.usuario} definiu a senha de {alvo}", flush=True)
    return jsonify({"ok": True})


@app.post("/desbloquear")
def desbloquear():
    conta = _usuarios().get(getattr(g, "usuario", ""), {})
    if not conta.get("admin"):
        return jsonify({"ok": False, "error": "Só o administrador pode desbloquear."}), 403
    alvo = str((request.get_json(silent=True) or {}).get("usuario") or "").strip().lower()
    if alvo not in _usuarios():
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    if not _pode_gerir(alvo):
        return jsonify({"ok": False, "error": "Você só pode mexer nos analistas da sua empresa."}), 403
    if not _limpar_bloqueio(alvo):
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    print(f"[LOGIN] {g.usuario} desbloqueou {alvo}", flush=True)
    return jsonify({"ok": True})


# ---------- Empresas ----------
# Cada analista pertence a uma empresa (ex.: MAP DE CAMPOS). Na aba Uso o
# ADM escolhe a empresa e vê só os analistas dela. A empresa pode ter a
# pergunta "é Leme?" na Renda (em Leme o adiantamento não é somado).
EMPRESA_INICIAL = "MAP DE CAMPOS"
USUARIOS_EMPRESA_INICIAL = ("wallace", "geyce", "mirian")
_EMPRESAS = {"empresas": None, "de": None, "quando": 0.0}


def _nome_empresa(valor):
    return re.sub(r"\s+", " ", _tirar_acentos(str(valor or "")).strip().upper())[:60]


def _dados_empresas(atualizar=False):
    """({empresa: pergunta_leme}, {usuario: empresa}). Sem banco: só a inicial."""
    padrao = ({EMPRESA_INICIAL: True}, {u: EMPRESA_INICIAL for u in USUARIOS_EMPRESA_INICIAL})
    if not os.getenv("DATABASE_URL", "").strip():
        return padrao
    agora = time.time()
    if atualizar or _EMPRESAS["empresas"] is None or agora - _EMPRESAS["quando"] > 60:
        try:
            def ler(con):
                return (con.run("SELECT nome, pergunta_leme FROM empresas") or [],
                        con.run("SELECT usuario, empresa FROM empresa_usuarios") or [])
            emp, de = _no_banco(ler)
            _EMPRESAS["empresas"] = {str(n): bool(l) for n, l in emp}
            _EMPRESAS["de"] = {str(u): str(e) for u, e in de}
            _EMPRESAS["quando"] = agora
        except Exception as e:
            print(f"[EMPRESA] banco indisponível ({type(e).__name__})", flush=True)
    if _EMPRESAS["empresas"] is None:
        return padrao
    return _EMPRESAS["empresas"], _EMPRESAS["de"]


def _empresa_de(usuario):
    return _dados_empresas()[1].get(str(usuario or "").lower(), "")


def _sem_empresa(usuario):
    """
    True quando o usuário NÃO está em nenhuma empresa (e não é o dono): não
    pode entrar. Se o banco estiver fora do ar e não houver lista conhecida,
    não barra ninguém (melhor não trancar o escritório inteiro).
    """
    u = str(usuario or "").lower()
    if not os.getenv("DATABASE_URL", "").strip() or _eh_dono(u):
        return False
    _dados_empresas()
    if _EMPRESAS["de"] is None:
        return False
    return not _EMPRESAS["de"].get(u)


MSG_SEM_EMPRESA = ("Seu usuário ainda não está cadastrado em uma empresa. "
                   "Fale com o administrador.")


def _pergunta_leme(usuario):
    empresas, de = _dados_empresas()
    return bool(empresas.get(de.get(str(usuario or "").lower(), ""), False))


# Dono do sistema: o único que vê todas as empresas, cria empresas e move
# analistas entre elas. Outro ADM (de uma empresa cliente) só vê a dele.
DONOS_SISTEMA = ("wallace", "camila")
DONO_SISTEMA = DONOS_SISTEMA[0]


def _eh_dono(usuario=None):
    u = str(usuario if usuario is not None else getattr(g, "usuario", "")).lower()
    return u in DONOS_SISTEMA and u in _usuarios()


def _pode_gerir(alvo):
    """O ADM logado pode mexer neste analista? Dono: todos; outro ADM: só da empresa dele."""
    if _eh_dono():
        return True
    alvo = str(alvo or "").lower()
    # ADM de empresa só mexe em ANALISTAS da empresa dele: nunca no dono nem
    # em outro ADM (a própria senha ele troca em "Trocar senha").
    if alvo in DONOS_SISTEMA or _usuarios().get(alvo, {}).get("admin"):
        return False
    meu = _empresa_de(getattr(g, "usuario", ""))
    return bool(meu) and _empresa_de(alvo) == meu


def _so_admin():
    if not _eh_dono():
        return jsonify({"ok": False, "error": "Você não tem permissão para isso."}), 403
    if not os.getenv("DATABASE_URL", "").strip():
        return jsonify({"ok": False, "error": "Sem banco configurado no servidor."}), 503
    return None


@app.post("/empresa")
def salvar_empresa():
    """ADM cria uma empresa ou liga/desliga a pergunta de Leme dela."""
    erro = _so_admin()
    if erro:
        return erro
    dados = request.get_json(silent=True) or {}
    nome = _nome_empresa(dados.get("nome"))
    if len(nome) < 2:
        return jsonify({"ok": False, "error": "Digite o nome da empresa."}), 400
    leme = _sim(dados.get("pergunta_leme"))
    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO empresas (nome, pergunta_leme) VALUES (:nome, :leme)"
            " ON CONFLICT (nome) DO UPDATE SET pergunta_leme = EXCLUDED.pergunta_leme",
            nome=nome, leme=leme))
    except Exception as e:
        print(f"[EMPRESA] não gravou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _dados_empresas(atualizar=True)
    print(f"[EMPRESA] {g.usuario} salvou {nome} (Leme={leme})", flush=True)
    return jsonify({"ok": True, "nome": nome})


@app.post("/empresa-do-analista")
def empresa_do_analista():
    """ADM coloca o analista numa empresa ("" = sem empresa)."""
    erro = _so_admin()
    if erro:
        return erro
    dados = request.get_json(silent=True) or {}
    alvo = str(dados.get("usuario") or "").strip().lower()
    if alvo not in _usuarios():
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    empresa = _nome_empresa(dados.get("empresa"))
    if empresa and empresa not in _dados_empresas(atualizar=True)[0]:
        return jsonify({"ok": False, "error": "Empresa não encontrada."}), 404
    try:
        if empresa:
            _no_banco(lambda con: con.run(
                "INSERT INTO empresa_usuarios (usuario, empresa) VALUES (:usuario, :empresa)"
                " ON CONFLICT (usuario) DO UPDATE SET empresa = EXCLUDED.empresa,"
                " atualizado_em = CURRENT_TIMESTAMP", usuario=alvo, empresa=empresa))
        else:
            _no_banco(lambda con: con.run(
                "DELETE FROM empresa_usuarios WHERE usuario = :usuario", usuario=alvo))
    except Exception as e:
        print(f"[EMPRESA] não gravou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _dados_empresas(atualizar=True)
    print(f"[EMPRESA] {g.usuario} colocou {alvo} em {empresa or '(sem empresa)'}", flush=True)
    return jsonify({"ok": True})


@app.post("/cadastrar-usuario")
def cadastrar_usuario():
    """Dono cadastra um usuário novo já numa empresa."""
    erro = _so_admin()
    if erro:
        return erro
    dados = request.get_json(silent=True) or {}
    usuario = str(dados.get("usuario") or "").strip().lower()
    nome = re.sub(r"\s+", " ", str(dados.get("nome") or "")).strip()[:80]
    senha = str(dados.get("senha_inicial") or "")
    empresa = _nome_empresa(dados.get("empresa"))
    admin = _sim(dados.get("admin"))
    if len(nome) < 2:
        return jsonify({"ok": False, "error": "Digite o nome da pessoa."}), 400
    if not LOGIN_VALIDO.match(usuario):
        return jsonify({"ok": False, "error": "Login com 3 a 30 letras minúsculas, números, ponto, traço ou _ (sem espaço e sem acento)."}), 400
    if usuario in _usuarios_todos():
        if usuario in _excluidos(atualizar=True):
            return jsonify({"ok": False, "error": "Esse login já existe e está excluído: use Restaurar em \"Logins excluídos\"."}), 409
        return jsonify({"ok": False, "error": "Já existe um usuário com esse login."}), 409
    if len(senha) < SENHA_MINIMO or len(senha) > 72:
        return jsonify({"ok": False, "error": f"A senha inicial precisa ter pelo menos {SENHA_MINIMO} caracteres."}), 400
    if senha.strip().lower() == usuario:
        return jsonify({"ok": False, "error": "A senha inicial não pode ser igual ao login."}), 400
    if not empresa or empresa not in _dados_empresas(atualizar=True)[0]:
        return jsonify({"ok": False, "error": "Escolha a empresa do usuário."}), 400

    def gravar(con):
        con.run("INSERT INTO usuarios_cadastrados (usuario, nome, senha_inicial, admin, criado_por)"
                " VALUES (:usuario, :nome, :senha, :admin, :por)",
                usuario=usuario, nome=nome, senha=_hash_senha(senha), admin=admin, por=g.usuario)
        con.run("INSERT INTO empresa_usuarios (usuario, empresa) VALUES (:usuario, :empresa)"
                " ON CONFLICT (usuario) DO UPDATE SET empresa = EXCLUDED.empresa,"
                " atualizado_em = CURRENT_TIMESTAMP", usuario=usuario, empresa=empresa)
        # Restos de um login antigo com o mesmo nome (não deveria haver).
        for tabela in ("senhas_usuarios", "sessoes_ativas", "bloqueios_login"):
            con.run(f"DELETE FROM {tabela} WHERE usuario = :usuario", usuario=usuario)
    try:
        _no_banco(gravar)
    except Exception as e:
        print(f"[CADASTRO] falhou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui salvar agora (o login pode já existir). Tente de novo em 1 minuto."}), 503
    _usuarios_do_banco(atualizar=True)
    _dados_empresas(atualizar=True)
    _senhas_proprias(atualizar=True)
    print(f"[CADASTRO] {g.usuario} cadastrou {usuario} em {empresa}" + (" (ADM)" if admin else ""), flush=True)
    return jsonify({"ok": True, "usuario": usuario})


@app.post("/tornar-admin")
def tornar_admin():
    """Dono liga/desliga o ADM de um usuário (ADM da empresa dele)."""
    erro = _so_admin()
    if erro:
        return erro
    dados = request.get_json(silent=True) or {}
    alvo = str(dados.get("usuario") or "").strip().lower()
    if alvo not in _usuarios():
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    if alvo in DONOS_SISTEMA:
        return jsonify({"ok": False, "error": "Esse usuário é sempre ADM."}), 400
    adm = _sim(dados.get("admin"))
    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO papeis_usuarios (usuario, admin) VALUES (:usuario, :admin)"
            " ON CONFLICT (usuario) DO UPDATE SET admin = EXCLUDED.admin, atualizado_em = CURRENT_TIMESTAMP",
            usuario=alvo, admin=adm))
    except Exception as e:
        print(f"[PAPEL] não gravou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _papeis(atualizar=True)
    print(f"[PAPEL] {g.usuario} deixou {alvo} como {'ADM' if adm else 'analista'}", flush=True)
    return jsonify({"ok": True})


@app.post("/excluir-usuario")
def excluir_usuario():
    """
    Exclui um login: não entra mais; apaga senha, sessão, bloqueio e empresa.
    Dono: qualquer um. ADM de empresa: só analistas (não ADM) da empresa dele.
    """
    conta = _usuarios().get(getattr(g, "usuario", ""), {})
    if not conta.get("admin"):
        return jsonify({"ok": False, "error": "Só o administrador pode excluir."}), 403
    if not os.getenv("DATABASE_URL", "").strip():
        return jsonify({"ok": False, "error": "Sem banco configurado no servidor."}), 503
    alvo = str((request.get_json(silent=True) or {}).get("usuario") or "").strip().lower()
    conta_alvo = _usuarios().get(alvo)
    if not conta_alvo:
        return jsonify({"ok": False, "error": "Usuário não encontrado."}), 404
    if alvo == g.usuario:
        return jsonify({"ok": False, "error": "Você não pode excluir o seu próprio login."}), 400
    if alvo in DONOS_SISTEMA:
        return jsonify({"ok": False, "error": "Esse login não pode ser excluído."}), 403
    if not _eh_dono():
        if conta_alvo.get("admin"):
            return jsonify({"ok": False, "error": "Você não pode excluir um ADM."}), 403
        if not _pode_gerir(alvo):
            return jsonify({"ok": False, "error": "Este analista é de outra empresa."}), 403

    def apagar(con):
        con.run("INSERT INTO usuarios_excluidos (usuario, excluido_por) VALUES (:usuario, :por)"
                " ON CONFLICT (usuario) DO NOTHING", usuario=alvo, por=g.usuario)
        for tabela in ("senhas_usuarios", "sessoes_ativas", "bloqueios_login", "empresa_usuarios", "papeis_usuarios"):
            con.run(f"DELETE FROM {tabela} WHERE usuario = :usuario", usuario=alvo)
    try:
        _no_banco(apagar)
    except Exception as e:
        print(f"[EXCLUIR] falhou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _SESSOES.pop(alvo, None)
    _BLOQUEIOS.pop(alvo, None)
    _excluidos(atualizar=True)
    _papeis(atualizar=True)
    _senhas_proprias(atualizar=True)
    _dados_empresas(atualizar=True)
    print(f"[EXCLUIR] {g.usuario} excluiu o login {alvo}", flush=True)
    return jsonify({"ok": True})


@app.post("/restaurar-usuario")
def restaurar_usuario():
    """Dono desfaz a exclusão: volta com a senha inicial e sem empresa."""
    erro = _so_admin()
    if erro:
        return erro
    alvo = str((request.get_json(silent=True) or {}).get("usuario") or "").strip().lower()
    if alvo not in _usuarios_todos():
        return jsonify({"ok": False, "error": "Esse login não existe mais na lista do servidor."}), 404
    try:
        _no_banco(lambda con: con.run("DELETE FROM usuarios_excluidos WHERE usuario = :usuario", usuario=alvo))
    except Exception as e:
        print(f"[EXCLUIR] restaurar falhou ({type(e).__name__})", flush=True)
        return jsonify({"ok": False, "error": "Não consegui falar com o banco agora. Tente de novo em 1 minuto."}), 503
    _excluidos(atualizar=True)
    print(f"[EXCLUIR] {g.usuario} restaurou o login {alvo}", flush=True)
    return jsonify({"ok": True})


# ---------- Bloqueio após 3 senhas erradas ----------
# Vale para os analistas (o ADM fica de fora). Conta erros seguidos de
# qualquer computador; acertar a senha zera. No 3º erro o usuário fica
# bloqueado até o ADM desbloquear/resetar na aba Uso (ou apagando a linha
# na tabela bloqueios_login do banco). Fica no banco para sobreviver ao
# servidor reiniciar; sem banco, fica na memória.
LIMITE_ERROS_LOGIN = 3
MSG_BLOQUEADO = ("Acesso bloqueado após 3 senhas erradas. "
                 "Fale com o administrador para liberar.")
_BLOQUEIOS = {}  # usuario -> erros seguidos (memória / cópia do banco)


def _bloqueio(usuario):
    """(erros seguidos, bloqueado?)."""
    if os.getenv("DATABASE_URL", "").strip():
        try:
            linhas = _no_banco(lambda con: con.run(
                "SELECT erros FROM bloqueios_login WHERE usuario = :usuario", usuario=usuario)) or []
            _BLOQUEIOS[usuario] = int(linhas[0][0]) if linhas else 0
        except Exception as e:
            print(f"[LOGIN] banco indisponível ({type(e).__name__}); bloqueio pela memória", flush=True)
    erros = _BLOQUEIOS.get(usuario, 0)
    return erros, erros >= LIMITE_ERROS_LOGIN


def _registrar_erro_login(usuario):
    """Soma 1 erro e devolve (erros seguidos, bloqueado?)."""
    erros = _BLOQUEIOS.get(usuario, 0) + 1
    if os.getenv("DATABASE_URL", "").strip():
        try:
            linhas = _no_banco(lambda con: con.run(
                "INSERT INTO bloqueios_login (usuario, erros) VALUES (:usuario, 1)"
                " ON CONFLICT (usuario) DO UPDATE SET erros = bloqueios_login.erros + 1,"
                " atualizado_em = CURRENT_TIMESTAMP RETURNING erros",
                usuario=usuario)) or []
            if linhas:
                erros = int(linhas[0][0])
        except Exception as e:
            print(f"[LOGIN] erro não gravado no banco ({type(e).__name__})", flush=True)
    _BLOQUEIOS[usuario] = erros
    return erros, erros >= LIMITE_ERROS_LOGIN


def _limpar_bloqueio(usuario):
    """Zera os erros/desbloqueia. False se não deu para gravar no banco."""
    _BLOQUEIOS.pop(usuario, None)
    if os.getenv("DATABASE_URL", "").strip():
        try:
            _no_banco(lambda con: con.run(
                "DELETE FROM bloqueios_login WHERE usuario = :usuario", usuario=usuario))
        except Exception as e:
            print(f"[LOGIN] desbloqueio não gravado ({type(e).__name__})", flush=True)
            return False
    return True


def _bloqueados():
    """Lista dos usuários bloqueados agora."""
    if os.getenv("DATABASE_URL", "").strip():
        try:
            linhas = _no_banco(lambda con: con.run(
                "SELECT usuario, erros FROM bloqueios_login")) or []
            _BLOQUEIOS.clear()
            _BLOQUEIOS.update({str(u): int(n) for u, n in linhas})
        except Exception as e:
            print(f"[LOGIN] banco indisponível ({type(e).__name__})", flush=True)
    return sorted(u for u, n in _BLOQUEIOS.items() if n >= LIMITE_ERROS_LOGIN)


# ---------- Contador de leituras por usuário ----------
# Com DATABASE_URL (Postgres do Neon) os números ficam guardados no banco;
# sem ele (ou se o banco falhar), ficam na memória e zeram ao reiniciar.
# Só guarda contagens (dia, usuário, tipo, quantidade) — nenhum dado de
# documento ou cliente.
_USO = defaultdict(int)
_USO_DESDE = datetime.now(FUSO_BR).strftime("%d/%m/%Y %H:%M")
_banco_pronto = {"tabela": False}


def _conectar_banco():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    import ssl
    import urllib.parse
    import pg8000.native  # puro Python (sem compilar)

    u = urllib.parse.urlparse(url)
    return pg8000.native.Connection(
        user=urllib.parse.unquote(u.username or ""),
        password=urllib.parse.unquote(u.password or ""),
        host=u.hostname,
        port=u.port or 5432,
        database=(u.path or "/").lstrip("/") or "postgres",
        ssl_context=ssl.create_default_context(),
        timeout=15,
    )


def _no_banco(funcao):
    """Abre a conexão, roda e fecha. Devolve None se não houver banco."""
    con = _conectar_banco()
    if con is None:
        return None
    try:
        if not _banco_pronto["tabela"]:
            con.run(
                "CREATE TABLE IF NOT EXISTS uso_leituras ("
                " dia DATE NOT NULL, usuario TEXT NOT NULL, tipo TEXT NOT NULL,"
                " quantidade INTEGER NOT NULL DEFAULT 0,"
                " PRIMARY KEY (dia, usuario, tipo))"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS sessoes_ativas ("
                " usuario TEXT PRIMARY KEY, sid TEXT NOT NULL,"
                " criada_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS senhas_usuarios ("
                " usuario TEXT PRIMARY KEY, senha TEXT NOT NULL,"
                " trocada_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS empresas ("
                " nome TEXT PRIMARY KEY, pergunta_leme BOOLEAN NOT NULL DEFAULT FALSE,"
                " criada_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS empresa_usuarios ("
                " usuario TEXT PRIMARY KEY, empresa TEXT NOT NULL,"
                " atualizado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            # Primeira empresa e seus analistas. Só entram se ainda não
            # existirem: o que o ADM mudar depois continua como ele deixou.
            con.run("INSERT INTO empresas (nome, pergunta_leme) VALUES (:nome, TRUE)"
                    " ON CONFLICT (nome) DO NOTHING", nome=EMPRESA_INICIAL)
            for u in USUARIOS_EMPRESA_INICIAL:
                con.run("INSERT INTO empresa_usuarios (usuario, empresa) VALUES (:usuario, :empresa)"
                        " ON CONFLICT (usuario) DO NOTHING", usuario=u, empresa=EMPRESA_INICIAL)
            con.run(
                "CREATE TABLE IF NOT EXISTS usuarios_cadastrados ("
                " usuario TEXT PRIMARY KEY, nome TEXT NOT NULL, senha_inicial TEXT NOT NULL,"
                " admin BOOLEAN NOT NULL DEFAULT FALSE, criado_por TEXT NOT NULL DEFAULT '',"
                " criado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS papeis_usuarios ("
                " usuario TEXT PRIMARY KEY, admin BOOLEAN NOT NULL,"
                " atualizado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS usuarios_excluidos ("
                " usuario TEXT PRIMARY KEY, excluido_por TEXT NOT NULL DEFAULT '',"
                " excluido_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            con.run(
                "CREATE TABLE IF NOT EXISTS bloqueios_login ("
                " usuario TEXT PRIMARY KEY, erros INTEGER NOT NULL DEFAULT 0,"
                " atualizado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            _banco_pronto["tabela"] = True
        return funcao(con)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _gravar_uso_no_banco(dia, usuario, tipo):
    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO uso_leituras (dia, usuario, tipo, quantidade)"
            " VALUES (CAST(:dia AS DATE), :usuario, :tipo, 1)"
            " ON CONFLICT (dia, usuario, tipo)"
            " DO UPDATE SET quantidade = uso_leituras.quantidade + 1",
            dia=dia, usuario=usuario, tipo=tipo,
        ))
    except Exception as e:
        print(f"[USO] banco indisponível ({type(e).__name__}); contagem só na memória", flush=True)


def _registrar_uso(tipo):
    usuario = getattr(g, "usuario", "") or "sem-login"
    dia = datetime.now(FUSO_BR).strftime("%Y-%m-%d")
    _USO[(dia, usuario, tipo)] += 1
    print(f"[USO] usuario={usuario} tipo={tipo}", flush=True)
    if os.getenv("DATABASE_URL", "").strip():
        # Em segundo plano: a leitura não espera o banco (o Neon "acorda"
        # em até 1 s depois de 5 min parado).
        import threading
        threading.Thread(target=_gravar_uso_no_banco, args=(dia, usuario, tipo), daemon=True).start()


@app.get("/uso")
def uso():
    if _login_ligado():
        conta = _usuarios().get(getattr(g, "usuario", ""), {})
        if not conta.get("admin"):
            return jsonify({"ok": False, "error": "Só o administrador vê o contador."}), 403
    nomes = {u: str(c.get("nome") or u) for u, c in _usuarios().items()}
    # Quem já criou a senha própria (True) e quem ainda usa a inicial (False).
    proprias = _senhas_proprias(atualizar=True)
    senhas = None if proprias is None else {u: u in proprias for u in nomes}
    bloqueados = [u for u in _bloqueados() if u in nomes]
    empresas, empresa_de = _dados_empresas(atualizar=True)
    dono = _eh_dono()
    if not dono:
        # ADM de empresa cliente: só os analistas e os números da empresa dele.
        minha = empresa_de.get(getattr(g, "usuario", ""), "")
        nomes = {u: n for u, n in nomes.items() if minha and empresa_de.get(u) == minha}
        senhas = None if senhas is None else {u: v for u, v in senhas.items() if u in nomes}
        bloqueados = [u for u in bloqueados if u in nomes]
        empresas = {minha: empresas.get(minha, False)} if minha else {}
    lista_empresas = [{"nome": n, "pergunta_leme": l} for n, l in sorted(empresas.items())]
    empresa_de = {u: empresa_de.get(u, "") for u in nomes}
    contas = _usuarios()
    admins = {u: bool(contas.get(u, {}).get("admin")) for u in nomes}
    todos = _usuarios_todos()
    excluidos = [{"usuario": u, "nome": str(todos[u].get("nome") or u)}
                 for u in sorted(_excluidos(atualizar=True)) if u in todos] if dono else []

    # Banco: este mês e o anterior.
    hoje = datetime.now(FUSO_BR).date()
    inicio = (hoje.replace(day=1) - timedelta(days=1)).replace(day=1)
    try:
        rows = _no_banco(lambda con: con.run(
            "SELECT CAST(dia AS TEXT), usuario, tipo, quantidade FROM uso_leituras"
            " WHERE dia >= CAST(:inicio AS DATE) ORDER BY dia, usuario, tipo",
            inicio=inicio.isoformat(),
        ))
        if rows is not None:
            linhas = [
                {"dia": str(r[0])[:10], "usuario": r[1], "tipo": r[2], "quantidade": int(r[3])}
                for r in rows if dono or r[1] in nomes
            ]
            return jsonify({"ok": True, "persistente": True, "desde": "", "linhas": linhas,
                            "nomes": nomes, "senhas": senhas, "bloqueados": bloqueados,
                            "empresas": lista_empresas, "empresa_de": empresa_de, "dono": dono,
                            "excluidos": excluidos, "admins": admins,
                            "donos": list(DONOS_SISTEMA) if dono else []})
    except Exception as e:
        print(f"[USO] leitura do banco falhou ({type(e).__name__})", flush=True)

    linhas = [
        {"dia": dia, "usuario": usuario, "tipo": tipo, "quantidade": n}
        for (dia, usuario, tipo), n in sorted(_USO.items()) if dono or usuario in nomes
    ]
    return jsonify({"ok": True, "persistente": False, "desde": _USO_DESDE, "linhas": linhas,
                    "nomes": nomes, "senhas": senhas, "bloqueados": bloqueados,
                            "empresas": lista_empresas, "empresa_de": empresa_de, "dono": dono,
                            "excluidos": excluidos, "admins": admins,
                            "donos": list(DONOS_SISTEMA) if dono else []})


# ============================================================
# PREPARO DA IMAGEM (precisão de leitura)
# A Groq cobra o mesmo número de tokens por imagem, seja qual for o
# tamanho (o Qwen reduz tudo para a mesma resolução). Por isso, quanto
# mais a imagem for "só o documento", maiores ficam as letras para a IA.
# ============================================================

ALVO_LADO_DOCUMENTO = 1600


def _abrir_imagem(data: bytes) -> Image.Image:
    """Abre a imagem respeitando a rotação gravada pelo celular (EXIF)."""
    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def _ordenar_cantos(pts):
    """Ordena 4 pontos: sup-esq, sup-dir, inf-dir, inf-esq."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # Ordena pelo ângulo em volta do centro (funciona mesmo girado a 45°)
    # e começa pelo canto mais próximo do topo-esquerdo.
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(angulos)]
    inicio = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -inicio, axis=0).astype(np.float32)


def _quad_de_contorno(contorno, preenchimento_min=0.85):
    peri = cv2.arcLength(contorno, True)
    if peri <= 0:
        return None
    aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
    if len(aprox) == 4 and cv2.isContourConvex(aprox):
        return aprox.reshape(4, 2).astype(np.float32)
    # Cartões têm cantos arredondados: aceita um contorno "quase retângulo".
    rect = cv2.minAreaRect(contorno)
    rw, rh = rect[1]
    if rw * rh <= 0:
        return None
    if cv2.contourArea(contorno) / (rw * rh) >= preenchimento_min:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def _medidas_quad(q):
    q = _ordenar_cantos(q)
    larg = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    alt = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    return float(larg), float(alt)


def _iou_caixas(a, b):
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    uni = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / uni if uni > 0 else 0.0


def _mapa_texto(cinza):
    """Pixels com cara de letra/número (escuro no claro e claro no escuro)."""
    k = max(9, int(min(cinza.shape) / 40)) | 1
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    escuro = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, nucleo)
    claro = cv2.morphologyEx(cinza, cv2.MORPH_TOPHAT, nucleo)
    _, texto = cv2.threshold(cv2.max(escuro, claro), 40, 255, cv2.THRESH_BINARY)
    return texto


def _texto_fica_dentro(quad, texto):
    """
    Garante que o recorte não deixa texto do documento de fora:
    a densidade de "letras" fora do quadrilátero tem de ser bem menor
    que a de dentro.
    """
    mascara = np.zeros(texto.shape, np.uint8)
    cv2.fillConvexPoly(mascara, _ordenar_cantos(quad).astype(np.int32), 255)
    dentro = mascara > 0
    area_in = int(dentro.sum())
    area_out = int(dentro.size - area_in)
    if area_in == 0:
        return False
    tx = texto > 0
    dens_in = tx[dentro].sum() / area_in
    if dens_in < 0.01:
        return False
    if area_out == 0:
        return True
    dens_out = tx[~dentro].sum() / area_out
    return dens_out <= 0.35 * dens_in


def _localizar_documento(img: Image.Image):
    """
    Procura o contorno do documento na foto.
    Devolve (4 cantos na escala da imagem original, fração da área) ou None.
    Critérios conservadores: na dúvida, não recorta.
    """
    W, H = img.size
    escala = min(1.0, 1000.0 / max(W, H))
    peq = img if escala >= 1.0 else img.resize(
        (max(1, int(W * escala)), max(1, int(H * escala))),
        Image.Resampling.BILINEAR,
    )
    cinza = np.asarray(peq.convert("L"))
    h, w = cinza.shape
    area_total = float(w * h)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    med = float(np.median(suave))
    texto = _mapa_texto(suave)

    mascaras = []
    # 1) Bordas do cartão.
    for lo, hi in (
        (int(max(0, 0.66 * med)), int(min(255, 1.33 * med))),
        (30, 90),
        (60, 180),
    ):
        bordas = cv2.Canny(suave, lo, hi)
        bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=2)
        bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mascaras.append(bordas)
    # 2) Documento mais claro/escuro ou mais colorido que o fundo.
    sat = cv2.GaussianBlur(
        cv2.cvtColor(np.asarray(peq), cv2.COLOR_RGB2HSV)[:, :, 1], (5, 5), 0
    )
    for canal in (suave, sat):
        _, otsu = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for m in (otsu, 255 - otsu):
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            mascaras.append(m)
    # 3) Mancha de texto (útil quando a borda do cartão quase não aparece).
    k = max(15, int(min(h, w) / 25)) | 1
    mancha = cv2.morphologyEx(texto, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mancha = cv2.morphologyEx(mancha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    candidatos = []
    # A mancha de texto é irregular (foto 3x4, campos vazios): aceita menos
    # preenchimento; as travas de proporção e de "texto dentro" seguem valendo.
    for m, preench in [(x, 0.85) for x in mascaras] + [(mancha, 0.6)]:
        contornos, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:12]:
            q = _quad_de_contorno(c, preench)
            if q is None:
                continue
            area = float(cv2.contourArea(_ordenar_cantos(q)))
            frac = area / area_total
            # Documento já grande na foto: o ganho é pequeno e o risco de
            # cortar um pedaço não compensa.
            if not (0.03 <= frac <= 0.6):
                continue
            larg, alt = _medidas_quad(q)
            if min(larg, alt) < 30:
                continue
            razao = max(larg, alt) / min(larg, alt)
            # Cartão/painel de documento (~1,4 a 1,6) ou dois painéis lado a lado.
            if not (1.2 <= razao <= 1.95 or 2.5 <= razao <= 3.6):
                continue
            if any(_iou_caixas(q, c2) > 0.9 for _, c2 in candidatos):
                continue
            candidatos.append((area, q))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[0], reverse=True)

    # Frente e verso lado a lado: junta dois painéis parecidos e vizinhos.
    def _juntar(area, melhor):
        larg, alt = _medidas_quad(melhor)
        centro = melhor.mean(axis=0)
        for area2, q2 in candidatos:
            if q2 is melhor or not (0.6 <= area2 / area <= 1.6):
                continue
            if _iou_caixas(melhor, q2) > 0.2:
                continue
            if np.linalg.norm(q2.mean(axis=0) - centro) > 1.5 * max(larg, alt):
                continue
            juntos = cv2.boxPoints(
                cv2.minAreaRect(np.vstack([melhor, q2]))
            ).astype(np.float32)
            area_j = float(cv2.contourArea(_ordenar_cantos(juntos)))
            if area_j / area_total <= 0.7:
                return area_j, juntos
        return area, melhor

    for area, q in candidatos:
        area, q = _juntar(area, q)
        if _texto_fica_dentro(q, texto):
            return q / escala, area / area_total
    return None


def _recortar_documento(img: Image.Image, frac_min: float = 0.0):
    """
    Recorta e endireita o documento quando ele está pequeno/inclinado na foto.
    Se não tiver certeza, devolve a imagem original.
    frac_min: só recorta se o achado ocupar ao menos essa fração da foto
    (certidão: evita recortar um quadro de dentro da folha).
    """
    if cv2 is None or np is None:
        return img, "sem_opencv"
    try:
        achado = _localizar_documento(img)
        if not achado:
            return img, "sem_recorte"
        quad, frac = achado
        if frac < frac_min:
            return img, "sem_recorte"
        quad = _ordenar_cantos(quad)

        # Margem de segurança para não cortar a borda do documento.
        centro = quad.mean(axis=0)
        quad = centro + (quad - centro) * 1.05
        quad[:, 0] = np.clip(quad[:, 0], 0, img.width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, img.height - 1)

        larg, alt = _medidas_quad(quad)
        larg, alt = int(round(larg)), int(round(alt))
        if larg < 60 or alt < 60:
            return img, "sem_recorte"
        destino = np.array(
            [[0, 0], [larg - 1, 0], [larg - 1, alt - 1], [0, alt - 1]],
            dtype=np.float32,
        )
        matriz = cv2.getPerspectiveTransform(quad.astype(np.float32), destino)
        arr = cv2.warpPerspective(
            np.asarray(img), matriz, (larg, alt),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        recorte = Image.fromarray(arr)

        # Recorte "vazio" (ex.: folha em branco): descarta.
        if float(np.asarray(recorte.convert("L")).std()) < 18:
            return img, "sem_recorte"

        # Documento pequeno na foto: amplia com qualidade e realça as letras.
        maior = max(recorte.size)
        if maior < ALVO_LADO_DOCUMENTO:
            fator = ALVO_LADO_DOCUMENTO / maior
            recorte = recorte.resize(
                (int(recorte.width * fator), int(recorte.height * fator)),
                Image.Resampling.LANCZOS,
            )
            if fator >= 1.3:
                recorte = recorte.filter(
                    ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=2)
                )
        return recorte, f"recortado({frac:.0%})"
    except Exception:
        return img, "erro_recorte"


def _texto_de_lado(img):
    """
    O texto do documento está de lado (girado 90°)? Olha as letras: em cada
    linha de texto, a letra vizinha mais próxima fica do lado (horizontal);
    com o documento de lado, fica em cima/embaixo. Devolve True, False ou
    None (poucas letras para decidir).
    """
    if cv2 is None or np is None:
        return None
    try:
        W, H = img.size
        e = min(1.0, 1400 / max(W, H))
        peq = img if e >= 1.0 else img.resize((max(1, int(W * e)), max(1, int(H * e))), Image.Resampling.BILINEAR)
        g = np.asarray(peq.convert("L"))
        tinta = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)
        n, _, st, cen = cv2.connectedComponentsWithStats(tinta, connectivity=8)
        w, h, a = st[1:, 2], st[1:, 3], st[1:, 4]
        maior, menor = np.maximum(w, h), np.maximum(1, np.minimum(w, h))
        letra = (a >= 15) & (maior >= 5) & (maior <= 45) & (maior <= 5 * menor) & (a <= 0.9 * w * h)
        idx = np.where(letra)[0] + 1
        if len(idx) < 30:
            return None
        if len(idx) > 2500:
            idx = idx[np.linspace(0, len(idx) - 1, 2500).astype(int)]
        pts = cen[idx]
        tam = np.maximum(st[idx, 2], st[idx, 3]).astype(float)
        hor = ver = 0
        for i in range(len(pts)):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            d[i] = 1e9
            ok = (tam > 0.6 * tam[i]) & (tam < 1.7 * tam[i]) & (d < 2.2 * tam[i])
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, d, 1e9)))
            dx, dy = abs(pts[j, 0] - pts[i, 0]), abs(pts[j, 1] - pts[i, 1])
            if dx > 2.5 * dy:
                hor += 1
            elif dy > 2.5 * dx:
                ver += 1
        if ver >= 25 and ver > 1.8 * hor:
            return True
        if hor >= 25 and hor > 1.8 * ver:
            return False
        return None
    except Exception:
        return None


def _pdf_area_documento(page):
    """
    PDF digital (ex.: CNH exportada do app): o cartão vem como imagens
    coloridas grandes e o resto da página é cabeçalho/QR code.
    Devolve (retângulo só com o cartão, zoom ideal) ou (None, None).
    """
    try:
        pagina = page.rect
        area_pag = pagina.width * pagina.height
        imagens = []
        for info in page.get_image_info():
            larg_px = int(info.get("width") or 0)
            alt_px = int(info.get("height") or 0)
            comps = int(info.get("colorspace") or 0)
            bpc = int(info.get("bpc") or 0)
            if larg_px < 600 or alt_px < 350 or comps < 3 or bpc < 8:
                continue
            caixa = pymupdf.Rect(info["bbox"]) & pagina
            if caixa.is_empty or caixa.width < 50 or caixa.height < 30:
                continue
            razao = max(caixa.width, caixa.height) / min(caixa.width, caixa.height)
            if not (1.2 <= razao <= 1.95):
                continue
            imagens.append((caixa, larg_px))
        if not imagens:
            return None, None

        uniao = pymupdf.Rect(imagens[0][0])
        for caixa, _ in imagens[1:]:
            uniao |= caixa
        frac = (uniao.width * uniao.height) / area_pag
        if not (0.08 <= frac <= 0.8):
            return None, None

        # Se houver dado do cliente (data, CPF) escrito FORA das imagens,
        # não recorta, para não perder informação.
        for bloco in page.get_text("blocks"):
            caixa_txt = pymupdf.Rect(bloco[:4])
            if caixa_txt.intersects(uniao):
                continue
            texto = str(bloco[4] or "")
            if re.search(r"\d{2}/\d{2}/\d{4}|\d{3}\.?\d{3}\.?\d{3}-?\d{2}", texto):
                return None, None

        uniao = (uniao + (-6, -6, 6, 6)) & pagina
        zoom = max(lp / cx.width for cx, lp in imagens)
        zoom = min(4.5, max(2.2, zoom))
        return uniao, zoom
    except Exception:
        return None, None


PAINEIS_PDF_DIGITAL = True


def _pdf_paineis(page):
    """
    PDF digital do gov.br (CNH/CIN): o cartão vem como 2-4 imagens (painéis)
    pequenas numa página A4. Devolve os painéis (de cima para baixo) na
    resolução original, ou None quando a página não é desse tipo.
    """
    pagina = page.rect
    paineis = []
    for info in page.get_image_info():
        larg_px = int(info.get("width") or 0)
        alt_px = int(info.get("height") or 0)
        comps = int(info.get("colorspace") or 0)
        bpc = int(info.get("bpc") or 0)
        if larg_px < 600 or alt_px < 350 or comps < 3 or bpc < 8:
            continue
        caixa = pymupdf.Rect(info["bbox"]) & pagina
        if caixa.is_empty or caixa.width < 50 or caixa.height < 30:
            continue
        razao = max(caixa.width, caixa.height) / min(caixa.width, caixa.height)
        if not (1.2 <= razao <= 1.95):
            continue
        paineis.append((caixa, larg_px))
    if not (2 <= len(paineis) <= 4):
        return None

    # Dado do cliente escrito como texto FORA dos painéis: não arrisca.
    for bloco in page.get_text("blocks"):
        caixa_txt = pymupdf.Rect(bloco[:4])
        if any(caixa_txt.intersects(c) for c, _ in paineis):
            continue
        if re.search(r"\d{2}/\d{2}/\d{4}|\d{3}\.?\d{3}\.?\d{3}-?\d{2}", str(bloco[4] or "")):
            return None

    paineis.sort(key=lambda p: (round(p[0].y0), p[0].x0))
    imgs = []
    for caixa, larg_px in paineis:
        zoom = max(1.0, min(5.0, larg_px / caixa.width))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=caixa, alpha=False)
        imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return imgs


def _pdf_paineis_lado_a_lado(page):
    """
    Monta uma faixa com os painéis do PDF digital LADO A LADO, na resolução
    original, sem cabeçalho/QR code: as letras chegam à IA com cerca do
    dobro do tamanho. Qualquer dúvida -> None (usa a página inteira).
    """
    try:
        imgs = _pdf_paineis(page)
        if not imgs:
            return None

        altura = max(i.height for i in imgs)
        imgs = [
            i if i.height == altura else
            i.resize((max(1, int(i.width * altura / i.height)), altura), Image.Resampling.LANCZOS)
            for i in imgs
        ]
        gap = 16
        largura = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
        faixa = Image.new("RGB", (largura, altura), "white")
        x = 0
        for i in imgs:
            faixa.paste(i, (x, 0))
            x += i.width + gap

        # Painéis pequenos (CIN 650x410): amplia até ~1,6 MP, o tamanho que a IA usa.
        fator = min(2.0, (1_600_000 / (largura * altura)) ** 0.5)
        if fator > 1.05:
            faixa = faixa.resize(
                (int(largura * fator), int(altura * fator)), Image.Resampling.LANCZOS
            )
        print(f"[LER] PDF digital: {len(imgs)} painéis lado a lado {faixa.size}", flush=True)
        return faixa
    except Exception as e:
        print(f"[LER] painéis do PDF não montados ({type(e).__name__}); usando a página", flush=True)
        return None


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


# Recorte do PDF digital (só os painéis do cartão) DESLIGADO: no teste real,
# a imagem alta e estreita fez a IA errar uma letra da filiação, enquanto a
# página inteira lia certo. Mantido o envio da página inteira, já validado.
RECORTE_PDF_DIGITAL = False


def _pdf_pagina_para_imagem(page):
    """Renderiza a página. Devolve (imagem, pode_recortar_como_foto)."""
    if PAINEIS_PDF_DIGITAL:
        faixa = _pdf_paineis_lado_a_lado(page)
        if faixa is not None:
            return faixa, False

    clip, zoom = (
        _pdf_area_documento(page) if RECORTE_PDF_DIGITAL else (None, None)
    )
    if clip is not None:
        try:
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False
            )
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            return img.copy(), False
        except Exception:
            pass

    zoom_pagina = 2.2
    try:
        area_pag0 = page.rect.width * page.rect.height
        if any((pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area_pag0
               for i in page.get_image_info()):
            zoom_pagina = 3.0   # scan/foto: aproveita a resolução original
    except Exception:
        pass
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom_pagina, zoom_pagina), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").copy()

    # PDF de foto/scan (uma imagem ocupando a página): trata como foto e tenta
    # recortar o documento. Vale mesmo com camada de texto escondida (OCR de
    # aplicativos de digitalização). PDFs digitais (CNH/CIN do gov.br) têm
    # várias imagens pequenas e não entram aqui.
    pode_recortar = False
    try:
        area_pag = page.rect.width * page.rect.height
        grandes = [
            i for i in page.get_image_info()
            if (pymupdf.Rect(i["bbox"]) & page.rect).get_area() >= 0.6 * area_pag
        ]
        pode_recortar = len(grandes) >= 1
    except Exception:
        pode_recortar = False
    return img, pode_recortar


def _to_jpeg(data: bytes, filename: str, mimetype: str) -> bytes:
    img, _modo = _preparar_imagem(data, filename, mimetype)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=91)
    return out.getvalue()


def _preparar_imagem(data: bytes, filename: str, mimetype: str):
    """
    Devolve (imagem, em_pe). em_pe=True quando o documento recortado ficou
    mais alto que largo: CNH, RG e CIN são "deitados" (paisagem), então ele
    foi fotografado de lado e precisa ser girado.

    Prioriza precisão de leitura para CNH/RG/CIN:
    - foto: corrige a rotação do celular e recorta/endireita o documento;
    - PDF digital: usa só a área do cartão (sem cabeçalho e QR code);
    - PDF escaneado: tratado como foto.
    """
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    em_pe = False
    dividir = False
    paginas_de_lado = None
    if is_pdf:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = []
            de_lado = []
            for i in range(min(2, len(doc))):
                img, pode_recortar = _pdf_pagina_para_imagem(doc[i])
                lado = False
                if pode_recortar:
                    img, info = _recortar_documento(img)
                    recortado = info.startswith("recortado")
                    # De lado? Pelas letras; sem letras suficientes, pela
                    # regra antiga (recorte mais alto que largo).
                    lado = _texto_de_lado(img)
                    if lado is None:
                        lado = recortado and img.height > 1.15 * img.width
                    if lado:
                        em_pe = True
                    elif not recortado and img.height > 1.3 * img.width:
                        # Scan de folha em pé (CamScanner, WhatsApp): o
                        # documento ocupa a folha toda e as letras ficam
                        # pequenas. Vai em 2 metades (cima/baixo).
                        dividir = True
                pages.append(img)
                de_lado.append(lado)
            if len(pages) > 1:
                em_pe = False
                dividir = False
                # Frente e verso fotografados de lado (cartão em pé na foto):
                # vão girados para a posição de leitura (ver _imagens_para_ia).
                if any(de_lado):
                    paginas_de_lado = (pages, de_lado)

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
        img = _abrir_imagem(data)
        img, info = _recortar_documento(img)
        em_pe = _texto_de_lado(img)
        if em_pe is None:
            em_pe = info.startswith("recortado") and img.height > 1.15 * img.width

    max_side = 3200
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize(
            (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    modo = "em_pe" if em_pe else ("dividir" if dividir else "normal")
    if paginas_de_lado:
        modo = "paginas_de_lado"
        img.info["paginas_de_lado"] = paginas_de_lado
    return img, modo


def _jpeg_para_ia(img):
    """Mesmo tamanho/qualidade usados no envio de 1 arquivo."""
    if img.width > 2 * img.height:
        # Faixa de painéis (PDF digital): limita pela área, não pela largura.
        area_max = 2_600_000
        if img.width * img.height > area_max:
            f = (area_max / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * f), int(img.height * f)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=92)
        return out.getvalue()
    max_w = 1650
    if img.width > max_w:
        img = img.resize(
            (max_w, max(1, int(img.height * max_w / img.width))),
            Image.Resampling.LANCZOS,
        )
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=91)
    return out.getvalue()


def _imagens_para_ia(arquivos):
    """
    Lista de imagens para a IA.
    - Normal: 1 imagem (1 ou 2 arquivos combinados).
    - Documento fotografado de lado (1 arquivo): 2 imagens, giradas para um
      lado e para o outro; a IA lê a que estiver na posição certa.
      Custa 1 imagem a mais (2.048 tokens) só nesses casos.
    """
    if len(arquivos) == 1:
        a = arquivos[0]
        img, modo = _preparar_imagem(a["data"], a["filename"], a["mimetype"])
        if modo == "em_pe":
            print("[LER] documento de lado: enviando 2 orientações", flush=True)
            return [
                _jpeg_para_ia(img.rotate(-90, expand=True)),
                _jpeg_para_ia(img.rotate(90, expand=True)),
            ], NOTA_ROTACOES
        if modo == "dividir":
            w, h = img.size
            cima = img.crop((0, 0, w, int(h * 0.54)))
            baixo = img.crop((0, int(h * 0.46), w, h))
            print("[LER] scan de folha em pé: enviando metade de cima e de baixo", flush=True)
            return [_jpeg_para_ia(cima), _jpeg_para_ia(baixo)], NOTA_METADES
        if modo == "paginas_de_lado":
            paginas, de_lado = img.info["paginas_de_lado"]
            imagens = []
            for angulo in (-90, 90):
                giradas = [
                    p.rotate(angulo, expand=True) if lado else p
                    for p, lado in zip(paginas, de_lado)
                ]
                imagens.append(_jpeg_para_ia(_empilhar_paginas(giradas)))
            print("[LER] frente e verso de lado: enviando 2 orientações", flush=True)
            return imagens, NOTA_ROTACOES_PAGINAS
        return [_jpeg_para_ia(img)], ""

    # 2 arquivos (frente + verso): se algum estiver de lado, vão girados.
    preparados = [_preparar_imagem(a["data"], a["filename"], a["mimetype"]) for a in arquivos[:2]]
    if any(modo in ("em_pe", "paginas_de_lado") for _, modo in preparados):
        paginas, de_lado = [], []
        for img, modo in preparados:
            if modo == "paginas_de_lado":
                ps, ls = img.info["paginas_de_lado"]
                paginas += list(ps)
                de_lado += list(ls)
            else:
                paginas.append(img)
                de_lado.append(modo == "em_pe")
        imagens = []
        for angulo in (-90, 90):
            giradas = [p.rotate(angulo, expand=True) if lado else p for p, lado in zip(paginas, de_lado)]
            imagens.append(_jpeg_para_ia(_empilhar_paginas(giradas)))
        print("[LER] frente e verso (2 arquivos) de lado: enviando 2 orientações", flush=True)
        return imagens, NOTA_ROTACOES_PAGINAS
    return [_combinar_arquivos_identificacao(arquivos)], ""


def _empilhar_paginas(paginas, largura=1650, gap=18):
    """Cartões (já deitados) um embaixo do outro, com a mesma largura."""
    prontas = []
    for p in paginas:
        f = largura / p.width
        prontas.append(p.resize((largura, max(1, int(p.height * f))), Image.Resampling.LANCZOS))
    altura = sum(p.height for p in prontas) + gap * (len(prontas) - 1)
    folha = Image.new("RGB", (largura, altura), "white")
    y = 0
    for p in prontas:
        folha.paste(p, (0, y))
        y += p.height + gap
    return folha


NOTA_ROTACOES_PAGINAS = (
    "ATENÇÃO: as 2 imagens a seguir mostram o MESMO documento (frente e verso), "
    "girado em sentidos opostos: numa delas o texto fica de cabeça para baixo. "
    "Leia os dados na imagem em que o texto está na posição normal de leitura."
)

NOTA_ROTACOES = (
    "ATENÇÃO: as 2 imagens a seguir são o MESMO documento, giradas em "
    "sentidos opostos (uma delas está de cabeça para baixo). Leia os "
    "dados na imagem em que o texto está na posição normal de leitura."
)
NOTA_METADES = (
    "ATENÇÃO: as 2 imagens a seguir são a METADE DE CIMA e a METADE DE BAIXO "
    "da MESMA folha escaneada (com uma pequena faixa repetida entre elas). "
    "Leia as duas como um único documento."
)


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


def _groq_call(image_bytes, nota=""):
    """image_bytes: uma imagem (bytes) ou uma lista com 1 ou 2 imagens."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY não configurada no Render.")

    imagens = image_bytes if isinstance(image_bytes, list) else [image_bytes]
    conteudo = [{"type": "text", "text": PROMPT}]
    if nota or len(imagens) > 1:
        conteudo.append({"type": "text", "text": nota or NOTA_ROTACOES})
    for img in imagens:
        conteudo.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(img).decode("ascii")
        }})
    models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
    ]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]

    last_error = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": conteudo}],
            "temperature": 0.0,
            "top_p": 0.8,
            "max_completion_tokens": 480,
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
            body = _groq_abrir(req)
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


def _cpf_valido(cpf):
    """Confere os 2 dígitos verificadores do CPF."""
    c = re.sub(r"\D", "", str(cpf or ""))
    if len(c) != 11 or len(set(c)) == 1:
        return False
    for n in (9, 10):
        soma = sum(int(c[i]) * ((n + 1) - i) for i in range(n))
        if (soma * 10) % 11 % 10 != int(c[n]):
            return False
    return True


def _sinal_cin(d):
    """
    A IA costuma chamar a CIN de "RG" porque o cartão também tem o título
    CARTEIRA DE IDENTIDADE. Sinais que só a CIN tem, conferidos sem IA:
    - zona de leitura mecânica (linhas com '<<<') no verso, que RG antigo
      não tem (a CNH já foi separada antes, pelos dados de CNH);
    - o número do documento é o próprio CPF (sem número de RG separado),
      com VALIDADE impressa ou modelo NOVO.
    Devolve "forte", "fraco" ou "".
    """
    mrz = re.sub(r"[^0-9A-Z<]", "", str(d.get("mrz") or "").upper())
    if mrz.count("<") >= 8 and len(mrz) >= 25 and "BRA" in mrz:
        return "forte"
    cpf = re.sub(r"\D", "", str(d.get("cpf") or ""))
    if not _cpf_valido(cpf):
        return ""
    rg = re.sub(r"[^0-9A-Za-z]", "", str(d.get("rg") or "")).upper()
    modelo = str(d.get("modelo_documento") or "").strip().upper()
    tem_validade = _parse_data_br(d.get("validade")) is not None
    if rg and re.sub(r"\D", "", rg) == cpf and (tem_validade or modelo.startswith("NOV")):
        return "forte"
    if not rg and tem_validade and modelo.startswith("NOV"):
        return "forte"
    if not rg and (tem_validade or modelo.startswith("NOV")):
        return "fraco"
    return ""


def _tipo_documento(d):
    """
    Padroniza o tipo para CNH, RG ou CIN.
    Devolve (tipo, deduzido): deduzido=True quando veio dos campos lidos,
    e não do tipo informado pela IA.
    """
    tipo, deduzido = _tipo_documento_base(d)
    if tipo == "RG":
        sinal = _sinal_cin(d)
        if sinal == "forte":
            return "CIN", False
        if sinal == "fraco":
            return "CIN", True
    return tipo, deduzido


def _tipo_documento_base(d):
    t = _tirar_acentos(str(d.get("tipo_documento") or "")).upper()
    t = re.sub(r"[^A-Z ]", " ", t)
    palavras = set(t.split())
    numero_cnh = re.sub(r"\D", "", str(d.get("numero_cnh") or ""))
    rg = re.sub(r"[^0-9A-Za-z]", "", str(d.get("rg") or ""))
    tem_dados_cnh = len(numero_cnh) == 11 or bool(str(d.get("data_primeira_habilitacao") or "").strip())
    if "CNH" in palavras or "HABILITACAO" in t or "DRIVER" in t:
        # "CNH" sem nenhum dado de CNH, mas com número de RG: é RG.
        if not tem_dados_cnh and rg:
            return "RG", True
        return "CNH", False
    if "CIN" in palavras or "IDENTIDADE NACIONAL" in t:
        return "CIN", False
    if palavras & {"RG", "IDENTIDADE", "REGISTRO"} or "REGISTRO GERAL" in t:
        return "RG", False

    # A IA não informou um tipo conhecido: deduz pelos campos.
    if tem_dados_cnh:
        return "CNH", True
    modelo = str(d.get("modelo_documento") or "").strip().upper()
    cpf = re.sub(r"\D", "", str(d.get("cpf") or ""))
    if modelo.startswith("NOV") and len(cpf) == 11:
        return "CIN", True
    if rg or len(cpf) == 11:
        return "RG", True
    return "", True


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


def _naturalidade_e_pessoa(nat_mun, d):
    """
    True quando a "naturalidade" lida é, na verdade, nome de pessoa: igual ou
    parecida com o nome do titular, do pai ou da mãe, e não é uma cidade
    oficial com esse nome.
    """
    import difflib
    chave = _chave_cidade(nat_mun)
    if not chave:
        return False
    # Existe cidade oficial com esse nome exato? Então é cidade (ex.: SANTA ROSA).
    if any(chave in cidades for cidades in _municipios().values()):
        return False
    for campo in ("nome", "filiacao_1", "filiacao_2"):
        pessoa = _chave_cidade(d.get(campo) or "")
        if len(pessoa) < 5:
            continue
        if chave == pessoa or chave in pessoa or pessoa in chave:
            return True
        if difflib.SequenceMatcher(None, chave, pessoa).ratio() >= 0.8:
            return True
    return False


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

# ============================================================
# MUNICÍPIOS OFICIAIS (IBGE) — municipios.json, o mesmo usado pelo site
# ============================================================
_MUNICIPIOS = None


def _chave_cidade(nome):
    """Forma de comparação: sem acento, maiúscula, sem apóstrofo/hífen."""
    t = _tirar_acentos(str(nome or "")).upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _municipios():
    """{UF: {chave: NOME OFICIAL}} carregado uma vez."""
    global _MUNICIPIOS
    if _MUNICIPIOS is None:
        try:
            caminho = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "municipios.json"
            )
            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)
            _MUNICIPIOS = {
                uf: {_chave_cidade(n): n for n in lista}
                for uf, lista in dados.items() if uf in UFS_BR
            }
        except Exception:
            _MUNICIPIOS = {}
    return _MUNICIPIOS


def _municipio_oficial(nome, uf):
    """
    Confere a cidade lida com a lista oficial do estado.
    Devolve (nome_oficial, uf, certeza):
    - achou igual: nome oficial, certeza;
    - parecido (erro de leitura/abreviação): corrige, mas sem certeza;
    - sem UF: procura no Brasil todo; se a cidade só existe em um estado,
      usa esse estado, sem certeza;
    - não achou: mantém o lido, sem certeza.
    """
    base = _municipios()
    if not base or not nome:
        return nome, uf, True
    chave = _chave_cidade(nome)

    if uf in base:
        cidades = base[uf]
        if chave in cidades:
            return cidades[chave], uf, True
        import difflib
        notas = sorted(
            ((difflib.SequenceMatcher(None, chave, c).ratio(), c) for c in cidades),
            reverse=True,
        )
        if notas and notas[0][0] >= 0.85 and (
            len(notas) == 1 or notas[0][0] - notas[1][0] >= 0.03
        ):
            return cidades[notas[0][1]], uf, False
        return nome, uf, False

    achados = [(u, c[chave]) for u, c in base.items() if chave in c]
    if len(achados) == 1:
        return achados[0][1], achados[0][0], False
    return nome, uf, False


# Opções do campo "Órgão Emissor" do SICAQ para RG/CIN (texto exato da lista).
ORGAO_SSP = "SECRETARIA DE SEGURANÇA PÚBLICA"
ORGAO_PC = "POLÍCIA CIVIL"
ORGAO_SDS = "SECRETARIA DEFESA SOCIAL"
ORGAO_CASA_CIVIL = "SECRETARIA DE ESTADO DA CASA CIVIL"
ORGAO_OUTRO = "OUTRO TIPO DE ÓRGÃO EMISSOR"


def _orgao_sicaq(orgao_bruto):
    """
    Converte o órgão expedidor lido no RG/CIN para a opção exata do SICAQ.
    Devolve (opção, certeza). Sem certeza, o site mostra o campo em amarelo.
    """
    t = _tirar_acentos(str(orgao_bruto or "")).upper()
    t = re.sub(r"[^A-Z0-9 /\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Tira a UF do final: SSP/SP, SSP-SP, SSP SP, PCMG -> PC.
    sem_uf = re.sub(r"[\s/\-]+(" + "|".join(sorted(UFS_BR)) + r")$", "", t).strip()
    s = sem_uf.replace(" ", "")
    if not s:
        return "", False

    def tem(*termos):
        return any(x in t for x in termos)

    # Casa Civil
    if tem("CASA CIVIL"):
        return ORGAO_CASA_CIVIL, True
    if s in ("SCC", "SECC", "SECCIVIL", "SECASACIVIL"):
        return ORGAO_CASA_CIVIL, False
    # Defesa Social (PE: SDS; MG: SEDS). SSPDS (CE) tem as duas palavras.
    if s.startswith("SSPDS") or tem("SEGURANCA PUBLICA E DEFESA SOCIAL"):
        return ORGAO_SSP, False
    if s in ("SDS", "SEDS") or tem("DEFESA SOCIAL"):
        return ORGAO_SDS, True
    # Siglas combinadas, ex.: IIRGD.SSP.SP, II-SSP, SDS PE.
    siglas = set(re.split(r"[\s/\-]+", sem_uf))
    if siglas & {"SDS", "SEDS"}:
        return ORGAO_SDS, True
    if siglas & {"SSP", "SESP", "IIRGD", "IRGD", "SEJUSP", "SESEG", "SESDEC", "SSPTC"}:
        return ORGAO_SSP, True
    # Segurança Pública (inclui IIRGD/IRGD de SP e Instituto de Identificação).
    if (
        s.startswith(("SSP", "SESP", "SESEG", "SEJUSP", "SESDEC", "SSPTC"))
        or s in ("IIRGD", "IRGD", "II", "IIPR", "IISP")
        or tem("SEGURANCA PUBLICA", "INSTITUTO DE IDENTIFICACAO", "RICARDO GUMBLETON")
    ):
        return ORGAO_SSP, True
    # Polícia Civil (PC, PCMG, PC/GO, PCIVIL, DGPC...).
    if (
        re.fullmatch(r"PC([A-Z]{2})?", s)
        or s.startswith(("PCIVIL", "DGPC", "DPC"))
        or tem("POLICIA CIVIL")
    ):
        return ORGAO_PC, True
    if s in ("IFP", "DIC", "IPF"):
        return ORGAO_PC, False
    # Outros emissores (trânsito, perícia, forças armadas, conselhos, PF).
    if s.startswith("DETRAN") or tem("TRANSITO"):
        return ORGAO_OUTRO, True
    if s.startswith(("MEX", "EB", "MAER", "COMAER", "MMA", "MD", "DPF", "PF", "OAB", "CR", "CO", "MTE")) or tem(
        "EXERCITO", "MARINHA", "AERONAUTICA", "POLICIA FEDERAL", "CONSELHO", "ORDEM DOS ADVOGADOS"
    ):
        return ORGAO_OUTRO, True
    if s.startswith(("IGP", "ITEP", "IPC")):
        return ORGAO_OUTRO, False
    # Sigla desconhecida: a opção mais comum, marcada para conferir.
    return ORGAO_SSP, False


def _rg_sp_valido(rg):
    """
    Dígito do RG de São Paulo (IIRGD): 8 dígitos com pesos 2..9; DV = 11 - resto
    (10 => X, 11 => 0). Ex.: 58.569.906-9. Devolve None quando o formato não é
    o de 7 ou 8 dígitos + DV (aí não dá para conferir).
    """
    r = re.sub(r"[^0-9X]", "", str(rg or "").upper())
    if len(r) == 8 and r[:7].isdigit():
        r = "0" + r  # RG antigo com 7 dígitos + DV (ex.: 4.752.686-1)
    if len(r) != 9 or not r[:8].isdigit():
        return None
    soma = sum(int(c) * w for c, w in zip(r[:8], range(2, 10)))
    calc = 11 - soma % 11
    dv = "X" if calc == 10 else ("0" if calc == 11 else str(calc))
    return r[8] == dv


# Sobrenomes que aparecem ligados por "E" dentro de um mesmo nome
# (ex.: SOUZA E SILVA). Depois do "E" vindo um destes, o "E" é do nome.
_SOBRENOMES_COM_E = {
    "SILVA", "SOUZA", "SOUSA", "SANTOS", "OLIVEIRA", "COSTA", "LIMA", "PEREIRA",
    "ALMEIDA", "CASTRO", "MELO", "MELLO", "CUNHA", "SA", "SILVEIRA", "ARAUJO",
    "BARROS", "CAMPOS", "MENEZES", "VASCONCELOS", "ALBUQUERQUE", "AZEVEDO",
    "CARVALHO", "FARIA", "FREITAS", "MOURA", "PINTO", "RAMOS", "REIS", "ROCHA",
    "SOARES", "TEIXEIRA", "VIEIRA", "GOMES", "MARTINS", "RIBEIRO", "RODRIGUES",
    "FERREIRA", "ALVES", "BRITO", "MOTA", "MOTTA", "PAIVA", "PRADO", "QUEIROZ",
    "SAMPAIO", "VALE", "CORDEIRO", "NOGUEIRA", "MACHADO", "MENDES", "BARBOSA",
    "FONSECA", "ANDRADE", "MIRANDA", "MOREIRA", "NUNES", "LOPES", "CORREA",
    "CORREIA", "FRANCO", "GUIMARAES", "LEITE", "MAIA", "MEDEIROS", "MONTEIRO",
    "NASCIMENTO", "PACHECO", "PIRES", "TAVARES", "XAVIER", "SIQUEIRA", "BRAGA",
}


def _separar_filiacao(pai, mae):
    """
    RG com filiação corrida ("PAI E MÃE" quebrando linha no meio do nome da
    mãe): a IA às vezes devolve pai "EDILSON ... SILVA E ELIZIA" e mãe
    "FRANCISCA DE SOUZA SILVA". Move o que vem depois do " E " para a mãe.
    Devolve (pai, mae, mudou).
    """
    pai = re.sub(r"\s+", " ", str(pai or "")).strip()
    mae = re.sub(r"\s+", " ", str(mae or "")).strip()
    mudou = False
    # Linha que terminou no "E" ou mãe que começou com ele.
    if re.search(r"\sE$", pai):
        pai, mudou = pai[:-2].strip(), True
    if re.match(r"^E\s", mae):
        mae, mudou = mae[2:].strip(), True
    partes = re.split(r"\s+E\s+", pai)
    if len(partes) >= 2:
        antes = " E ".join(partes[:-1]).strip()
        depois = partes[-1].strip()
        primeira = _tirar_acentos(depois.split(" ")[0]).upper() if depois else ""
        if antes and depois and primeira not in _SOBRENOMES_COM_E and len(antes.split()) >= 2:
            pai = antes
            if not mae:
                mae = depois
            elif not _tirar_acentos(mae).upper().startswith(_tirar_acentos(depois).upper()):
                mae = f"{depois} {mae}"
            mudou = True
    return pai, mae, mudou


def _normalizar_saida(d):
    d = _corrigir_datas_cnh(d)
    d = dict(d or {})
    pai, mae, mudou = _separar_filiacao(d.get("filiacao_1"), d.get("filiacao_2"))
    if mudou:
        d["filiacao_1"], d["filiacao_2"] = pai, mae
        d["campos_incertos"] = list(d.get("campos_incertos") or []) + ["filiacao_1", "filiacao_2"]
    tipo = str(d.get("tipo_documento") or "").strip().upper()
    nacionalidade = str(d.get("nacionalidade") or "").strip().upper()
    # CIN imprime "BRA"; documentos bilíngues podem trazer BRAZIL/BRAZILIAN.
    if nacionalidade.startswith("BRASILEIR") or nacionalidade in (
        "BR", "BRA", "BRASIL", "BRAZIL", "BRAZILIAN"
    ):
        nacionalidade = "BRASILEIRA"

    nat_mun = _normalizar_municipio_naturalidade(
        d.get("naturalidade_municipio") or ""
    )
    nat_uf = _uf_naturalidade(
        d.get("naturalidade_uf"),
        d.get("naturalidade_municipio"),
    )
    naturalidade_descartada = False
    if nat_mun and _naturalidade_e_pessoa(nat_mun, d):
        # Ex.: CNH antiga sem naturalidade e a IA pôs o nome da mãe.
        print("[LER] naturalidade descartada: era nome de pessoa", flush=True)
        nat_mun, nat_uf, naturalidade_descartada = "", "", True
        d["campos_incertos"] = list(d.get("campos_incertos") or []) + ["naturalidade_municipio"]

    sexo = _tirar_acentos(str(d.get("sexo") or "").strip().upper())
    if sexo in ("M", "MASC", "MASCULINO"):
        sexo = "MASCULINO"
    elif sexo in ("F", "FEM", "FEMININO"):
        sexo = "FEMININO"

    # Exibição no site: CIDADE/UF (ex.: SOROCABA/SP).
    # O site separa pela barra e envia à extensão o município (sem UF)
    # e a UF em campos próprios, como o SICAQ exige.
    incertos = list(d.get("campos_incertos") or [])

    # Confere a cidade com a lista oficial do IBGE (a mesma da caixa do site).
    if nat_mun:
        uf_lida = nat_uf
        nat_mun, nat_uf, certeza = _municipio_oficial(nat_mun, nat_uf)
        nat_uf = nat_uf or ""
        if not certeza and "naturalidade_municipio" not in incertos:
            incertos.append("naturalidade_municipio")
        if nat_uf and not uf_lida and "naturalidade_uf" not in incertos:
            incertos.append("naturalidade_uf")  # UF deduzida pela cidade

    if nat_mun and nat_uf:
        naturalidade = f"{nat_mun}/{nat_uf}"
    else:
        naturalidade = nat_mun
        if nat_mun and "naturalidade_uf" not in incertos:
            incertos.append("naturalidade_uf")

    # CPF lido que não fecha os dígitos verificadores: provável erro de leitura.
    cpf_lido = re.sub(r"\D", "", str(d.get("cpf") or ""))
    if tipo in ("RG", "CIN") and cpf_lido and not _cpf_valido(cpf_lido) and "cpf" not in incertos:
        incertos.append("cpf")

    # RG de SP: confere o dígito (pega o número do espelho lido no lugar do RG).
    uf_doc = _tirar_acentos(str(d.get("uf_emissao") or "")).strip().upper()
    if tipo == "RG" and uf_doc == "SP" and _rg_sp_valido(d.get("rg")) is False and "rg" not in incertos:
        incertos.append("rg")

    # Órgão emissor do RG/CIN já no texto exato da opção do SICAQ.
    orgao_sicaq = ""
    if tipo in ("RG", "CIN"):
        orgao_sicaq, certeza = _orgao_sicaq(d.get("orgao_emissor"))
        if not certeza and "orgao_emissor" not in incertos:
            incertos.append("orgao_emissor")

    return {
        "documento": tipo,
        "nome": _tirar_acentos(str(d.get("nome") or "").strip().upper()),
        "cpf": "".join(c for c in str(d.get("cpf") or "") if c.isdigit()),
        "rg": "".join(c for c in str(d.get("rg") or "").upper() if c.isalnum()),
        "numero_cnh": "".join(c for c in str(d.get("numero_cnh") or "") if c.isdigit()),
        "orgao_emissor": _tirar_acentos(str(d.get("orgao_emissor") or "").strip().upper()),
        "orgao_sicaq": orgao_sicaq,
        "uf": _tirar_acentos(str(d.get("uf_emissao") or "").strip().upper()),
        "nascimento": _formatar_data_br(d.get("data_nascimento")),
        "primeira": _formatar_data_br(d.get("data_primeira_habilitacao")),
        "emissao": _formatar_data_br(d.get("data_emissao")),
        "validade": _formatar_data_br(d.get("validade")),
        "sexo": sexo,
        "nacionalidade": nacionalidade or "BRASILEIRA",
        "naturalidade": naturalidade,
        "naturalidade_uf": nat_uf,
        "naturalidade_descartada": naturalidade_descartada,
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
Leia este comprovante de renda brasileiro (HOLERITE / CONTRACHEQUE, recibo
do eSocial de empregado doméstico ou EXTRATO DE PAGAMENTO DE BENEFÍCIO DO
INSS) e devolva SOMENTE JSON válido.
A imagem pode ser FOTO (papel torto/dobrado): siga cada linha da tabela com
cuidado, sem pular para a linha de cima ou de baixo.

Chaves, NESTA ORDEM:
origem, competencia, empregador_tipo, empregador_documento, cpf_funcionario,
numero_beneficio, especie_beneficio, data_admissao, itens, total_proventos,
total_descontos, liquido_original, folha_estado_sp, campos_incertos.

Regras:
- origem: "INSS" se for extrato/demonstrativo de pagamento de benefício do
  INSS (aposentadoria, pensão por morte, auxílio etc.); "ESOCIAL" se for
  recibo de empregado DOMÉSTICO do eSocial (empregador pessoa física);
  "HOLERITE" nos demais casos.
- competencia: mês/ano do comprovante em MM/AAAA.
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
    for ln in linhas:
        ws = sorted(ln["ws"], key=lambda w: w[0])
        textos = [_tirar_acentos(str(w[4])).upper() for w in ws]
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


def _imagem_holerite(data, filename, mimetype):
    ext = PathLikeSuffix(filename)
    is_pdf = ext == ".pdf" or mimetype == "application/pdf"

    if not is_pdf:
        # Foto: corrige a rotação do celular e recorta o papel quando ele
        # está pequeno/torto na foto (mesma rotina dos documentos).
        img = _abrir_imagem(data)
        img, _ = _recortar_documento(img)
        max_lado = 1600
        if max(img.size) > max_lado:
            escala = max_lado / max(img.size)
            img = img.resize(
                (max(1, int(img.width * escala)), max(1, int(img.height * escala))),
                Image.Resampling.LANCZOS,
            )
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90, optimize=True)
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
            body = _groq_abrir(req)
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


def _normalizar_holerite(resultado):
    resultado = dict(resultado or {})
    fonte = _fonte_renda(resultado)
    competencia = _holerite_competencia_mm_aaaa(resultado.get("competencia"))
    validas = _holerite_competencias_validas()

    if competencia not in validas:
        identificada = competencia or "não identificada"
        raise ValueError(
            f"HOLERITE_FORA_PERIODO|Competência identificada: {identificada}. "
            f"Período aceito: {validas[0]}, {validas[1]} e {validas[2]}."
        )

    incertos = [str(x) for x in (resultado.get("campos_incertos") or [])]
    bruta = _holerite_numero(resultado.get("total_proventos") or resultado.get("bruta"))
    total_desc = _holerite_numero(resultado.get("total_descontos"))
    liquido_original = _holerite_numero(resultado.get("liquido_original"))
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
        "campos_incertos": sorted(set(incertos)),
    }


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
                         "irrf_anual": _valor_br(valores[2]) if len(valores) > 2 else 0.0}
                fontes.append(atual)
                continue
            if atual is not None and not valores and not any(
                    k in ln for k in ("NOME DA FONTE", "REND. RECEBIDOS", "DE PES. JURIDICA", "OFICIAL", "SALARIO")):
                atual["nome"] = (atual["nome"] + " " + str(bruto).strip()).strip()
        return {"exercicio": exercicio, "ano_calendario": ano_cal, "cpf_titular": cpf_titular,
                "fontes": fontes, "total": total}
    return None


def _exercicios_irpf_aceitos(hoje=None):
    """Exercício do ano atual; de janeiro a maio (prazo de entrega) também o anterior."""
    hoje = hoje or datetime.now(FUSO_BR).date()
    aceitos = [hoje.year]
    if hoje.month <= 5:
        aceitos.append(hoje.year - 1)
    return aceitos


def _normalizar_irpf(info, hoje=None):
    aceitos = _exercicios_irpf_aceitos(hoje)
    if info.get("exercicio") not in aceitos:
        ex = info.get("exercicio") or "não identificado"
        lista = " ou ".join(f"{a} (ano-calendário {a - 1})" for a in aceitos)
        raise ValueError(f"IRPF_FORA_PERIODO|Declaração do exercício {ex}. Agora vale a do exercício {lista}.")
    fontes = [f for f in info.get("fontes") or [] if f.get("anual", 0) > 0]
    if not fontes:
        raise ValueError(
            "A declaração não tem rendimentos tributáveis recebidos de pessoa jurídica pelo titular."
        )
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
    }


# ============================================================
# AVISO DE "PASTA" (PDF com vários documentos juntos)
# A leitura continua igual (só as primeiras páginas). Aqui só se olha o
# TEXTO de cada página, sem IA, para avisar quando o PDF parece uma pasta:
# - o documento pedido está numa página que a leitura não usa -> não gasta
#   IA e pede para mandar só o documento;
# - a parte lida mistura outros documentos -> lê e avisa para conferir.
# PDF escaneado (sem texto) não dá para classificar: só avisa se tiver
# mais páginas do que a leitura usa.
# ============================================================

_SINAIS_PAGINA = [
    ("renda", (r"\bVENCIMENTOS\b|\bPROVENTOS\b", r"\bDESCONTOS\b")),
    ("renda", (r"CONTRACHEQUE|CONTRA-CHEQUE|HOLERITE|HOLERITH|RECIBO DE PAGAMENTO|DEMONSTRATIVO DE PAGAMENTO",)),
    ("renda", (r"\bINSS\b|PREVIDENCIA SOCIAL", r"BENEFICIO", r"COMPETENCIA|CREDITO|VALOR LIQUIDO|MENSALIDADE")),
    ("identidade", (r"SENATRAN|CARTEIRA NACIONAL DE HABILITACAO|PERMISSAO PARA DIRIGIR",)),
    ("identidade", (r"REGISTRO GERAL|CARTEIRA DE IDENTIDADE|IDENTIDADE NACIONAL",)),
    ("certidao", (r"CERTIDAO DE (NASCIMENTO|CASAMENTO|OBITO)|REGISTRO CIVIL DAS PESSOAS NATURAIS",)),
    ("trabalho", (r"CARTEIRA DE TRABALHO|\bFGTS\b",)),
    ("conta", (r"NOTA FISCAL|FATURA|CONTA DE (ENERGIA|LUZ|AGUA|GAS)|CODIGO DE BARRAS",)),
]

_NOME_DOC_PASTA = {
    "renda": ("o holerite / comprovante de renda", "só o holerite (ou só a página dele)"),
    "identidade": ("o documento (CNH, RG ou CIN)", "só o documento (ou só a página dele)"),
    "certidao": ("a certidão", "só a certidão (ou só a página dela)"),
    "conta": ("o comprovante de endereço", "só o comprovante (ou só a página dele)"),
}


def _classe_pagina(texto):
    t = _tirar_acentos(str(texto or "")).upper()
    if len(t.strip()) < 40:
        return ""  # sem texto (escaneada/foto)
    for classe, padroes in _SINAIS_PAGINA:
        if all(re.search(p, t) for p in padroes):
            return classe
    return "outro"


_CLASSES_CONHECIDAS = {"renda", "identidade", "certidao", "trabalho", "conta"}


def _nas_paginas(nums):
    nums = sorted(nums)
    return f"na página {nums[0]}" if len(nums) == 1 else f"nas páginas {_faixa_paginas(nums)}"


def _faixa_paginas(nums):
    nums = sorted(nums)
    if not nums:
        return ""
    grupos, ini, ant = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == ant + 1:
            ant = n
            continue
        if ini == ant:
            grupos.append(f"{ini}")
        else:
            grupos.append(f"{ini} e {ant}" if ant == ini + 1 else f"{ini} a {ant}")
        if n is not None:
            ini = ant = n
    return ", ".join(grupos)


def _classes_pdf(item, maximo=60):
    """(nº de páginas, [classe de cada página]) de um PDF; None se não for PDF."""
    if PathLikeSuffix(item.get("filename", "")) != ".pdf" and item.get("mimetype") != "application/pdf":
        return None
    try:
        doc = pymupdf.open(stream=item["data"], filetype="pdf")
    except Exception:
        return None
    try:
        classes = []
        for i in range(min(len(doc), maximo)):
            try:
                classes.append(_classe_pagina(doc[i].get_text("text") or ""))
            except Exception:
                classes.append("")
        return len(doc), classes
    finally:
        doc.close()


def _avaliar_pasta(arquivos, quer, lidas=None):
    """
    quer: 'renda', 'identidade' ou 'certidao'. lidas: quantas páginas a
    leitura usa (None = holerite: 8 com texto, 3 escaneado).
    Devolve (bloqueio, aviso): bloqueio = mensagem para NÃO ler (o documento
    está fora das páginas lidas); aviso = ler e pedir conferência.
    """
    nome, mandar = _NOME_DOC_PASTA.get(quer, ("o documento", "só o documento"))
    bloqueio, aviso = "", ""
    for item in arquivos:
        info = _classes_pdf(item)
        if not info:
            continue
        n, classes = info
        usa = lidas if lidas else (8 if any(classes[:2]) else 3)
        alvo = [i + 1 for i, c in enumerate(classes) if c == quer]
        # Outros documentos reconhecidos (CNH, conta, CTPS...) nas páginas lidas.
        outros = [i + 1 for i, c in enumerate(classes) if c in _CLASSES_CONHECIDAS and c != quer]
        misturado = any(i <= usa for i in outros)
        if n <= usa and not misturado:
            continue  # PDF comum (frente/verso, vários meses de holerite...)
        so_usa = "a página 1" if usa == 1 else f"as {usa} primeiras páginas"
        nas_lidas = "na página 1" if usa == 1 else f"nas {usa} primeiras páginas"
        if alvo and not any(i <= usa for i in alvo):
            bloqueio = (
                f"Esse PDF parece uma pasta com {n} páginas: {nome} está {_nas_paginas(alvo)}, "
                f"mas a leitura só usa {so_usa}. Mande {mandar}."
            )
            break
        dentro = [i for i in alvo if i <= usa]
        if dentro:
            aviso = (f"Esse PDF tem {n} páginas e parece uma pasta. Li {nome} {_nas_paginas(dentro)}: "
                     f"confira os dados. Se algo vier errado, mande {mandar}.")
        elif misturado:
            aviso = (f"Esse PDF tem {n} páginas e parece uma pasta: não reconheci {nome} {nas_lidas}, "
                     f"que são as que eu leio. Confira o resultado; se vier errado, mande {mandar}.")
        else:
            aviso = (f"Esse PDF tem {n} páginas e eu leio só {so_usa}. "
                     f"Se {nome} estiver depois, mande {mandar}.")
    return bloqueio, aviso


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

    # Declaração de Imposto de Renda (PDF da Receita): lida pelo texto, sem IA.
    irpf = None
    try:
        irpf = _irpf_do_pdf(arquivos)
    except Exception as e:
        print(f"[IRPF] leitura falhou ({type(e).__name__})", flush=True)
    if irpf is not None:
        try:
            dados = _normalizar_irpf(irpf)
        except ValueError as e:
            txt = str(e)
            code = "IRPF_FORA_PERIODO" if txt.startswith("IRPF_FORA_PERIODO|") else ""
            return jsonify({"ok": False, "code": code, "error": txt.split("|", 1)[-1]}), 422
        _registrar_uso("renda")
        print(f"[IRPF] exercício {irpf['exercicio']}, {len(dados['fontes_ir'])} fonte(s) PJ", flush=True)
        return jsonify({
            "ok": True, "dados": dados, "modelo": "texto do PDF", "arquivos_lidos": len(arquivos),
            "tempo_api": 0, "tempo_total": round(time.perf_counter() - inicio, 3),
            "competencias_validas": _holerite_competencias_validas(),
        })

    # PDF tipo "pasta" com o holerite fora das páginas lidas: não gasta IA.
    bloqueio, aviso_pasta = _avaliar_pasta(arquivos, "renda")
    if bloqueio:
        print("[PASTA] holerite fora das páginas lidas", flush=True)
        return jsonify({"ok": False, "code": "PASTA", "error": bloqueio}), 422

    try:
        resultado, modelo, tempo_api = _groq_holerite_call(arquivos)
        # PDF com texto: confere o cabeçalho do Estado de SP sem depender da IA.
        if _holerite_estado_sp_pdf(arquivos):
            resultado = {**(resultado or {}), "folha_estado_sp": True}
        dados = _normalizar_holerite(resultado)
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


# ============================================================
# PIS / PASEP / NIS / NIT (extrato do FGTS, Carteira de Trabalho Digital,
# Cartão Cidadão ou qualquer documento com o número escrito)
# ============================================================

PROMPT_PIS = """
Encontre no documento o número do PIS/PASEP (também chamado NIS ou NIT).
Ele tem 11 dígitos e costuma aparecer como 123.45678.90-1, perto de
"PIS", "PASEP", "PIS/PASEP", "NIS" ou "NIT".
Não confunda com CPF, CTPS, título de eleitor, conta, matrícula ou CNPJ.
Responda SOMENTE JSON: {"pis": "somente dígitos ou vazio", "nome": "nome do titular, se aparecer"}.
""".strip()


def _pis_valido(pis):
    """Confere o dígito verificador do PIS/PASEP/NIS (pesos 3,2,9,8,7,6,5,4,3,2)."""
    p = re.sub(r"\D", "", str(pis or ""))
    if len(p) != 11 or len(set(p)) == 1:
        return False
    pesos = (3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    resto = 11 - sum(int(p[i]) * pesos[i] for i in range(10)) % 11
    dv = 0 if resto in (10, 11) else resto
    return dv == int(p[10])


def _pis_do_texto(texto):
    """
    Acha o PIS no texto de um PDF digital (ex.: extrato do FGTS) sem usar a IA.
    Só aceita número válido perto das palavras PIS/PASEP/NIS/NIT.
    """
    t = _tirar_acentos(str(texto or "")).upper()
    numero = r"(\d{3}[.\s]?\d{5}[.\s]?\d{2}[\-.\s]?\d)"
    achados = []
    for m in re.finditer(r"\b(PIS|PASEP|NIS|NIT)\b", t):
        trecho = t[m.end(): m.end() + 80]
        n = re.search(numero, trecho)
        if n:
            digitos = re.sub(r"\D", "", n.group(1))
            if _pis_valido(digitos) and digitos not in achados:
                achados.append(digitos)
    # Um único número válido: certeza. Vários diferentes: deixa para a IA.
    return achados[0] if len(achados) == 1 else ""


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


@app.post("/ler-pis")
def ler_pis():
    arquivo = request.files.get("file") or (request.files.getlist("files") or [None])[0]
    if not arquivo or not (arquivo.filename or "").strip():
        return jsonify({"ok": False, "error": "Arquivo do PIS não enviado."}), 400
    data = arquivo.read()
    if not data:
        return jsonify({"ok": False, "error": "Arquivo vazio."}), 400
    nome_arq = arquivo.filename or ""
    mime = arquivo.mimetype or ""
    inicio = time.perf_counter()

    try:
        # 1) PDF digital (ex.: extrato do FGTS): lê o texto, sem gastar IA.
        if PathLikeSuffix(nome_arq) == ".pdf" or mime == "application/pdf":
            texto = _pdf_texto_holerite(data)
            pis = _pis_do_texto(texto)
            if pis:
                print("[PIS] encontrado no texto do PDF (sem IA)", flush=True)
                _registrar_uso("pis")
                return jsonify({
                    "ok": True, "pis": pis, "valido": True, "nome": "",
                    "fonte": "texto", "tempo_total": round(time.perf_counter() - inicio, 3),
                })

            # PDF digital tipo "pasta": o PIS pode estar depois das 8
            # primeiras páginas, e a IA só veria as páginas 1 e 2.
            info = _classes_pdf({"data": data, "filename": nome_arq, "mimetype": mime})
            if info and info[0] > 8 and any(info[1][:2]):
                print("[PASTA] PIS não achado nas 8 primeiras páginas", flush=True)
                return jsonify({
                    "ok": False, "code": "PASTA",
                    "error": (f"Não achei o PIS nas 8 primeiras páginas desse PDF ({info[0]} páginas, "
                              "parece uma pasta). Mande só o extrato do FGTS ou a Carteira de Trabalho, "
                              "ou digite o número."),
                }), 422

        # 2) Foto/print ou PDF escaneado: leitura curta da IA.
        imagem = _combinar_arquivos_identificacao(
            [{"data": data, "filename": nome_arq, "mimetype": mime}]
        )
        resultado, tempo_api = _groq_json_imagem(PROMPT_PIS, imagem)
        pis = re.sub(r"\D", "", str(resultado.get("pis") or ""))
        nome = _tirar_acentos(str(resultado.get("nome") or "").strip().upper())
        print(f"[PIS] lido pela IA; valido={_pis_valido(pis)}", flush=True)
        _registrar_uso("pis")
        return jsonify({
            "ok": True, "pis": pis, "valido": _pis_valido(pis), "nome": nome,
            "fonte": "ia", "tempo_api": round(tempo_api, 3),
            "tempo_total": round(time.perf_counter() - inicio, 3),
        })
    except GroqLimite as e:
        return jsonify({"ok": False, "error": str(e)}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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


def _sim(valor):
    if isinstance(valor, bool):
        return valor
    return _tirar_acentos(str(valor or "")).strip().upper() in {
        "TRUE", "SIM", "S", "1", "YES", "VERDADEIRO",
    }


def _texto_limpo(valor):
    return re.sub(r"\s+", " ", _tirar_acentos(str(valor or "")).upper()).strip()


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
- numero: só o número do imóvel (ou "SN" quando não tiver número).
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
        if not numero or numero in ("SN", "S N"):
            numero = re.sub(r"\D", "", m.group(1))
        rua = rua[:m.start()]
    rua = limpo(rua)
    if re.fullmatch(r"S ?N[O]?|SEM NUMERO|SEM N", numero):
        numero = "SN"
    uf = limpo(r.get("uf"))[:2]
    cidade = limpo(r.get("cidade"))
    incertos = [str(x) for x in (r.get("campos_incertos") or [])]
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


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Assistente SICAQ Web",
        "groq_configurada": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "login": _login_ligado(),
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
        # PDF "CNH Digital" do gov.br: caminho próprio (campos + MRZ sem IA).
        cnh_digital = _cnh_digital_senatran(arquivos)
        bloqueio, aviso_pasta = _avaliar_pasta(arquivos, "identidade", 1 if cnh_digital else 2)
        if bloqueio:
            print("[PASTA] documento fora das páginas lidas", flush=True)
            return jsonify({"ok": False, "code": "PASTA", "error": bloqueio}), 422
        if cnh_digital:
            image_bytes, nota_ia = [cnh_digital["imagem"]], cnh_digital["nota"]
        else:
            image_bytes, nota_ia = _imagens_para_ia(arquivos)
        tempo_preparo = time.perf_counter() - t_preparo

        dados_raw, modelo, tempo_api = _groq_call(image_bytes, nota_ia)

        aviso_documento = ""
        if cnh_digital:
            dados_raw["tipo_documento"] = "CNH"
            if cnh_digital.get("sem_naturalidade"):
                # Modelo antigo: não existe naturalidade impressa.
                dados_raw["naturalidade_municipio"] = ""
                dados_raw["naturalidade_uf"] = ""
                dados_raw["campos_incertos"] = list(dados_raw.get("campos_incertos") or []) + ["naturalidade_municipio"]
                aviso_documento = ("CNH modelo antigo: ela não informa a naturalidade (local de nascimento). "
                                   "Preencha à mão, pelo RG ou certidão do cliente.")
            if cnh_digital["mrz"]:
                # MRZ lida sem IA vem primeiro; a da IA fica de reserva.
                dados_raw["mrz"] = cnh_digital["mrz"] + "\n" + str(dados_raw.get("mrz") or "")

        # Tipo do documento sempre como CNH, RG ou CIN (ou vazio se não der).
        tipo_lido = str(dados_raw.get("tipo_documento") or "")
        tipo, deduzido = _tipo_documento(dados_raw)
        dados_raw["tipo_documento"] = tipo
        if deduzido or not tipo:
            dados_raw["campos_incertos"] = list(dados_raw.get("campos_incertos") or []) + ["tipo_documento"]
        # Só o rótulo do tipo vai para o log (nenhum dado pessoal).
        print(f"[LER] tipo lido={tipo_lido[:40]!r} -> {tipo or 'NAO IDENTIFICADO'}", flush=True)

        # RG/CIN: validade igual à emissão é cópia errada (RG antigo nem tem).
        if tipo in ("RG", "CIN"):
            emissao_d = _parse_data_br(dados_raw.get("data_emissao"))
            if emissao_d and _parse_data_br(dados_raw.get("validade")) == emissao_d:
                dados_raw["validade"] = ""
                if tipo == "CIN":
                    dados_raw["campos_incertos"] = list(dados_raw.get("campos_incertos") or []) + ["validade"]

        # Confere nascimento/validade pela MRZ (rodapé '<<<'), quando houver.
        dados_raw = _aplicar_mrz(dados_raw)

        # CNH: uma única chamada ao Groq para não estourar o limite gratuito.
        # A conferência final é feita localmente, sem consumir tokens.
        if str(dados_raw.get("tipo_documento") or "").strip().upper() == "CNH":
            dados_raw = _validar_cnh_final(dados_raw)

        dados = _normalizar_saida(dados_raw)
        _registrar_uso("documento")

        if not aviso_documento and dados.get("naturalidade_descartada"):
            aviso_documento = ("A naturalidade lida não era uma cidade e foi apagada. "
                               "Preencha à mão, pelo documento do cliente.")
        dados.pop("naturalidade_descartada", None)
        return jsonify({
            "ok": True,
            "aviso_pasta": aviso_pasta,
            "aviso_documento": aviso_documento,
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
