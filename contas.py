"""Login, senhas, empresas, donos/ADM, bloqueios e o painel de uso/custos."""

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
    _ip,
    _origin_ok,
    _rate_ok,
    _sim,
    _tirar_acentos,
    app,
)



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
DONOS_SISTEMA = ("wallace", "camila", "gustavo", "angelica")



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
                "CREATE TABLE IF NOT EXISTS uso_tokens ("
                " dia DATE NOT NULL, usuario TEXT NOT NULL, tipo TEXT NOT NULL,"
                " leituras INTEGER NOT NULL DEFAULT 0, chamadas INTEGER NOT NULL DEFAULT 0,"
                " tokens_entrada BIGINT NOT NULL DEFAULT 0, tokens_saida BIGINT NOT NULL DEFAULT 0,"
                " PRIMARY KEY (dia, usuario, tipo))"
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



# ---------- Custo simulado da IA (como se fosse a API paga) ----------
# A cada leitura grava os tokens usados (tabela uso_tokens). O custo do mês
# = leituras de cada tipo x média de tokens medida daquele tipo x preço.
_TIPO_DA_ROTA = {"/ler": "documento", "/ler-holerite": "renda", "/ler-pis": "pis",
                 "/ler-certidao": "certidao", "/ler-endereco": "endereco"}



PRECO_SIMULADO = {"modelo": "Claude Haiku 5.5", "entrada_usd": 0.10, "saida_usd": 0.50}



PRECO_COMPARACAO = {"modelo": "Claude Sonnet 5.5", "entrada_usd": 2.00, "saida_usd": 10.00}



# Até ter medições, usa uma estimativa por tipo (tokens por leitura).
TOKENS_ESTIMADOS = {"documento": (4500, 450), "renda": (5000, 650), "pis": (2500, 60),
                    "certidao": (3500, 250), "endereco": (3500, 250)}



def _cotacao_dolar():
    try:
        return float(os.getenv("COTACAO_DOLAR", "").replace(",", ".") or 5.5)
    except ValueError:
        return 5.5



def _gravar_tokens_no_banco(dia, usuario, tipo, chamadas, entrada, saida):
    try:
        _no_banco(lambda con: con.run(
            "INSERT INTO uso_tokens (dia, usuario, tipo, leituras, chamadas, tokens_entrada, tokens_saida)"
            " VALUES (CAST(:dia AS DATE), :usuario, :tipo, 1, :chamadas, :entrada, :saida)"
            " ON CONFLICT (dia, usuario, tipo) DO UPDATE SET leituras = uso_tokens.leituras + 1,"
            " chamadas = uso_tokens.chamadas + EXCLUDED.chamadas,"
            " tokens_entrada = uso_tokens.tokens_entrada + EXCLUDED.tokens_entrada,"
            " tokens_saida = uso_tokens.tokens_saida + EXCLUDED.tokens_saida",
            dia=dia, usuario=usuario, tipo=tipo, chamadas=chamadas, entrada=entrada, saida=saida))
    except Exception as e:
        print(f"[CUSTO] tokens não gravados ({type(e).__name__})", flush=True)



@app.after_request
def _gravar_tokens(resp):
    try:
        chamadas = getattr(g, "chamadas_ia", 0)
        tipo = _TIPO_DA_ROTA.get(request.path)
        if chamadas and tipo and os.getenv("DATABASE_URL", "").strip():
            import threading
            dia = datetime.now(FUSO_BR).strftime("%Y-%m-%d")
            usuario = getattr(g, "usuario", "") or "sem-login"
            threading.Thread(target=_gravar_tokens_no_banco, daemon=True, args=(
                dia, usuario, tipo, chamadas, getattr(g, "tokens_entrada", 0), getattr(g, "tokens_saida", 0))).start()
    except Exception:
        pass
    return resp



def _medias_tokens():
    """{tipo: {entrada, saida, medidas}}: média real por leitura (últimos 90 dias)."""
    medias = {t: {"entrada": e, "saida": s, "medidas": 0} for t, (e, s) in TOKENS_ESTIMADOS.items()}
    if not os.getenv("DATABASE_URL", "").strip():
        return medias
    try:
        linhas = _no_banco(lambda con: con.run(
            "SELECT tipo, SUM(leituras), SUM(tokens_entrada), SUM(tokens_saida) FROM uso_tokens"
            " WHERE dia >= CURRENT_DATE - 90 GROUP BY tipo")) or []
        for tipo, n, ent, sai in linhas:
            n = int(n or 0)
            if n > 0:
                medias[str(tipo)] = {"entrada": round(int(ent or 0) / n), "saida": round(int(sai or 0) / n), "medidas": n}
    except Exception as e:
        print(f"[CUSTO] médias indisponíveis ({type(e).__name__})", flush=True)
    return medias



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
    medias = _medias_tokens()
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
                            "donos": list(DONOS_SISTEMA),
                            "custo": {"medias": medias, "preco": PRECO_SIMULADO, "comparacao": PRECO_COMPARACAO,
                                      "cotacao": _cotacao_dolar()}})
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
                            "donos": list(DONOS_SISTEMA),
                            "custo": {"medias": medias, "preco": PRECO_SIMULADO, "comparacao": PRECO_COMPARACAO,
                                      "cotacao": _cotacao_dolar()}})
