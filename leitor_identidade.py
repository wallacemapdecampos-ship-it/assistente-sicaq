"""RG, CIN e a rota /ler (documento de identificação: CNH/RG/CIN)."""

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
    _parse_data_br,
    _tirar_acentos,
    app,
)

from contas import (  # noqa: F401
    _registrar_uso,
)

from imagem import (  # noqa: F401
    NOTA_ROTACOES,
    _imagens_para_ia,
)

from pasta import (  # noqa: F401
    _avaliar_pasta,
)

from leitor_cnh import (  # noqa: F401
    _aplicar_mrz,
    _cnh_digital_senatran,
    _corrigir_datas_cnh,
    _formatar_data_br,
    _validar_cnh_final,
)



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
        # Diagnóstico sem dado pessoal: só os NOMES dos campos que vieram vazios.
        _chaves = ("nome", "cpf", "rg", "data_nascimento", "numero_cnh", "data_emissao", "validade",
                   "filiacao_1", "filiacao_2", "naturalidade_municipio", "mrz")
        _vazios = [k for k in _chaves if not str(dados_raw.get(k) or "").strip()]
        print(f"[LER] modelo={modelo} vazios={','.join(_vazios) or 'nenhum'}; imagens={len(image_bytes)}", flush=True)

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
