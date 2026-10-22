# assistente-sicaq

Site (`index.html`, GitHub Pages) + servidor (Flask no Render, `gunicorn app:app`).

## Onde fica cada coisa no servidor

Cada documento tem o seu arquivo: mexer num não encosta nos outros.

| Arquivo | O que faz |
|---|---|
| `app.py` | Só junta os módulos, a rota `/health` e a "leitura repetida" (cache de 30 min). |
| `comum.py` | Flask, IA (Groq + reserva Cloudflare), limites e funções usadas por vários leitores. |
| `contas.py` | Login, senhas, empresas, donos/ADM, bloqueios e o painel de uso/custos. |
| `imagem.py` | Preparo das imagens para a IA (recorte de foto, PDF, rotação, páginas). |
| `pasta.py` | Aviso de PDF "pasta" (vários documentos juntos). |
| `leitor_cnh.py` | CNH: CNH Digital do gov.br, CNH antiga, MRZ sem IA e conferências da CNH. |
| `leitor_identidade.py` | RG, CIN e a rota `/ler` (CNH/RG/CIN). |
| `leitor_renda.py` | Holerite, eSocial, INSS, Estado de SP e a rota `/ler-holerite`. |
| `leitor_irpf.py` | Declaração de Imposto de Renda (texto do PDF ou imagem). |
| `leitor_pis.py` | PIS/PASEP/NIS e a rota `/ler-pis`. |
| `leitor_certidao.py` | Certidão -> estado civil e a rota `/ler-certidao`. |
| `leitor_endereco.py` | Comprovante de endereço, CEP e as rotas `/ler-endereco` e `/conferir-cep`. |

Regra de importação: um módulo só importa dos que vêm antes dele nesta ordem:
`comum → contas → imagem → pasta → leitor_cnh → leitor_identidade → leitor_irpf →
leitor_renda → leitor_pis → leitor_certidao → leitor_endereco → app`.
