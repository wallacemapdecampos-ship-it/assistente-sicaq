// Monta a pasta "site/" que o Render publica: só os arquivos da tela (nada de .py),
// com o JavaScript embaralhado (nomes trocados, comentários removidos) e o HTML enxuto.
// Uso: node build_site.mjs arquivo1 arquivo2 ...   (o index.html é sempre incluído)
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { transformSync } from "esbuild";

const SAIDA = "site";
rmSync(SAIDA, { recursive: true, force: true });
mkdirSync(SAIDA);

let html = readFileSync("index.html", "utf8");
let scripts = 0;
html = html.replace(/<script>([\s\S]*?)<\/script>/g, (_, js) => {
  scripts++;
  // Tudo dentro de uma função: os nomes deixam de ser globais e podem ser trocados.
  const r = transformSync(`(()=>{${js}\n})();`, {
    minify: true, target: "es2020", legalComments: "none", charset: "utf8",
  });
  return "<script>" + r.code.replace(/<\/script/gi, "<\\/script") + "</script>";
});
html = html.replace(/<style>([\s\S]*?)<\/style>/g, (_, css) =>
  "<style>" + transformSync(css, { loader: "css", minify: true, legalComments: "none", charset: "utf8" }).code + "</style>");
html = html.replace(/<!--(?!\[if)[\s\S]*?-->/g, "");      // comentários do HTML
writeFileSync(`${SAIDA}/index.html`, html);

for (const f of process.argv.slice(2)) copyFileSync(f, `${SAIDA}/${f}`);
console.log(`site/ pronto: index.html (${scripts} script(s) embaralhado(s)) + ${process.argv.length - 2} arquivo(s)`);
