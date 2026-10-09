"""
pipeline/model/gerar_relatorio.py
===================================
Gera o relatório de acurácia (CSV + resumo em Markdown) a partir do
cache por fold do confusion_matrix.py — sem retreinar nada. O PNG da
matriz já sai do próprio confusion_matrix.py.

Saídas (em --out-dir):
    acuracia_por_sinalizador.csv   acurácia de cada fold (sinalizador de teste)
    acuracia_por_classe.csv        recall por sinal + com quem mais é confundido
    matriz_confusao.csv            matriz completa (linhas = real, colunas = previsto)
    top_confusoes.csv              pares real -> previsto, em ordem decrescente
    resumo.md                      texto curto com os números principais

Uso (PowerShell, na raiz do repo, tudo numa linha):
    python -m pipeline.model.gerar_relatorio --cache-dir confusion_cache_e100s2 --annotations dataset/minds-libras/annotations.csv --out-dir relatorio_e100s2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Relatório de acurácia a partir do cache por fold.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--annotations", required=True, help="annotations.csv (de onde saem os nomes das classes)")
    parser.add_argument("--out-dir", default="relatorio")
    parser.add_argument("--title", default=None, help="Título do resumo (padrão: nome da pasta do cache)")
    args = parser.parse_args()

    folds = []
    for p in sorted(Path(args.cache_dir).glob("fold_*.json")):
        with open(p, encoding="utf-8") as f:
            folds.append(json.load(f))
    if not folds:
        raise SystemExit(f"Nenhum fold_*.json em {args.cache_dir}")

    params = folds[0].get("params", {})
    if any(f.get("params") != params for f in folds):
        print("[AVISO] os folds do cache têm hiperparâmetros diferentes entre si; o relatório mistura rodadas.")

    ann = pd.read_csv(args.annotations)
    classes = sorted(ann["class"].astype(str).unique())  # mesma ordem de build_label_mapping
    k = len(classes)
    if params.get("n_classes") not in (None, k):
        raise SystemExit(f"O cache tem {params['n_classes']} classes, mas o CSV tem {k}. Confira o --annotations.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((k, k), dtype=int)
    signer_rows = []
    for f in folds:
        t, p = np.array(f["y_true"]), np.array(f["y_pred"])
        np.add.at(cm, (t, p), 1)
        signer_rows.append(
            {"sinalizador": f["signer"], "amostras": len(t), "acertos": int((t == p).sum()),
             "acuracia": round(float((t == p).mean()), 4)}
        )
    per_signer = pd.DataFrame(signer_rows)
    per_signer.to_csv(out / "acuracia_por_sinalizador.csv", index=False, encoding="utf-8-sig")

    total, correct = int(cm.sum()), int(np.trace(cm))
    overall = correct / total
    chance = 1 / k

    class_rows = []
    for i, c in enumerate(classes):
        row = cm[i].copy()
        n_i = int(row.sum())
        row[i] = 0
        worst = int(row.argmax()) if row.sum() else None
        class_rows.append(
            {"classe": c, "amostras": n_i, "acertos": int(cm[i, i]),
             "acuracia": round(cm[i, i] / n_i, 4) if n_i else 0.0,
             "mais_confundida_com": classes[worst] if worst is not None else "",
             "vezes": int(row[worst]) if worst is not None else 0}
        )
    per_class = pd.DataFrame(class_rows).sort_values("acuracia", ascending=False)
    per_class.to_csv(out / "acuracia_por_classe.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(cm, index=classes, columns=classes).rename_axis("real \\ previsto").to_csv(
        out / "matriz_confusao.csv", encoding="utf-8-sig"
    )

    pairs = [
        {"real": classes[i], "previsto": classes[j], "contagem": int(cm[i, j]),
         "pct_da_classe_real": round(100 * cm[i, j] / max(1, cm[i].sum()), 1)}
        for i in range(k) for j in range(k) if i != j and cm[i, j] > 0
    ]
    top = pd.DataFrame(pairs).sort_values("contagem", ascending=False)
    top.to_csv(out / "top_confusoes.csv", index=False, encoding="utf-8-sig")

    best, worst_s = per_signer.loc[per_signer["acuracia"].idxmax()], per_signer.loc[per_signer["acuracia"].idxmin()]
    cfg = ", ".join(f"{k_}={v}" for k_, v in params.items())
    lines = [
        f"# Relatório de acurácia — {args.title or Path(args.cache_dir).name}",
        "",
        f"Validação leave-one-signer-out ({len(folds)} sinalizadores, {total} amostras de teste, {k} classes).",
        f"Configuração: {cfg}",
        "",
        f"- **Acurácia agregada: {100 * overall:.1f}%** (acaso: {100 * chance:.1f}%, {overall / chance:.1f}x acima do acaso)",
        f"- Média por sinalizador: {100 * per_signer['acuracia'].mean():.1f}% "
        f"(melhor: {best['sinalizador']} {100 * best['acuracia']:.0f}%; pior: {worst_s['sinalizador']} {100 * worst_s['acuracia']:.0f}%)",
        "",
        "## Classes mais bem reconhecidas",
        *[f"- {r.classe}: {100 * r.acuracia:.0f}%" for r in per_class.head(5).itertuples()],
        "",
        "## Classes menos reconhecidas",
        *[f"- {r.classe}: {100 * r.acuracia:.0f}% (mais confundida com {r.mais_confundida_com})" for r in per_class.tail(5).iloc[::-1].itertuples()],
        "",
        "## Principais confusões (real -> previsto)",
        *[f"- {r.real} -> {r.previsto}: {r.contagem}x ({r.pct_da_classe_real}% da classe)" for r in top.head(5).itertuples()],
        "",
    ]
    (out / "resumo.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Arquivos salvos em {out.resolve()}")


if __name__ == "__main__":
    main()