r"""Figuras de publicación del protocolo v7 (español; títulos, ejes y leyendas completos). Todas usan sólo hogares de T."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metricas as MT

AZUL, ROJO, VERDE, GRIS, MORADO, OCRE = "#2563A6", "#C2410C", "#2E7D5B", "#6B7280", "#6D28D9", "#B7791F"


def _plt():
    import matplotlib.pyplot as plt
    from .. import viz
    viz.estilo()
    return plt


def _guardar(fig, ruta, mostrar=True):
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    if mostrar:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return ruta


def umbral_y_pr(y, s, umbral, titulo, ruta=None, mostrar=True):
    """(a) F1, precisión y exhaustividad según el umbral, con la meseta (F1 ≥ máx − 0,001) y F̂/2; (b) curva PR con
    curvas de iso-F1 y el punto de operación."""
    plt = _plt()
    thr, tp, fp, P, f = MT.curva_f1(y, s)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    i = int(np.argmax(f)); me = MT.meseta(y, s)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    k = np.linspace(0, len(thr) - 1, min(3000, len(thr))).astype(int)
    ax[0].plot(thr[k], f[k], color=ROJO, lw=2, label="F1")
    ax[0].plot(thr[k], prec[k], color=AZUL, lw=1.2, label="precisión")
    ax[0].plot(thr[k], rec[k], color=VERDE, lw=1.2, label="exhaustividad")
    lo, hi = me["meseta_0.001"]
    ax[0].axvspan(lo, hi, color=ROJO, alpha=.12, label=f"meseta F1 ≥ máx − 0,001: [{lo:.3f}; {hi:.3f}]")
    ax[0].axvline(umbral, color="k", ls=":", lw=1.2, label=f"umbral elegido = {umbral:.3f}")
    ax[0].axvline(f[i] / 2, color=MORADO, ls="--", lw=1, label=f"F̂/2 = {f[i] / 2:.3f} (Lipton et al., 2014)")
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="umbral sobre la probabilidad predicha", ylabel="valor",
              title="F1, precisión y exhaustividad según el umbral (OOF, sin Bogotá)")
    ax[0].legend(fontsize=8, loc="lower left")
    for fi in (0.6, 0.7, 0.8):
        r_ = np.linspace(fi / (2 - fi) + 1e-3, 1, 200); p_ = fi * r_ / (2 * r_ - fi)
        ax[1].plot(r_, p_, color=GRIS, lw=.7, ls=":"); ax[1].text(r_[-1], p_[-1], f"F1 = {fi}", fontsize=7, color=GRIS, va="bottom", ha="right")
    ax[1].plot(rec[k], prec[k], color=MORADO, lw=2, label=f"AUC-PR = {MT.auc_pr(y, s):.3f}")
    yh = s >= umbral; tp_, fp_, fn_, _ = MT.confusion(y, yh)
    ax[1].scatter([tp_ / (tp_ + fn_)], [tp_ / max(tp_ + fp_, 1)], color="k", zorder=5, s=50, label="punto de operación")
    ax[1].axhline(np.mean(y), color=GRIS, ls="--", lw=1, label=f"azar (tasa = {np.mean(y):.3f})")
    ax[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="exhaustividad (1 − subcobertura)", ylabel="precisión (1 − filtración)",
              title="Curva precisión–exhaustividad con curvas de iso-F1")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)


def calibracion(y, p, titulo, ruta=None, bins=20, mostrar=True):
    plt = _plt()
    t = MT.tabla_confiabilidad(y, p, bins)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1, label="calibración perfecta")
    ax[0].errorbar(t.p_media, t.frecuencia, yerr=1.96 * t.ee, fmt="o-", color=AZUL, ms=4, capsize=2,
                   label=f"{bins} bins de igual masa (IC 95 %)")
    ax[0].set(xlabel="probabilidad predicha media del bin", ylabel="frecuencia observada de pobreza",
              title=f"Diagrama de confiabilidad (ACE = {MT.ace(y, p):.4f}; Brier = {MT.brier(y, p):.4f})")
    ax[0].legend(fontsize=8)
    b = np.linspace(0, 1, 41)
    ax[1].hist(np.asarray(p)[np.asarray(y) == 0], bins=b, color=AZUL, alpha=.6, density=True, label="no pobres")
    ax[1].hist(np.asarray(p)[np.asarray(y) == 1], bins=b, color=ROJO, alpha=.6, density=True, label="pobres")
    ax[1].set(xlabel="probabilidad predicha", ylabel="densidad", title="Distribución de las probabilidades por clase")
    ax[1].legend(fontsize=8)
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)


def parsimonia(tabla: pd.DataFrame, titulo, ruta=None, tipo="logloss", mostrar=True):
    """Curva de parsimonia: F1 anidado y pérdida frente al número de preguntas y de variables (± EE de influencia), con
    la banda de la ROPE del mejor y el conjunto parsimonioso marcado."""
    plt = _plt()
    t = tabla.sort_values("n_preguntas")
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    mejorF = t.F1_anidado.max()
    for a, xcol, xl in ((ax[0], "n_preguntas", "preguntas de la encuesta c(S) (media de los pliegues)"),
                        (ax[1], "n_vars", "número de variables (media de los pliegues)")):
        a.errorbar(t[xcol], t.F1_anidado, yerr=t.EE_psi, fmt="o-", color=AZUL, capsize=3, ms=5, label="F1 anidado (± EE ψ)")
        a.axhspan(mejorF - MT.ROPE_F1, mejorF, color=VERDE, alpha=.15, label="ROPE del mejor (0,002)")
        p = t[t.parsimonioso]
        a.scatter(p[xcol], p.F1_anidado, s=220, facecolors="none", edgecolors="k", lw=1.8, zorder=6, label="parsimonioso")
        for _, r in t.iterrows():
            a.annotate(str(r.punto)[:14], (r[xcol], r.F1_anidado), fontsize=7, xytext=(4, 4), textcoords="offset points", color=GRIS)
        a.set(xscale="log" if t[xcol].max() / max(t[xcol].min(), 1) > 20 else "linear", xlabel=xl, ylabel="F1 anidado (T)",
              title=f"F1 frente a {'preguntas' if xcol == 'n_preguntas' else 'variables'}")
        a.legend(fontsize=8, loc="lower right")
    ax[2].plot(t.n_preguntas, t.perdida, "s-", color=ROJO)
    ax[2].axhspan(t.perdida.min(), t.perdida.min() + MT.rope_de(tipo), color=VERDE, alpha=.15, label="ROPE de la pérdida")
    ax[2].set(xlabel="preguntas de la encuesta c(S)", ylabel=f"{'log-loss' if tipo == 'logloss' else 'Brier'} (T, OOF)",
              title="Pérdida propia frente a preguntas"); ax[2].legend(fontsize=8)
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)


def importancia_preguntas(imp: pd.DataFrame, MP, titulo, ruta=None, top=25, mostrar=True,
                          xlabel="aumento de la log-loss al permutar juntas las variables que dependen de la pregunta (± EE entre pliegues)"):
    """imp: columnas `pregunta`, `media`, `ee` (dispersión entre pliegues/semillas)."""
    plt = _plt()
    d = imp.sort_values("media", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(d) + 1)))
    ax.barh([MP.etiqueta(q)[:55] for q in d.pregunta], d.media, xerr=d.get("ee"), color=AZUL, alpha=.9, error_kw={"capsize": 2})
    ax.set(xlabel=xlabel, title=titulo)
    ax.tick_params(axis="y", labelsize=8)
    return _guardar(fig, ruta, mostrar)


def dependencia_parcial(motor, aj, X, variables, titulo, ruta=None, n=2000, n_grid=25, semilla=0, mostrar=True):
    """Dependencia parcial (Friedman, 2001) de las variables principales: media de la predicción al fijar la variable
    en una rejilla de cuantiles, sobre una muestra de hogares de T; en gris, la distribución de la variable."""
    plt = _plt()
    rng = np.random.default_rng(semilla)
    Xs = X.iloc[rng.choice(len(X), min(n, len(X)), replace=False)]
    vs = [v for v in variables if v in X.columns and not isinstance(X[v].dtype, pd.CategoricalDtype)][:6]
    if not vs:
        return None
    fig, axs = plt.subplots(2, 3, figsize=(17, 9)); axs = axs.ravel()
    for a, v in zip(axs, vs):
        xv = Xs[v].to_numpy(float); xv = xv[np.isfinite(xv)]
        if len(xv) == 0:
            continue
        lo, hi = np.quantile(xv, [.02, .98])
        grid = np.unique(np.quantile(xv, np.linspace(.02, .98, n_grid)))
        pd_ = []
        for g in grid:
            Z = Xs.copy(); Z[v] = np.float32(g); pd_.append(float(np.mean(motor.predecir(aj, Z))))
        a.plot(grid, pd_, "-o", color=ROJO, ms=3)
        a2 = a.twinx(); a2.hist(xv[(xv >= lo) & (xv <= hi)], bins=40, color=GRIS, alpha=.25); a2.set_yticks([]); a2.grid(False)
        if hi > lo:
            a.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
        a.set(title=v, xlabel=f"valor de {v}", ylabel="P(pobre) media (dependencia parcial)")
    for a in axs[len(vs):]:
        a.axis("off")
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)


def disyuntiva(y, s, titulo, ruta=None, mostrar=True):
    """Filtración (1 − precisión) frente a subcobertura (1 − exhaustividad) con los umbrales F1, F2, costo explícito
    c_FN = 2·c_FP y tasa predicha = prevalencia (memo §5.6)."""
    plt = _plt()
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    thr, tp, fp, P, f = MT.curva_f1(y, s)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    f2 = 5 * tp / np.maximum(5 * tp + 4 * (P - tp) + fp, 1)
    puntos = {"F1-óptimo": int(np.argmax(f)), "F2-óptimo (subcobertura pesa más)": int(np.argmax(f2)),
              "costo c_FN = 2·c_FP (umbral 1/3)": int(np.argmin(np.abs(thr - 1 / 3))),
              "tasa predicha = prevalencia": int(np.argmin(np.abs((tp + fp) / len(y) - y.mean())))}
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    k = np.linspace(0, len(thr) - 1, min(3000, len(thr))).astype(int)
    ax.plot(1 - rec[k], 1 - prec[k], color=GRIS)
    filas = []
    for (nom, i), c in zip(puntos.items(), [ROJO, AZUL, VERDE, MORADO]):
        ax.scatter(1 - rec[i], 1 - prec[i], s=70, color=c, zorder=5, label=f"{nom}: umbral {thr[i]:.3f}, F1 {f[i]:.3f}")
        filas.append({"criterio": nom, "umbral": float(thr[i]), "F1": float(f[i]), "subcobertura": float(1 - rec[i]),
                      "filtracion": float(1 - prec[i]), "tasa_predicha": float((tp[i] + fp[i]) / len(y))})
    ax.set(xlim=(0, .7), ylim=(0, .7), xlabel="error de exclusión (subcobertura) = 1 − exhaustividad",
           ylabel="error de inclusión (filtración) = 1 − precisión", title=titulo)
    ax.legend(fontsize=8)
    _guardar(fig, ruta, mostrar)
    return pd.DataFrame(filas)


def f1_dominio(tabla: pd.DataFrame, f1_global, titulo, ruta=None, mostrar=True):
    plt = _plt()
    t = tabla.sort_values("F1")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(t) + 1.5))
    ax.errorbar(t.F1, t.Dominio, xerr=[t.F1 - t.IC95_inf, t.IC95_sup - t.F1], fmt="o", color=AZUL, capsize=2)
    ax.axvline(f1_global, color=ROJO, ls="--", label=f"global = {f1_global:.3f}")
    for _, r in t.iterrows():
        ax.text(r.IC95_sup + 0.005, r.Dominio, f"n = {r.hogares:,}".replace(",", "."), va="center", fontsize=7, color=GRIS)
    ax.set(xlabel="F1 anidado (IC 95 % bootstrap)", title=titulo); ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    return _guardar(fig, ruta, mostrar)


def estabilidad(conjuntos_pliegues, MP, titulo, ruta=None, top=40, mostrar=True, con_final=False):
    """Indicador de selección de cada pregunta en cada pliegue (y en el conjunto final) y su frecuencia."""
    plt = _plt()
    qs = [set(MP.preguntas_de(c)) for c in conjuntos_pliegues]
    todas = sorted(set().union(*qs), key=lambda q: -sum(q in s for s in qs))[:top]
    M = np.array([[q in s for s in qs] for q in todas], float)
    fig, ax = plt.subplots(figsize=(0.6 * len(qs) + 5, 0.28 * len(todas) + 1.5))
    ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=1); ax.grid(False)
    ax.set_yticks(range(len(todas))); ax.set_yticklabels([MP.etiqueta(q)[:45] for q in todas], fontsize=7)
    et = [f"pliegue {i + 1}" for i in range(len(qs))]
    if con_final:
        et[-1] = "todo el entrenamiento"
    ax.set_xticks(range(len(qs))); ax.set_xticklabels(et, rotation=45, fontsize=8)
    for i, q in enumerate(todas):
        ax.text(len(qs) - 0.4, i, f"{M[i, :len(qs) - (1 if con_final else 0)].mean():.0%}", va="center", fontsize=7)
    ax.set(title=titulo)
    return _guardar(fig, ruta, mostrar)


def busqueda(tabla: pd.DataFrame, imp: pd.Series, titulo, ruta=None, etiqueta="log-loss", mostrar=True):
    plt = _plt()
    c = tabla[tabla.estado == "COMPLETE"].sort_values("ensayo"); pr = tabla[tabla.estado == "PRUNED"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    for fase, col in (("sobol", AZUL), ("tpe", VERDE)):
        d = c[c.fase == fase]
        ax[0].scatter(d.ensayo, d.objetivo, s=22, color=col, label=f"{'Sobol' if fase == 'sobol' else 'TPE'} (completo)")
    if len(pr):
        ax[0].scatter(pr.ensayo, np.full(len(pr), np.nan), marker="x", color=GRIS, label=f"podados: {len(pr)}")
    if len(c):
        ax[0].plot(c.ensayo, c.objetivo.cummin(), color=ROJO, label="mejor hasta el ensayo")
        lo, hi = np.nanquantile(c.objetivo, [0, .9]) if len(c) > 3 else (c.objetivo.min(), c.objetivo.max())
        ax[0].set_ylim(lo - .1 * (hi - lo) - 1e-5, hi + .3 * (hi - lo) + 1e-5)
    ax[0].set(xlabel="ensayo", ylabel=f"{etiqueta} OOF en T", title="Historia de la búsqueda"); ax[0].legend(fontsize=8)
    if "F1_anidado" in c:
        ax[1].scatter(c.objetivo, c.F1_anidado, color=MORADO, s=22)
        ax[1].set(xlabel=f"{etiqueta} OOF en T", ylabel="F1 anidado", title="Pérdida propia frente a F1 (ensayos completos)")
    if len(imp):
        ax[2].barh(list(imp.index)[::-1], list(imp.values)[::-1], color=VERDE)
        ax[2].set(xlabel="fracción de la varianza del objetivo (fANOVA)", title="Importancia de los hiperparámetros")
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)


def comparaciones(comp: pd.DataFrame, titulo, ruta=None, tipo="logloss", mostrar=True):
    """Diferencias pareadas (A − referencia) con IC 95 % del t corregido de Nadeau–Bengio y la ROPE."""
    plt = _plt()
    if comp is None or comp.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.5 * len(comp) + 2.5))
    y = np.arange(len(comp))
    for a, met, rope, xl in ((ax[0], "dF1", MT.ROPE_F1, "ΔF1 anidado (A − referencia)"),
                             (ax[1], f"d{tipo}", MT.rope_de(tipo), f"Δ{tipo} (A − referencia; negativo = mejor)")):
        ic = np.array(comp[f"{met}_IC95"].tolist())
        a.errorbar(comp[met], y, xerr=[comp[met] - ic[:, 0], ic[:, 1] - comp[met]], fmt="o", color=AZUL, capsize=3)
        a.axvspan(-rope, rope, color=VERDE, alpha=.15, label=f"ROPE ±{rope}")
        a.axvline(0, color="k", lw=.8)
        a.set_yticks(y); a.set_yticklabels([f"{r.A}\nvs {r.B}" for r in comp.itertuples()], fontsize=8)
        a.set(xlabel=xl); a.legend(fontsize=8)
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, ruta, mostrar)
