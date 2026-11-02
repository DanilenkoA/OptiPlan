"""OptiPlan — systém optimalizácie výroby.

Streamlit aplikácia, ktorá pomocou lineárneho programovania (SciPy / HiGHS)
nájde výrobný plán s maximálnym ziskom pri obmedzení dopytu a zdrojov.
Vzhľad je definovaný v `styles.css` a `.streamlit/config.toml`.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="OptiPlan | Optimalizácia výroby",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Štýly
# ----------------------------------------------------------------------------
def load_css() -> None:
    css = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


load_css()

# ----------------------------------------------------------------------------
# Pomocné funkcie (formátovanie, HTML, kompatibilita)
# ----------------------------------------------------------------------------
NBSP = " "


def fnum(x: float, d: int = 2) -> str:
    """Slovenský formát čísla: 12 345,67"""
    return f"{x:,.{d}f}".replace(",", NBSP).replace(".", ",")


def eur(x: float, d: int = 2) -> str:
    return f"{fnum(x, d)}{NBSP}€"


def esc(x) -> str:
    return html.escape(str(x))


def md(markup: str) -> None:
    """Vloží HTML blok (bez prázdnych riadkov, aby ho Markdown nerozbil)."""
    st.markdown(markup, unsafe_allow_html=True)


def stretch(fn, *args, **kwargs):
    """Roztiahne prvok na šírku kontajnera naprieč verziami Streamlitu."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def icon(path: str) -> str:
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true">' + path + "</svg>"
    )


# Ikony Lucide (jednotný čiarový štýl)
ICO_SLIDERS = icon('<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/>')
ICO_CALC = icon('<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M8 6h8M16 14v4M16 10h.01M12 10h.01M8 10h.01M12 14h.01M8 14h.01M12 18h.01M8 18h.01"/>')
ICO_CHART = icon('<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9M13 17V5M8 17v-3"/>')
ICO_INFO = icon('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>')
ICO_ALERT = icon('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>')
ICO_CHECK = icon('<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>')

BRAND_MARK = (
    '<svg class="op-brand-mark" viewBox="0 0 34 34" aria-hidden="true">'
    '<rect width="34" height="34" rx="9" fill="#22c55e"/>'
    '<rect x="8" y="17" width="4.5" height="9" rx="1.2" fill="#12151a"/>'
    '<rect x="14.8" y="12" width="4.5" height="14" rx="1.2" fill="#12151a"/>'
    '<rect x="21.6" y="7" width="4.5" height="19" rx="1.2" fill="#12151a"/></svg>'
)

# ----------------------------------------------------------------------------
# Počiatočné dáta
# ----------------------------------------------------------------------------
def default_products() -> pd.DataFrame:
    return pd.DataFrame({
        "Produkt": ["Stôl Modern", "Stolička Komfort", "Skriňa Loft"],
        "Predajná cena (€)": [180, 75, 260],
        "Variabilné náklady (€)": [95, 38, 145],
        "Dopyt (ks)": [80, 150, 45],
        "Drevo (m²)": [3.0, 1.2, 5.0],
        "Kov (kg)": [2.0, 1.5, 4.0],
        "Práca (h)": [4.0, 1.5, 6.0],
    })


def default_resources() -> pd.DataFrame:
    return pd.DataFrame({
        "Zdroj": ["Drevo (m²)", "Kov (kg)", "Práca (h)"],
        "Dostupné množstvo": [300, 260, 420],
    })


PAGES = ["Domov", "Optimalizácia", "Výsledky", "Analýza", "O projekte"]

ss = st.session_state
ss.setdefault("nav", "Domov")
ss.setdefault("products", default_products())      # aktuálne (rozpracované) údaje
ss.setdefault("resources", default_resources())
ss.setdefault("p_base", default_products())        # základ pre editory
ss.setdefault("r_base", default_resources())
ss.setdefault("ed_ver", 0)                         # verzia editorov (vynúti ich nové vykreslenie)
ss.setdefault("result", None)
ss.setdefault("last_page", None)


def go_to(page: str) -> None:
    ss.nav = page


def reset_inputs() -> None:
    ss.products = default_products()
    ss.resources = default_resources()
    ss.p_base = default_products()
    ss.r_base = default_resources()
    ss.ed_ver += 1


# ----------------------------------------------------------------------------
# Matematický model (lineárne programovanie) — bez zmien
# ----------------------------------------------------------------------------
def solve(p: pd.DataFrame, r: pd.DataFrame):
    required = ["Produkt", "Predajná cena (€)", "Variabilné náklady (€)", "Dopyt (ks)"]
    if p.empty or r.empty or any(c not in p.columns for c in required):
        raise ValueError("Skontrolujte, či tabuľka produktov obsahuje všetky povinné stĺpce.")

    names = r["Zdroj"].astype(str).tolist()
    if not names or any(n not in p.columns for n in names):
        raise ValueError("Pre každý zdroj musí existovať zodpovedajúci stĺpec spotreby v tabuľke produktov.")

    p = p.copy()
    for c in required[1:]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    # Matica spotreby: riadky = zdroje (j), stĺpce = produkty (i), teda A[j, i] = spotreba zdroja j na 1 ks produktu i.
    # (Pôvodná verzia ju mala transponovanú, čo pri 3×3 údajoch dávalo nesprávny, nižší zisk.)
    A = p[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
    b = pd.to_numeric(r["Dostupné množstvo"], errors="coerce").to_numpy(dtype=float)

    if p[required[1:]].isna().any().any() or np.isnan(A).any() or np.isnan(b).any():
        raise ValueError("Všetky číselné polia musia obsahovať platné čísla.")
    if (A < 0).any() or (b < 0).any() or (p[required[1:]] < 0).any().any():
        raise ValueError("Záporné hodnoty nie sú v modeli povolené.")

    margin = (p["Predajná cena (€)"] - p["Variabilné náklady (€)"]).to_numpy(float)
    result = linprog(-margin, A_ub=A, b_ub=b,
                      bounds=[(0, float(x)) for x in p["Dopyt (ks)"]], method="highs")
    if not result.success:
        raise ValueError("Optimálne riešenie sa nepodarilo nájsť: " + result.message)

    q = result.x
    out = p[["Produkt", "Predajná cena (€)", "Variabilné náklady (€)"]].copy()
    out["Optimálna výroba (ks)"] = q
    out["Tržby (€)"] = q * p["Predajná cena (€)"].to_numpy()
    out["Náklady (€)"] = q * p["Variabilné náklady (€)"].to_numpy()
    out["Zisk (€)"] = q * margin

    used = A @ q
    balance = pd.DataFrame({"Zdroj": names, "Spotreba": used, "Dostupné množstvo": b})
    balance["Zostatok"] = balance["Dostupné množstvo"] - balance["Spotreba"]
    balance["Využitie (%)"] = np.where(b > 0, 100 * used / b, 0)
    return out, balance


# ----------------------------------------------------------------------------
# Príprava vstupov, spustenie modelu a odvodené ukazovatele
# ----------------------------------------------------------------------------
def prepare_products(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(how="all").reset_index(drop=True)
    if "Produkt" in d.columns:
        names = d["Produkt"].astype("object")
        blank = names.isna() | (names.astype(str).str.strip() == "")
        d["Produkt"] = [f"Produkt {i + 1}" if b else str(n).strip()
                        for i, (n, b) in enumerate(zip(names, blank))]
    return d


def prepare_resources(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(how="all").reset_index(drop=True)
    if "Zdroj" in d.columns:
        d["Zdroj"] = d["Zdroj"].astype("object").map(lambda x: x if pd.isna(x) else str(x).strip())
    return d


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and a.to_dict("list") == b.to_dict("list")


def run_model(products: pd.DataFrame, resources: pd.DataFrame) -> dict:
    p = prepare_products(products)
    r = prepare_resources(resources)
    out, balance = solve(p, r)

    out = out.copy()
    for c in ["Optimálna výroba (ks)", "Tržby (€)", "Náklady (€)", "Zisk (€)"]:
        out[c] = out[c].where(out[c].abs() > 1e-9, 0.0)        # odstránenie numerického šumu
    balance = balance.copy()
    for c in ["Spotreba", "Zostatok", "Využitie (%)"]:
        balance[c] = balance[c].where(balance[c].abs() > 1e-9, 0.0)

    out["Marža na kus (€)"] = out["Predajná cena (€)"] - out["Variabilné náklady (€)"]
    out["Marža (%)"] = np.where(out["Predajná cena (€)"] > 0,
                                 100 * out["Marža na kus (€)"] / out["Predajná cena (€)"], 0.0)
    out["Dopyt (ks)"] = pd.to_numeric(p["Dopyt (ks)"]).to_numpy(float)
    out["Pokrytie dopytu (%)"] = np.where(out["Dopyt (ks)"] > 0,
                                           100 * out["Optimálna výroba (ks)"] / out["Dopyt (ks)"], 0.0)
    return {"out": out, "balance": balance, "p": p, "r": r,
            "time": datetime.now().strftime("%H:%M")}


def is_stale() -> bool:
    res = ss.result
    if res is None:
        return False
    return not (same(prepare_products(ss.products), res["p"])
                and same(prepare_resources(ss.resources), res["r"]))


def totals(res: dict) -> dict:
    out = res["out"]
    revenue, costs, profit = out["Tržby (€)"].sum(), out["Náklady (€)"].sum(), out["Zisk (€)"].sum()
    return {
        "revenue": revenue, "costs": costs, "profit": profit,
        "margin": 100 * profit / revenue if revenue > 0 else 0.0,
        "units": out["Optimálna výroba (ks)"].sum(),
        "active": int((out["Optimálna výroba (ks)"] > 1e-6).sum()),
        "count": len(out),
    }


BOTTLENECK = 99.5   # využitie (%), od ktorého je zdroj považovaný za úzke miesto
HIGH_USE = 85.0


def util_state(u: float) -> tuple[str, str, str]:
    """(text, trieda odznaku, trieda lišty)"""
    if u >= BOTTLENECK:
        return "Úzke miesto", "warn", "warn"
    if u >= HIGH_USE:
        return "Vysoké využitie", "ok", ""
    return "Voľná kapacita", "off", ""


# ----------------------------------------------------------------------------
# Opakované HTML komponenty
# ----------------------------------------------------------------------------
def page_head(eyebrow: str, title: str, lead: str) -> None:
    md(f'<div class="op-pagehead"><div class="op-eyebrow">{esc(eyebrow)}</div>'
       f'<h1 class="op-title">{esc(title)}</h1><p class="op-lead">{esc(lead)}</p></div>')


def section(title: str, sub: str = "", step: str = "") -> None:
    badge = f'<span class="op-step">{esc(step)}</span>' if step else ""
    subtitle = f"<p>{esc(sub)}</p>" if sub else ""
    md(f'<div class="op-section"><div><h3>{badge}{esc(title)}</h3>{subtitle}</div></div>')


def note(text_html: str, kind: str = "", ico: str = ICO_INFO) -> None:
    md(f'<div class="op-note {kind}">{ico}<div>{text_html}</div></div>')


def kpi_html(label: str, value: str, sub: str, cls: str = "") -> str:
    return (f'<div class="op-kpi {cls}"><div class="op-kpi-label">{esc(label)}</div>'
            f'<div class="op-kpi-value">{value}</div><div class="op-kpi-note">{sub}</div></div>')


def bar_html(pct: float, label: str, cls: str = "") -> str:
    w = max(0.0, min(100.0, pct))
    return (f'<div class="op-bar"><div class="op-bar-track"><span class="op-bar-fill {cls}" '
            f'style="width:{w:.1f}%"></span></div><span class="op-bar-val">{label}</span></div>')


def empty_state() -> None:
    with st.container(key="panel_empty"):
        md('<div class="op-panel-title">Zatiaľ nie je k dispozícii žiadny výpočet</div>'
           '<p class="op-panel-sub">Zadajte vstupné údaje a spustite optimalizáciu — '
           'výsledky sa zobrazia v tejto sekcii.</p>')
        st.button("Prejsť na optimalizáciu", key="cta_empty", type="primary",
                  on_click=go_to, args=("Optimalizácia",))


# ----------------------------------------------------------------------------
# Stránka: Domov
# ----------------------------------------------------------------------------
def page_home() -> None:
    with st.container(key="hero"):
        left, right = st.columns([1.5, 1], gap="large", vertical_alignment="center")
        with left:
            md('<div class="op-hero-kicker">Plánovanie výroby · Lineárna optimalizácia</div>'
               '<h1 class="op-hero-title">Plánujte múdrejšie.<em>Vyrábajte ziskovejšie.</em></h1>'
               '<p class="op-hero-text">OptiPlan z cien, nákladov, dopytu a dostupných zdrojov vypočíta '
               'výrobný plán s najvyšším možným ziskom — prehľadne, transparentne a v priebehu sekúnd.</p>')
            with st.container(key="btnrow_hero"):
                b1, b2 = st.columns(2, gap="small")
                with b1:
                    st.button("Prejsť na optimalizáciu", key="cta_hero", type="primary",
                              on_click=go_to, args=("Optimalizácia",))
                with b2:
                    st.button("O modeli", key="hero_about", on_click=go_to, args=("O projekte",))
        with right:
            md('<div class="op-model-card"><h4>Model v skratke</h4>'
               '<div class="op-model-row"><span>Cieľ</span><b>Maximalizácia zisku</b></div>'
               '<div class="op-model-row"><span>Obmedzenia</span><b>Zdroje a dopyt</b></div>'
               '<div class="op-model-row"><span>Premenné</span><b>Objem výroby ≥ 0</b></div>'
               '<div class="op-model-row"><span>Riešiteľ</span><b>HiGHS (SciPy)</b></div>'
               '<div class="op-formula">Zisk = Σ (cena − var. náklady) × množstvo</div></div>')

    res = ss.result
    n_products = len(prepare_products(ss.products))
    n_resources = len(prepare_resources(ss.resources))
    if res is None:
        status, status_note, profit_val, profit_note, pos = "Nevypočítaný", "Spustite optimalizáciu", "—", "Po výpočte", ""
    else:
        t = totals(res)
        status = "Vypočítaný"
        status_note = f"Posledný výpočet o {res['time']}" + (" · vstupy sa zmenili" if is_stale() else "")
        profit_val, profit_note, pos = eur(t["profit"]), f"Marža {fnum(t['margin'], 1)}{NBSP}%", "pos"
    md('<div class="op-stats">'
       f'<div class="op-stat"><div class="op-stat-label">Produkty vo vstupe</div><div class="op-stat-value">{n_products}</div><div class="op-stat-note">Položky v tabuľke produktov</div></div>'
       f'<div class="op-stat"><div class="op-stat-label">Sledované zdroje</div><div class="op-stat-value">{n_resources}</div><div class="op-stat-note">Kapacitné obmedzenia</div></div>'
       f'<div class="op-stat"><div class="op-stat-label">Stav plánu</div><div class="op-stat-value">{status}</div><div class="op-stat-note">{esc(status_note)}</div></div>'
       f'<div class="op-stat"><div class="op-stat-label">Maximálny zisk</div><div class="op-stat-value {pos}">{profit_val}</div><div class="op-stat-note">{profit_note}</div></div>'
       '</div>')

    section("Ako OptiPlan funguje", "Tri kroky od vstupných údajov k odporúčanému výrobnému plánu.")
    md('<div class="op-steps">'
       f'<div class="op-step-item"><div class="op-step-ico">{ICO_SLIDERS}</div><div class="op-step-no">KROK 1</div><h4>Zadajte vstupné údaje</h4>'
       '<p>Doplňte produkty, predajné ceny, variabilné náklady, dopyt, spotrebu zdrojov a ich dostupné množstvo.</p></div>'
       f'<div class="op-step-item"><div class="op-step-ico">{ICO_CALC}</div><div class="op-step-no">KROK 2</div><h4>Spustite optimalizáciu</h4>'
       '<p>Model lineárneho programovania nájde plán s najvyšším ziskom pri dodržaní všetkých obmedzení.</p></div>'
       f'<div class="op-step-item"><div class="op-step-ico">{ICO_CHART}</div><div class="op-step-no">KROK 3</div><h4>Vyhodnoťte plán</h4>'
       '<p>Prezrite si objemy výroby, tržby, náklady, zisk a využitie zdrojov vrátane úzkych miest.</p></div>'
       '</div>')


# ----------------------------------------------------------------------------
# Stránka: Optimalizácia
# ----------------------------------------------------------------------------
def product_column_config(df: pd.DataFrame) -> dict:
    cfg = {"Produkt": st.column_config.TextColumn("Produkt", required=True, width="medium")}
    fmt = {"Predajná cena (€)": "%.2f", "Variabilné náklady (€)": "%.2f", "Dopyt (ks)": "%.0f"}
    for c in df.columns:
        if c == "Produkt":
            continue
        cfg[c] = st.column_config.NumberColumn(c, min_value=0, format=fmt.get(c, "%.2f"))
    return cfg


def page_optimize() -> None:
    page_head("Vstupné údaje", "Optimalizácia výroby",
              "Zadajte produkty, ceny, náklady, dopyt a dostupné zdroje. Po spustení výpočtu model nájde "
              "plán s maximálnym ziskom.")

    # Pri návrate na stránku sa editory znova vytvoria z posledných rozpracovaných údajov.
    if ss.last_page != "Optimalizácia":
        ss.p_base = ss.products.copy()
        ss.r_base = ss.resources.copy()
        ss.ed_ver += 1
    ver = ss.ed_ver

    section("Produkty a náklady", "Predajné ceny, variabilné náklady, dopyt a spotreba zdrojov na jeden kus.", "01")
    with st.container(key="panel_products"):
        st.caption("Riadky môžete pridávať a odstraňovať. Stĺpce spotreby musia zodpovedať názvom zdrojov "
                   "v tabuľke nižšie. Prázdne riadky sa pri výpočte ignorujú.")
        products = stretch(st.data_editor, ss.p_base, hide_index=True, num_rows="dynamic",
                           key=f"p_editor_{ver}", column_config=product_column_config(ss.p_base),
                           row_height=44)
    ss.products = products

    section("Dostupné zdroje", "Maximálne množstvo každého zdroja, ktoré je k dispozícii pre plánované obdobie.", "02")
    with st.container(key="row_resources"):
        c_res, c_help = st.columns([5, 7], gap="medium")
        with c_res:
            with st.container(key="panel_resources"):
                resources = stretch(
                    st.data_editor, ss.r_base, hide_index=True, num_rows="dynamic", key=f"r_editor_{ver}",
                    column_config={
                        "Zdroj": st.column_config.TextColumn("Zdroj", required=True),
                        "Dostupné množstvo": st.column_config.NumberColumn("Dostupné množstvo", min_value=0, format="%.2f"),
                    },
                    row_height=44)
        ss.resources = resources
        with c_help:
            with st.container(key="help_col"):
                note("<b>Názov zdroja musí presne zodpovedať názvu stĺpca spotreby</b> v tabuľke produktov "
                     "(napríklad „Drevo (m²)“). Pri nezhode model upozorní na chybu. Ak zdroj odstránite z tabuľky "
                     "zdrojov, model ho pri výpočte nezohľadní.")
                note("<b>Model počíta so spojitými množstvami.</b> Odporúčaný objem výroby môže byť desatinné číslo; "
                     "pre prax ho zaokrúhlite podľa technológie výroby.", ico=ICO_INFO)

    run = False
    with st.container(key="row_actions"):
        c_btns, _ = st.columns([5, 7], gap="medium")   # rovnaká šírka ako karta zdrojov
        with c_btns:
            c1, c2 = st.columns([1.25, 1], gap="small")
            with c1:
                run = st.button("Vypočítať optimálny plán", key="cta_run", type="primary")
            with c2:
                st.button("Obnoviť ukážkové údaje", key="reset", on_click=reset_inputs)

    if run:
        try:
            ss.result = run_model(products, resources)
        except Exception as e:  # noqa: BLE001 — chyby modelu zobrazujeme používateľovi
            ss.result = None
            st.error(str(e))

    res = ss.result
    if res is not None:
        t = totals(res)
        section("Výsledok posledného výpočtu", f"Vypočítané o {res['time']}.")
        if is_stale():
            note("<b>Vstupné údaje sa zmenili od posledného výpočtu.</b> Spustite výpočet znova, "
                 "aby výsledky zodpovedali aktuálnym údajom.", "amber", ICO_ALERT)
        else:
            note("<b>Optimálny plán bol úspešne vypočítaný.</b> Všetky obmedzenia zdrojov a dopytu sú splnené.",
                 "green", ICO_CHECK)
        md('<div class="op-stats">'
           f'<div class="op-stat"><div class="op-stat-label">Maximálny zisk</div><div class="op-stat-value pos">{eur(t["profit"])}</div><div class="op-stat-note">Marža {fnum(t["margin"], 1)}{NBSP}%</div></div>'
           f'<div class="op-stat"><div class="op-stat-label">Celkové tržby</div><div class="op-stat-value">{eur(t["revenue"])}</div><div class="op-stat-note">Z predaja výrobkov</div></div>'
           f'<div class="op-stat"><div class="op-stat-label">Variabilné náklady</div><div class="op-stat-value">{eur(t["costs"])}</div><div class="op-stat-note">Náklady na výrobu</div></div>'
           f'<div class="op-stat"><div class="op-stat-label">Produkty vo výrobe</div><div class="op-stat-value">{t["active"]} z {t["count"]}</div><div class="op-stat-note">S nenulovou výrobou</div></div>'
           '</div>')
        with st.container(key="btnrow_after"):
            b1, b2 = st.columns(2, gap="small")
            with b1:
                st.button("Zobraziť výsledky", key="cta_to_results", type="primary", on_click=go_to, args=("Výsledky",))
            with b2:
                st.button("Otvoriť analýzu", key="to_analysis", on_click=go_to, args=("Analýza",))


# ----------------------------------------------------------------------------
# Stránka: Výsledky
# ----------------------------------------------------------------------------
def plan_table(out: pd.DataFrame) -> str:
    head = ("<thead><tr>"
            "<th>Produkt</th>"
            '<th class="num">Predajná cena<small>€ / ks</small></th>'
            '<th class="num">Var. náklady<small>€ / ks</small></th>'
            '<th class="num">Marža<small>€ / ks</small></th>'
            '<th class="num">Dopyt<small>ks</small></th>'
            '<th class="num">Optimálna výroba<small>ks</small></th>'
            "<th>Pokrytie dopytu</th>"
            '<th class="num">Tržby<small>€</small></th>'
            '<th class="num">Náklady<small>€</small></th>'
            '<th class="num">Zisk<small>€</small></th>'
            "</tr></thead>")
    rows = []
    for _, r in out.iterrows():
        q = r["Optimálna výroba (ks)"]
        qcls = "strong" if q > 1e-6 else "zero"
        rows.append(
            "<tr>"
            f'<td class="name">{esc(r["Produkt"])}</td>'
            f'<td class="num">{fnum(r["Predajná cena (€)"])}</td>'
            f'<td class="num">{fnum(r["Variabilné náklady (€)"])}</td>'
            f'<td class="num">{fnum(r["Marža na kus (€)"])}</td>'
            f'<td class="num">{fnum(r["Dopyt (ks)"], 0)}</td>'
            f'<td class="num {qcls}">{fnum(q)}</td>'
            f'<td>{bar_html(r["Pokrytie dopytu (%)"], fnum(r["Pokrytie dopytu (%)"], 0) + NBSP + "%")}</td>'
            f'<td class="num">{fnum(r["Tržby (€)"])}</td>'
            f'<td class="num">{fnum(r["Náklady (€)"])}</td>'
            f'<td class="num pos">{fnum(r["Zisk (€)"])}</td>'
            "</tr>")
    foot = ("<tfoot><tr><td>Spolu</td><td></td><td></td><td></td><td></td>"
            f'<td class="num">{fnum(out["Optimálna výroba (ks)"].sum())}</td><td></td>'
            f'<td class="num">{fnum(out["Tržby (€)"].sum())}</td>'
            f'<td class="num">{fnum(out["Náklady (€)"].sum())}</td>'
            f'<td class="num pos">{fnum(out["Zisk (€)"].sum())}</td></tr></tfoot>')
    return f'<div class="op-table-wrap"><table class="op-table">{head}<tbody>{"".join(rows)}</tbody>{foot}</table></div>'


def balance_table(bal: pd.DataFrame) -> str:
    head = ("<thead><tr><th>Zdroj</th>"
            '<th class="num">Dostupné množstvo</th><th class="num">Spotreba</th><th class="num">Zostatok</th>'
            "<th>Využitie</th><th>Stav</th></tr></thead>")
    rows = []
    for _, r in bal.iterrows():
        u = float(r["Využitie (%)"])
        text, badge, bar = util_state(u)
        rows.append(
            "<tr>"
            f'<td class="name">{esc(r["Zdroj"])}</td>'
            f'<td class="num">{fnum(r["Dostupné množstvo"])}</td>'
            f'<td class="num strong">{fnum(r["Spotreba"])}</td>'
            f'<td class="num">{fnum(r["Zostatok"])}</td>'
            f'<td>{bar_html(u, fnum(u, 1) + NBSP + "%", bar)}</td>'
            f'<td><span class="op-badge {badge}">{text}</span></td>'
            "</tr>")
    return f'<div class="op-table-wrap"><table class="op-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'


def page_results() -> None:
    page_head("Výstup modelu", "Výsledky optimalizácie",
              "Odporúčaný výrobný plán, finančné ukazovatele a bilancia zdrojov podľa posledného výpočtu.")
    res = ss.result
    if res is None:
        empty_state()
        return
    out, bal, t = res["out"], res["balance"], totals(res)

    if is_stale():
        note("<b>Vstupné údaje sa zmenili od posledného výpočtu.</b> Zobrazené výsledky zodpovedajú "
             "predchádzajúcim údajom — spustite výpočet znova v sekcii Optimalizácia.", "amber", ICO_ALERT)

    md('<div class="op-kpis">'
       + kpi_html("Maximálny zisk", eur(t["profit"]), f"Marža {fnum(t['margin'], 1)}{NBSP}% z tržieb", "primary")
       + kpi_html("Celkové tržby", eur(t["revenue"]), "Z predaja odporúčaného objemu")
       + kpi_html("Variabilné náklady", eur(t["costs"]), "Náklady na výrobu plánu")
       + kpi_html("Optimálny objem výroby", f'{fnum(t["units"], 1)}{NBSP}ks', f'{t["active"]} z {t["count"]} produktov vo výrobe')
       + "</div>")

    section("Odporúčaný výrobný plán", "Optimálny objem výroby jednotlivých produktov a jeho finančný dopad.")
    md(plan_table(out))
    with st.container(key="btnrow_dl"):
        d1, d2 = st.columns(2, gap="small")
        with d1:
            st.download_button("Stiahnuť plán (CSV)",
                               out.round(2).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               "optiplan_plan.csv", "text/csv", key="dl_plan")
        with d2:
            st.download_button("Stiahnuť bilanciu zdrojov (CSV)",
                               bal.round(2).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               "optiplan_zdroje.csv", "text/csv", key="dl_bal")

    section("Využitie zdrojov", "Porovnanie spotreby s dostupným množstvom. Úzke miesta obmedzujú ďalší rast zisku.")
    md(balance_table(bal))
    st.button("Otvoriť analýzu", key="to_analysis_r", on_click=go_to, args=("Analýza",))


# ----------------------------------------------------------------------------
# Stránka: Analýza
# ----------------------------------------------------------------------------
C_GREEN, C_INK, C_GREY, C_LIGHT, C_AMBER = "#22c55e", "#12151a", "#c9cfd9", "#e8ebf0", "#b7791f"


def style_fig(fig: go.Figure, height: int = 330, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=56, t=(12 if layout.get("showlegend") is False else 44), b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#3b4049"),
        separators=f",{NBSP}",
        hoverlabel=dict(bgcolor=C_INK, font=dict(color="#ffffff", family="Inter, sans-serif", size=12), bordercolor=C_INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
        **layout,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e3e6ec", ticks="", automargin=True)
    fig.update_yaxes(gridcolor="#eef0f4", zeroline=False, showline=False, ticks="", automargin=True)
    return fig


def chart(fig: go.Figure, key: str) -> None:
    stretch(st.plotly_chart, fig, key=key, config={"displayModeBar": False})


def panel_title(title: str, sub: str) -> None:
    md(f'<div><div class="op-panel-title">{esc(title)}</div><p class="op-panel-sub">{esc(sub)}</p></div>')


def insight(res: dict) -> None:
    out, bal = res["out"], res["balance"]
    tight = bal[bal["Využitie (%)"] >= BOTTLENECK]["Zdroj"].tolist()
    idle = out[out["Optimálna výroba (ks)"] <= 1e-6]["Produkt"].tolist()
    capped = bool((out["Optimálna výroba (ks)"] >= out["Dopyt (ks)"] - 1e-6).all())
    parts = []
    if tight:
        names = ", ".join(f"<b>{esc(n)}</b>" for n in tight)
        parts.append(f"Výrobu obmedzujú zdroje: {names}. Navýšenie ich kapacity môže zvýšiť zisk, "
                     "kým výrobu neobmedzí dopyt alebo iný zdroj.")
        kind, ico = "amber", ICO_ALERT
    elif capped:
        parts.append("<b>Plán pokrýva celý dopyt a žiadny zdroj nie je vyčerpaný</b> — výrobu obmedzuje len dopyt.")
        kind, ico = "green", ICO_CHECK
    else:
        parts.append("Žiadny zdroj nie je vyčerpaný na hranicu svojej kapacity.")
        kind, ico = "", ICO_INFO
    if idle:
        parts.append("V optimálnom pláne sa nevyrábajú: " + ", ".join(esc(n) for n in idle) + ".")
    note(" ".join(parts), kind, ico)


def page_analysis() -> None:
    page_head("Analytický prehľad", "Analýza výrobného plánu",
              "Grafické porovnanie objemu výroby, zisku, marží a využitia zdrojov. "
              "Pomáha identifikovať úzke miesta výroby.")
    res = ss.result
    if res is None:
        empty_state()
        return
    out, bal = res["out"], res["balance"]

    if is_stale():
        note("<b>Vstupné údaje sa zmenili od posledného výpočtu.</b> Grafy zodpovedajú predchádzajúcim údajom.",
             "amber", ICO_ALERT)
    insight(res)

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(key="panel_volume"):
            panel_title("Optimálny objem výroby", "Odporúčaná výroba v porovnaní s dopytom (ks)")
            fig = go.Figure()
            fig.add_bar(x=out["Produkt"], y=out["Dopyt (ks)"], name="Dopyt", marker_color=C_GREY,
                        hovertemplate="%{x}<br>Dopyt: %{y:,.0f} ks<extra></extra>")
            fig.add_bar(x=out["Produkt"], y=out["Optimálna výroba (ks)"], name="Optimálna výroba", marker_color=C_GREEN,
                        text=out["Optimálna výroba (ks)"], texttemplate="%{text:,.1f}", textposition="outside",
                        cliponaxis=False, hovertemplate="%{x}<br>Výroba: %{y:,.2f} ks<extra></extra>")
            style_fig(fig, barmode="group", bargap=0.35, bargroupgap=0.08, yaxis_title="Množstvo (ks)")
            chart(fig, "ch_volume")
    with g2:
        with st.container(key="panel_profit"):
            panel_title("Zisk podľa produktu", "Príspevok jednotlivých produktov k celkovému zisku (€)")
            d = out.sort_values("Zisk (€)")
            fig = go.Figure(go.Bar(
                x=d["Zisk (€)"], y=d["Produkt"], orientation="h", marker_color=C_INK,
                text=d["Zisk (€)"], texttemplate="%{text:,.0f}" + NBSP + "€", textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>Zisk: %{x:,.2f} €<extra></extra>"))
            style_fig(fig, showlegend=False, bargap=0.4)
            fig.update_xaxes(showgrid=True, gridcolor="#eef0f4", title="Zisk (€)", range=[0, max(1.0, float(d["Zisk (€)"].max())) * 1.22])
            fig.update_yaxes(showgrid=False)
            chart(fig, "ch_profit")

    g3, g4 = st.columns(2, gap="medium")
    with g3:
        with st.container(key="panel_margin"):
            panel_title("Marža podľa produktu", "Podiel marže na predajnej cene (%)")
            d = out.sort_values("Marža (%)")
            fig = go.Figure(go.Bar(
                x=d["Marža (%)"], y=d["Produkt"], orientation="h", marker_color=C_GREEN,
                text=d["Marža (%)"], texttemplate="%{text:,.1f}" + NBSP + "%", textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>Marža: %{x:,.1f} %<extra></extra>"))
            style_fig(fig, showlegend=False, bargap=0.4)
            fig.update_xaxes(showgrid=True, gridcolor="#eef0f4", title="Marža (%)", range=[0, max(100.0, float(d["Marža (%)"].max()) * 1.15)])
            fig.update_yaxes(showgrid=False)
            chart(fig, "ch_margin")
    with g4:
        with st.container(key="panel_util"):
            panel_title("Využitie zdrojov", "Spotreba voči dostupnému množstvu; úzke miesta sú zvýraznené")
            used = bal["Využitie (%)"].clip(lower=0, upper=100)
            colors = [C_AMBER if u >= BOTTLENECK else C_GREEN for u in used]
            detail = [f"{fnum(s)} z {fnum(a)}" for s, a in zip(bal["Spotreba"], bal["Dostupné množstvo"])]
            fig = go.Figure()
            fig.add_bar(y=bal["Zdroj"], x=used, orientation="h", name="Spotreba", marker_color=colors,
                        text=[f"{fnum(u, 1)}{NBSP}%" for u in used], textposition="inside", insidetextanchor="start",
                        textfont=dict(color="#ffffff", size=12), customdata=detail,
                        hovertemplate="%{y}<br>Využitie: %{x:,.1f} %<br>%{customdata}<extra></extra>")
            fig.add_bar(y=bal["Zdroj"], x=100 - used, orientation="h", name="Voľná kapacita", marker_color=C_LIGHT,
                        hovertemplate="%{y}<br>Voľná kapacita: %{x:,.1f} %<extra></extra>")
            style_fig(fig, barmode="stack", bargap=0.45, showlegend=False)
            fig.update_xaxes(range=[0, 100], ticksuffix=NBSP + "%", showgrid=True, gridcolor="#eef0f4")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            chart(fig, "ch_util")


# ----------------------------------------------------------------------------
# Stránka: O projekte
# ----------------------------------------------------------------------------
def page_about() -> None:
    page_head("O projekte", "OptiPlan — systém optimalizácie výroby",
              "Analytický nástroj na plánovanie výroby založený na lineárnom programovaní.")

    a, b = st.columns([1, 1.15], gap="medium")
    with a:
        with st.container(key="panel_purpose"):
            md('<div class="op-panel-title">Účel aplikácie</div>'
               '<p class="op-panel-sub" style="font-size:13.5px;color:#3b4049;line-height:1.65">'
               'OptiPlan pomáha nájsť kombináciu výrobkov, ktorá maximalizuje zisk pri súčasnom dodržaní '
               'dostupných zdrojov (materiál, pracovný čas, kapacity) a obmedzení dopytu na trhu. '
               'Výsledkom je odporúčaný výrobný plán, finančné ukazovatele a prehľad využitia zdrojov '
               'vrátane identifikácie úzkych miest.</p>')
            md('<div class="op-panel-title" style="margin-top:8px">Postup práce</div>'
               '<p class="op-panel-sub" style="font-size:13.5px;color:#3b4049;line-height:1.8">'
               '1. V sekcii <b>Optimalizácia</b> zadajte produkty a dostupné zdroje.<br>'
               '2. Spustite výpočet tlačidlom <b>Vypočítať optimálny plán</b>.<br>'
               '3. Plán nájdete v sekcii <b>Výsledky</b>, grafy v sekcii <b>Analýza</b>.<br>'
               '4. Výsledky môžete exportovať vo formáte CSV.</p>')
    with b:
        with st.container(key="panel_model"):
            md('<div class="op-panel-title">Matematický model</div>'
               '<p class="op-panel-sub">Úloha lineárneho programovania s cieľovou funkciou a obmedzeniami.</p>')
            st.latex(r"\max\; Z = \sum_{i=1}^{n} (c_i - v_i)\, x_i")
            st.latex(r"\sum_{i=1}^{n} a_{ji}\, x_i \le b_j \quad (j = 1,\dots,m), \qquad 0 \le x_i \le d_i")
            md('<p class="op-panel-sub" style="line-height:1.8">'
               '<b>x<sub>i</sub></b> — vyrobené množstvo produktu i &nbsp;·&nbsp; '
               '<b>c<sub>i</sub></b> — predajná cena &nbsp;·&nbsp; '
               '<b>v<sub>i</sub></b> — variabilné náklady na kus<br>'
               '<b>a<sub>ji</sub></b> — spotreba zdroja j na jeden kus produktu i &nbsp;·&nbsp; '
               '<b>b<sub>j</sub></b> — dostupné množstvo zdroja j &nbsp;·&nbsp; '
               '<b>d<sub>i</sub></b> — dopyt po produkte i</p>')

    section("Použité technológie")
    md('<div class="op-table-wrap"><table class="op-table"><thead><tr><th>Oblasť</th><th>Technológia</th><th>Úloha</th></tr></thead><tbody>'
       '<tr><td class="name">Rozhranie</td><td>Streamlit, HTML a CSS</td><td>Webová aplikácia a vlastný vizuálny štýl</td></tr>'
       '<tr><td class="name">Optimalizácia</td><td>SciPy (<code>linprog</code>, HiGHS)</td><td>Riešenie úlohy lineárneho programovania</td></tr>'
       '<tr><td class="name">Dáta</td><td>Pandas, NumPy</td><td>Spracovanie vstupov a výsledkov</td></tr>'
       '<tr><td class="name">Vizualizácia</td><td>Plotly</td><td>Interaktívne grafy</td></tr>'
       '</tbody></table></div>')

    section("Predpoklady modelu")
    note("Náklady, ceny a spotreba zdrojov sú lineárne — rastú úmerne s objemom výroby. Model plánuje jedno obdobie "
         "a pracuje so spojitými množstvami, preto môže navrhnúť aj desatinný počet kusov. "
         "Dopyt je chápaný ako horná hranica predaja.")


# ----------------------------------------------------------------------------
# Bočný panel (vykresľuje sa po stránke, aby stav zodpovedal aktuálnemu výpočtu)
# ----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        md(f'<div class="op-brand">{BRAND_MARK}<div>'
           '<div class="op-brand-name">Opti<b>Plan</b></div>'
           '<div class="op-brand-sub">Systém optimalizácie výroby</div></div></div>')
        md('<div class="op-side-label">Navigácia</div>')
        st.radio("Navigácia", PAGES, key="nav", label_visibility="collapsed")
        md('<div class="op-side-label">Stav</div>')

        res = ss.result
        if res is None:
            md('<div class="op-side-card"><div class="op-side-title">Stav optimalizácie</div>'
               '<div class="op-status"><span class="op-dot"></span><span>Plán nebol vypočítaný</span></div>'
               '<p class="op-side-text" style="margin-top:8px">Zadajte údaje a spustite výpočet.</p></div>')
        else:
            t = totals(res)
            stale = is_stale()
            dot, label = ("warn", "Vstupy sa zmenili") if stale else ("ok", "Plán vypočítaný")
            md(f'<div class="op-side-card"><div class="op-side-title">Stav optimalizácie</div>'
               f'<div class="op-status"><span class="op-dot {dot}"></span><span>{label}</span></div>'
               f'<div class="op-side-row"><span>Zisk</span><b>{eur(t["profit"], 0)}</b></div>'
               f'<div class="op-side-row"><span>Vypočítané</span><b>{res["time"]}</b></div></div>')

        md('<div class="op-side-card"><div class="op-side-title">Model</div>'
           '<p class="op-side-text">Lineárne programovanie (HiGHS) s obmedzeniami dopytu a dostupnosti zdrojov.</p></div>')


# ----------------------------------------------------------------------------
# Spustenie
# ----------------------------------------------------------------------------
PAGE_FUNCS = {
    "Domov": page_home,
    "Optimalizácia": page_optimize,
    "Výsledky": page_results,
    "Analýza": page_analysis,
    "O projekte": page_about,
}

PAGE_FUNCS[ss.nav]()
render_sidebar()
ss.last_page = ss.nav