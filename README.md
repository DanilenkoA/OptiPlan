# OptiPlan — systém optimalizácie výroby

Webová aplikácia na plánovanie výroby, ktorá pomocou lineárneho programovania nájde
výrobný plán s najvyšším možným ziskom pri rešpektovaní dopytu a dostupných zdrojov.
Celé rozhranie je v slovenčine.

## Štruktúra aplikácie

Navigácia je v bočnom paneli (spolu so stavom optimalizácie a stručným popisom modelu).

- **Domov** — úvodná stránka, prehľad stavu plánu a postup práce.
- **Optimalizácia** — zadanie produktov, cien, nákladov, dopytu a dostupných zdrojov; spustenie výpočtu.
- **Výsledky** — odporúčaný výrobný plán, tržby, náklady, zisk a bilancia zdrojov; export do CSV.
- **Analýza** — grafy objemu výroby, zisku, marže a využitia zdrojov (identifikácia úzkych miest).
- **O projekte** — účel aplikácie, matematický model a použité technológie.

## Štruktúra projektu

```
app.py                  aplikácia (stránky, model, grafy)
styles.css              vizuálny štýl (CSS premenné, komponenty, responzivita)
.streamlit/config.toml  téma Streamlitu (farby, písma) a nastavenia servera
requirements.txt        závislosti
```

## Matematický model

Cieľová funkcia maximalizuje celkový zisk:

```
Zisk = Σ (Predajná cena − Variabilné náklady) × Vyrobené množstvo
```

Obmedzenia:
- spotreba každého zdroja nesmie prekročiť jeho dostupné množstvo,
- výroba každého produktu nesmie prekročiť jeho dopyt,
- objem výroby je nezáporný.

Úloha sa rieši metódou HiGHS (knižnica SciPy, `scipy.optimize.linprog`).

## Spustenie vo VS Code

1. Otvorte priečinok projektu.
2. V termináli spustite: `py -m pip install -r requirements.txt`
3. Potom spustite: `py -m streamlit run app.py`
4. Otvorte `http://localhost:8501`.

Aplikácia bola overená so Streamlit 1.66 a Python 3.13.

## Poznámky

- Exportované CSV používa oddeľovač `;` a desatinnú čiarku, takže sa v slovenskom Exceli otvorí správne.
- Názov každého zdroja v tabuľke zdrojov musí presne zodpovedať názvu stĺpca spotreby v tabuľke produktov.
