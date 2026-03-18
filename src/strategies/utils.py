import re
import numpy as np
import pandas as pd
from datetime import datetime



def _get_data():
    df = pd.read_csv("bot_messages.csv")
    recap_mask = ["🏁 • 🆚" in x and "@" not in x and "🤖" not in x for x in df["text"]]
    df_text = df[recap_mask]['text'][:: - 1]
    return df_text



def _extract_matches(text: str, previous_date: str | None) -> list[dict]:
    if not text:
        return []

    text = str(text)

    date_match = re.search(r"Today's Matches \((\d{2}-\d{2}-\d{4})\)", text)
    match_date = date_match.group(1) if date_match else previous_date

    rows = []

    # Divide il testo in blocchi, uno per match
    blocks = re.split(r'(?=🏁\s*•\s*🆚)', text)

    for block in blocks:
        block = block.strip()

        # Salta tutto ciò che non è un blocco match valido
        if not block.startswith("🏁"):
            continue

        teams_match = re.search(r"🏁\s*•\s*🆚\s*(.*?)\s*-\s*(.*?)\n", block)
        time_match = re.search(r"🕒\s*Time:\s*(.*?)\n", block)
        league_match = re.search(r"🏆\s*League:\s*(.*?)\n", block)
        match_id_match = re.search(r"🆔\s*Match ID:\s*(\S+)", block)

        strategies_match = re.search(
            r"[🧩🧠]\s*Strategies:\s*\n(.*?)(?=\n\s*🆔\s*Match ID:|\Z)",
            block,
            re.S
        )

        # Se il blocco è incompleto, skippalo
        if not all([teams_match, time_match, league_match, strategies_match, match_id_match]):
            continue

        home_team = teams_match.group(1).strip()
        away_team = teams_match.group(2).strip()
        time = time_match.group(1).strip()
        league = league_match.group(1).strip()
        match_id = match_id_match.group(1).strip()
        strategies_block = strategies_match.group(1)

        for line in strategies_block.splitlines():
            line = line.strip()
            if not line:
                continue

            m = re.match(r"([✅❌❓])\s*(.*?)\s*\(([+-]?\d+(?:\.\d+)?)\)\s*$", line)
            if not m:
                continue

            icon, strategy, ret = m.groups()

            # Skippa strategie con ❓
            if icon == "❓":
                continue

            rows.append({
                "date": match_date,
                "match_id": match_id,
                "time": time,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "strategy": strategy.strip(),
                "result": True if icon == "✅" else False,
                "return": float(ret),
            })

    return rows



def _normalize_strategies(strategy):
    if strategy == "Win X2 HT":
        return "Win x2 HT"
    if strategy == "Over 1.5 (M-2X)":
        return "Over 1.5 (M2-X)"
    return strategy



def get_match_df():
    df_text = _get_data()
    tot_matches = list()
    previous_date = None
    for row in df_text.values:
        item = _extract_matches(row, previous_date=previous_date)
        for d in item:
            if d["date"]:
                previous_date = d["date"]
        tot_matches += item

    df_final = pd.DataFrame(tot_matches)
    df_final = df_final.set_index("match_id", drop=True)
    df_final["date"] = [datetime.strptime(x, "%d-%m-%Y") for x in df_final["date"]]
    df_final["week"] = df_final["date"].dt.to_period("W")

    df_final = df_final.sort_values("date") 

    monday_start = df_final["date"].min() - pd.to_timedelta(df_final["date"].min().weekday(), unit="D")
    df_final["week"] = ((df_final["date"] - monday_start).dt.days // 7) + 1
    df_final["strategy"] = [_normalize_strategies(x) for x in df_final["strategy"]]
    return df_final

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def analyze_strategies(
    df: pd.DataFrame,
    strategies,
    ci: float = 0.95,
    print_pi: bool = False
):
    """
    ci_low e ci_high: intervallo di confidenza del return medio settimanale atteso del processo. Sotto l'ipotesi che il 
    processo resti stabile, il vero guadagno medio settimanale della strategia è plausibilmente compreso tra ci_low e ci_high con confidenza 95%.

    pi_low e pi_high: intervallo di previsione per una singola prossima settimana. Sotto l'ipotesi che il processo resti stabile, 
    il return aggregato osservato in una futura singola settimana può plausibilmente cadere tra pi_low e pi_high con confidenza 95%.

    All'aumentare delle settimane:

    - conosci sempre meglio la media vera -> l'intervallo di ci_low e ci_high si assottiglia
    - resta comunque la variabilità intrinseca della singola settimana futura.
 
    Quindi:
    - il CI si assottiglia molto
    - il PI si assottiglia poco e poi si stabilizza intorno alla volatilità naturale del processo
    """
    
    if isinstance(strategies, str):
        strategies = [strategies]

    if not strategies:
        raise ValueError("La lista 'strategies' è vuota")

    # filtro strategie
    df_strategy = df[df["strategy"].isin(strategies)].copy()

    if df_strategy.empty:
        raise ValueError(f"Nessun dato trovato per strategies={strategies}")

    # aggregazione settimanale: somma dei return per settimana
    # se hai più strategy, qui vengono già sommate insieme
    df_aggregate = (
        df_strategy.groupby("week", as_index=True)["return"]
        .sum()
        .sort_index()
    )

    n = len(df_aggregate)
    mean = df_aggregate.mean()
    std = df_aggregate.std(ddof=1)

    if n < 2:
        sem = np.nan
        ci_low, ci_high = np.nan, np.nan
        pi_low, pi_high = np.nan, np.nan
    else:
        alpha = 1 - ci
        t_crit = t.ppf(1 - alpha / 2, df=n - 1)

        # Confidence interval della media
        sem = std / np.sqrt(n)
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem

        # Prediction interval della prossima osservazione settimanale
        pred_se = std * np.sqrt(1 + 1 / n)
        pi_low = mean - t_crit * pred_se
        pi_high = mean + t_crit * pred_se

    # etichetta leggibile
    strategy_label = strategies[0] if len(strategies) == 1 else " + ".join(strategies)

    # plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    x = df_aggregate.index
    y = df_aggregate.values

    ax.plot(
        x, y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=f"{strategy_label} weekly aggregated return"
    )

    # area sopra/sotto zero
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.12, interpolate=True)
    ax.fill_between(x, y, 0, where=(y < 0), alpha=0.12, interpolate=True)

    # linee statistiche
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axhline(
        mean, linestyle="-", linewidth=2,
        label=f"Mean = {mean:.3f}"
    )

    if n >= 2:
        # Confidence interval della media
        ax.fill_between(
            x,
            ci_low,
            ci_high,
            alpha=0.12,
            label=f"{int(ci*100)}% CI mean [{ci_low:.3f}, {ci_high:.3f}]"
        )

        if print_pi:
            # Prediction interval prossima settimana
            ax.axhline(
                pi_low, linestyle=":", linewidth=1.8,
                label=f"{int(ci*100)}% PI next week low = {pi_low:.3f}"
            )
            ax.axhline(
                pi_high, linestyle=":", linewidth=1.8,
                label=f"{int(ci*100)}% PI next week high = {pi_high:.3f}"
            )
            ax.fill_between(
                x,
                pi_low,
                pi_high,
                alpha=0.07,
                label=f"{int(ci*100)}% PI next week"
            )

    ax.set_title(f"Weekly aggregated return - strategies: {strategy_label}", fontsize=14, pad=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Aggregated return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()

    stats = {
        "strategy": strategies if len(strategies) > 1 else strategies[0],
        "n_weeks": n,
        "mean_weekly_agg_return": mean,
        "std_weekly_agg_return": std,
        "sem": sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pi_low": pi_low,
        "pi_high": pi_high,
    }

    return df_aggregate, stats

