import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from datetime import datetime


def _get_data():
    df = pd.read_csv("bot_messages.csv")
    recap_mask = ["🏁 • 🆚" in x and "@" not in x and "🤖" not in x for x in df["message"].fillna("")]
    df_text = df[recap_mask]['message'][:: - 1]
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
        goal_match = re.search(r'G\((\d+)-(\d+)\)', block)

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
        try:
            goal_home = goal_match.group(1)
            goal_away = goal_match.group(2)
        except:
            goal_home = None
            goal_away = None

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
                "goal_home": goal_home,
                "goal_away": goal_away,
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


def get_match_df(filter_league: bool = False):
    """
    filter_league (bool): Filter out no more used leagues
    """
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
    df_final["goal_home"] = df_final["goal_home"].astype(int, errors="ignore")
    df_final["goal_away"] = df_final["goal_away"].astype(int, errors="ignore")
    
    # Dropping DNB 1 and DNB 2 tied games
    other_strategies_mask = [not x for x in df_final['strategy'].isin(["DNB 1", "DNB 2"])]
    dnb_mask = (df_final['strategy'].isin(["DNB 1", "DNB 2"]) & (df_final['goal_home'] != df_final['goal_away']))
    df_final = df_final[other_strategies_mask | dnb_mask]


    df_final = df_final.sort_values("date") 

    monday_start = df_final["date"].min() - pd.to_timedelta(df_final["date"].min().weekday(), unit="D")
    df_final["week"] = ((df_final["date"] - monday_start).dt.days // 7) + 1

    if filter_league:
        # Filter out no more used leagues
        last_week = max(df_final["week"])
        available_leagues = df_final[df_final["week"] == last_week]["league"].unique()
        df_final = df_final[df_final["league"].isin(available_leagues)]

    df_final["strategy"] = [_normalize_strategies(x) for x in df_final["strategy"]]
    return df_final


def analyze_strategies(
    df: pd.DataFrame,
    strategies=None,
    leagues=None,
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

    if strategies:
        # filtro strategie
        df_strategy = df[df["strategy"].isin(strategies)].copy()
    else:
        df_strategy = df.copy()

    if leagues:
        df_strategy = df_strategy[df_strategy["league"].isin(leagues)]


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
    if strategies:
        strategy_label = strategies[0] if len(strategies) == 1 else " + ".join(strategies)
    else:
        strategy_label = "Any strategy"
    
    if leagues:
        league_label = leagues[0] if len(leagues) == 1 else " + ".join(leagues)
    else:
        league_label = "Any league"

    # plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    # ymin, ymax = ax.get_ylim()
    # ax.set_yticks(np.arange(ymin, ymax + 2.5, 2.5))

    x = df_aggregate.index
    y = df_aggregate.values

    ax.plot(
        x, y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=f"Weekly return"
    )

    # area sopra/sotto zero
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.12, interpolate=True)
    ax.fill_between(x, y, 0, where=(y < 0), alpha=0.12, interpolate=True)

    # linee statistiche
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axhline(
        mean, linestyle="-", linewidth=2,
        label=f"Weekly Mean = {mean:.3f}"
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

    ax.set_title(f"Weekly return: {strategy_label}", fontsize=10, pad=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Aggregated return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()

    if not strategies:
        strategies = ["Any"]
    
    if not leagues:
        leagues = ["Any"]

    stats = {
        "strategy": strategies if len(strategies) > 1 else strategies[0],
        "leagues": leagues if len(leagues) > 1 else leagues[0],
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


def get_best_leagues(df, strategies: list[str] = None):
    # filtro strategie

    if strategies:
        df_strategies = df[df["strategy"].isin(strategies)].copy()
    else:
        df_strategies = df.copy()

    # aggregazione per campionato
    league_stats = (
        df_strategies
        .groupby("league")
        .agg(
            total_return=("return", "sum"),
            n_matches=("return", "size"),
            avg_return=("return", "mean")
        )
        .sort_values("total_return", ascending=False)
    )

    # top campionati positivi
    top_leagues = (
        league_stats[league_stats["total_return"] > 0]
        .sort_values("total_return", ascending=True)
    )

    # worst campionati negativi
    worst_leagues = (
        league_stats[league_stats["total_return"] < 0]
        .sort_values("total_return", ascending=True)
    )

    _, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Top leagues
    axes[0].barh(top_leagues.index, top_leagues["total_return"])
    axes[0].axvline(0, linestyle="--", linewidth=1)
    axes[0].set_title(f"Top {len(top_leagues)} campionati per return totale")
    axes[0].set_xlabel("Return totale")
    axes[0].set_ylabel("League")

    for i, (league, row) in enumerate(top_leagues.iterrows()):
        axes[0].text(
            row["total_return"],
            i,
            f"  n={int(row['n_matches'])} | avg={row['avg_return']:.2f}",
            va="center"
        )

    # Worst leagues
    axes[1].barh(worst_leagues.index, worst_leagues["total_return"])
    axes[1].axvline(0, linestyle="--", linewidth=1)
    axes[1].set_title(f"Worst {len(worst_leagues)} campionati per return totale")
    axes[1].set_xlabel("Return totale")
    axes[1].set_ylabel("League")

    for i, (_, row) in enumerate(worst_leagues.iterrows()):
        axes[1].text(
            row["total_return"],
            i,
            f"  n={int(row['n_matches'])} | avg={row['avg_return']:.2f}",
            va="center"
        )

    plt.tight_layout()
    plt.show()
    return top_leagues.sort_values(by="total_return", ascending=False)


def get_best_strategies(df, leagues: list[str] = None, n_weeks: int = None):
    """
    n_weeks (int): no. of rolling weeks to evaluate
    leagues (list): leagues to consider. If None any league is considered
    """
    pd.set_option('display.float_format', '{:.2f}'.format)

    last_week = max(df["week"])
    df = df.sort_values(by="date", ascending=False)

    if not n_weeks:
        n_weeks = last_week

    week_df = df[df["week"] > last_week - n_weeks]

    if leagues:
        week_df = week_df[week_df["league"].isin(leagues)]
    strategy_dict = dict()

    # Removing week 7 considering it an outlier
    # week_df = week_df[week_df["week"] != 7]
    available_strategies = week_df[week_df["week"] == last_week]["strategy"].unique()

    for strategy in available_strategies:
        gain = week_df[week_df['strategy'] == strategy].groupby("week")["return"].sum().mean()
        std = week_df[week_df['strategy'] == strategy].groupby("week")["return"].sum().std()
        counts = week_df[week_df['strategy'] == strategy].groupby("week").size().to_list()

        while len(counts) < n_weeks:
            counts += [0]
        
        mean_counts = np.mean(counts)
        
        total_counts = (np.sum(counts))
        accuracy = week_df[week_df['strategy'] == strategy].groupby("week")["result"].mean().mean()
        strategy_dict = strategy_dict | {strategy: [gain, std, mean_counts, total_counts, accuracy]}

    strategy_df = pd.DataFrame(strategy_dict, index=["weekly_mean_return", "weekly_std_return", "weekly_counts", "total_counts", "accuracy"]).T.sort_values(by='weekly_mean_return', ascending=False)
    strategy_df["total_counts"] = strategy_df["total_counts"].astype(int)
    strategy_df = strategy_df.dropna()
    return strategy_df 



async def main(output_csv):
    client = TelegramClient(session_name, api_id, api_hash)
    await client.start()  # al primo avvio chiede numero, codice e forse password 2FA

    entity = await client.get_entity(group_input)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "date",
            "sender_id",
            "sender_name",
            "message",
            "reply_to_msg_id",
            "views",
            "forwards"
        ])

        # reverse=True => dal più vecchio al più recente
        async for msg in client.iter_messages(entity, reverse=True):
            sender = await msg.get_sender() if msg.sender_id else None

            sender_name = ""
            if sender:
                if getattr(sender, "username", None):
                    sender_name = sender.username
                else:
                    first = getattr(sender, "first_name", "") or ""
                    last = getattr(sender, "last_name", "") or ""
                    sender_name = (first + " " + last).strip()

            writer.writerow([
                msg.id,
                msg.date.isoformat() if msg.date else "",
                msg.sender_id,
                sender_name,
                msg.message or "",
                msg.reply_to_msg_id,
                getattr(msg, "views", None),
                getattr(msg, "forwards", None),
            ])

    await client.disconnect()
    print(f"Esportazione completata: {output_csv}")

if __name__ == "__main__":
    import asyncio
    import csv
    from telethon import TelegramClient

    # Inserisci qui le tue credenziali
    api_id = 29726995
    api_hash = "d7b8f07290d475df82300e97bf9e898d"

    # Nome file sessione locale (Telethon salverà qui la sessione)
    session_name = "telegram_session"

    # Username/link/titolo del gruppo
    # Esempi:
    # group_input = "https://t.me/nomegruppo"
    # group_input = "nomegruppo"
    # group_input = "Nome Gruppo"
    group_input = "Bet Master Group"

    output_csv = "bot_messages.csv"
    asyncio.run(main(output_csv))