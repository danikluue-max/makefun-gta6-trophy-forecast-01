# makefun-gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/danikluue-max/makefun-gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/danikluue-max/makefun-gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles a trophy, scored by a pinball loss on the logit scale. The program works in counts, not rates: Exophase's page only moves when it refreshes, and a refresh adds newly tracked players (who have earned almost nothing yet) while recording the progress of the players already tracked. So it reads every refresh step in the case's history, splits each step's new earners into old-player progress and new-cohort earns with a ridge-regularised two-parameter fit, fits how the tracked-player count grows, and rolls both forward to the target. When the history is flat it falls back on a launch-night prior (one refresh due, players multiply, the new cohort holds the one trophy everybody gets plus a small share of every other trophy's rate). A log-time trend line is blended in as a hedge, and the spread comes from how much the steps and the trend disagree.

## Run it

```sh
python3 -P -s /bot/main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=0zdpscn2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
