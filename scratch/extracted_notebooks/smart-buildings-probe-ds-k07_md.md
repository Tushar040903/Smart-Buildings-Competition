<!-- ===== MD CELL 0 ===== -->
# Smart Buildings: per-zone PI control and AHU cooling hold

**Task.** Control the HVAC of an unseen building every 5 minutes for 30 days
(8,640 steps). **Metric:** mean over scenarios of `abs(mean(reward))`, lower is
better, where reward weighs comfort 0.6, energy cost 0.2 and carbon 0.2.

**Per-zone PI control (main gain of this version).** Per-step logs showed
occupied zones swinging 2–4 K every 5–15 minutes: the organizer zone loop is
proportional only and opens the damper fully at 2 K of error, so a zone slightly
out of band gets a burst of air and overshoots through the whole band. Zones
averaged inside the 21–24 °C band yet were outside it 57–61% of work hours, and
comfort was 60% of the loss. A softer proportional loop stops the oscillation
but leaves zones sitting outside the band (no integral action).

Here every zone runs its own PI loop on (temperature − band centre) from one
hour before occupancy to the end of the work day (Kp 0.07, Ki
0.0035 per 5-minute step, conditional integration against wind-up).
One signal drives the damper (|u|) and the reheat valve (−u); AHUs run through
those hours and zone reheat demand raises the hot-water temperature. Night
logic is unchanged. On full training episodes the comfort loss nearly
disappears (0.0006 mean on five scenarios with Kp 0.1, Ki 0.005).

**Cooling-run hold.** The organizer PID controller stops an air handler (AHU)
the moment its zones are back inside the comfort band, and restarts it at the
next excursion: every AHU switched on or off 50–107 times a day. Keeping a unit
that was started **for cooling** running for at least 288 steps
(24 h) at its last commands removes most of the loss in
hot climates and leaves heating-dominated climates unchanged.

**Night cooling ceiling.** Outside work hours the cooling side of the band is
capped at the daytime comfort ceiling instead of the eco ceiling (1 K higher),
so hot buildings do not drift up overnight and start the day overheated; the
heating setback is kept. This idea comes from the public notebook
[Neurips 2026 Smart Buildings Smartpolicy](https://www.kaggle.com/code/royalacecat/neurips-2026-smart-buildings-smartpolicy)
by E-Max AI.

| Version | Controller | Public LB |
| --- | --- | --- |
| 3 | First baseline (mean-zone PID, sensor comfort bands, decoded weather) | 0.06439 |
| 4 | + cooling-run hold, 6 steps | 0.05882 |
| 5 | + cooling-run hold, 288 steps + night cooling ceiling | 0.05195 |
| this | v5 + per-zone PI to the band centre (Kp 0.07, Ki 0.0035) | see leaderboard |

Sources: [competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings),
[starter kit](https://github.com/smart-buildings-2026/starter-kit). The base
controller is the organizers' `PIDWaterResetPolicy` (Apache-2.0). The EDA reads
only published training scenarios and is skipped during the hidden rerun.


<!-- ===== MD CELL 2 ===== -->
## Exploratory data analysis

30 training scenarios: 6 buildings × 5 climates (Atlanta, Denver, Fairbanks,
Phoenix, Singapore); 30–85 zones and 3–9 AHUs per building. Below: building
sizes, observation and action dimensions, initial sensor gaps, and the actual
30-day weather windows the simulator uses. A zero continuous action is the
middle of the range (e.g. a half-open damper), not "off". AHUs can only cool;
heat comes from zone hot-water reheat, which needs airflow to deliver it.


<!-- ===== MD CELL 5 ===== -->
## How the improvement was found

All numbers are full 30-day episodes run on Kaggle CPU with the official
simulator. Lower is better.

1. **The comparison is noisy.** Two runs that differ by a hundredth of a pascal
   in one command track each other for days, then jump ~10 K apart when a
   switching decision flips. A *null group* (the baseline plus three runs with
   such negligible nudges) moves the episode score by 1–6%. A candidate is
   judged by `z = (score − null mean) / null sd`; |z| < 2 is noise.
2. **Rejected:** flow-preserving fan pressure reset (inside the noise),
   colder supply air (−74% on Singapore), delayed AHU starts (worse comfort).
3. **Minimum run time** helped hot climates but hurt Fairbanks, where it
   lengthened heating bursts. Holding **only cooling runs** kept the hot-climate
   gain and removed the cold-climate loss, and the gain grew with hold length.
4. **Zone oscillation.** After the hold, comfort was still 60% of the loss:
   zones swung 2–4 K every few steps. Softer proportional gains removed the
   swings but left a steady offset; per-zone PI to the band centre fixed both.
   On five full scenarios (two from a building not used for tuning) the PI
   controller scored 0.0204 on average against 0.0483 for the hold alone.

Pilot screening (building_train_8, three climates). `z_by_scenario` order:
Fairbanks / Phoenix / Singapore; negative is better.


<!-- ===== MD CELL 7 ===== -->
## Check on all 30 training scenarios

The hold length was tuned on building_train_8, so the other five buildings
(25 scenarios) are the out-of-sample check. Below: per-scenario scores of the
baseline (`transfer`) and the hold (`min_on_cool_288`), and the mean by climate.
These are training scenarios; they do not estimate the hidden leaderboard.


<!-- ===== MD CELL 9 ===== -->
## Controller

`BaselinePolicy` is the first submitted controller: the organizers' PID with
per-zone comfort bands read from sensors and outside temperature decoded from
its normalized observation before the economizer decision. `CoolingHoldPolicy`
adds the cooling-run hold and, when enabled, the night cooling ceiling. Actuator indices come from each building's metadata,
so the controller adapts to any number of zones and AHUs.


<!-- ===== MD CELL 13 ===== -->
## Inference

The official server runs the public example during a normal commit and the
hidden scenarios during a submission rerun, writing `submission.parquet`.
Internet is off.
