<!-- ===== MD CELL 1 ===== -->
## Exploratory data analysis

Training set: 6 buildings x 5 climates = 30 scenarios (30-85 zones, 3-9 AHUs per building).
The cell below reports building size, observation and action dimensions, initial NaNs, material
properties, a floor-plan grid, and the 30-day outside-temperature range per climate.
Key facts for control design: a zero continuous action is mid-range (half-open damper), not off;
AHUs can only cool; heating comes from zone hot-water reheat, which needs airflow.

<!-- ===== MD CELL 4 ===== -->
## Controllers

Actuator indices come from each building's metadata, so the policy adapts to any number of zones and AHUs.

<!-- ===== MD CELL 8 ===== -->
## Inference

During a normal commit the server runs the public example scenario; during the submission rerun it runs the hidden scenarios and writes `submission.parquet`. Internet is off.