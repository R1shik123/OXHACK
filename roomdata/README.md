# Room data checklist

Drop photos and notes here (this folder stays on the laptop). Only include people who agreed to be photographed.

1. Photos: each slow-scanning wall (straight on + at an angle), the chairs and tables, the room from each corner.
2. Measurements (tape), also typed into `measurements.json` for the replay tuner: room length / width / ceiling height, one door width, one table top height + size,
   one chair seat height, marker height, marker -> two landmarks (e.g. door frame, opposite wall).
3. A floor-plan sketch: rooms, doors, exits, furniture.
4. Second iPhone: model + iOS version.
5. Operations: team size, time budget per person (minutes), walking speed assumption, buddy distance, which doors are exits.
6. Room capture: commander page -> "Room capture (90 s)" while both phones walk the room slowly, sweeping walls and furniture.
7. Tuning: `cd server && ../.venv/bin/python -m sightline.replay --measure ../roomdata/measurements.json`, then delete the capture on the commander page.
