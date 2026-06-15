# Pitcher Similarity Finder

A Streamlit app that compares one target pitcher against comparison CSVs by pitch type and overall arsenal similarity.

## What v14 adds

- Top Comp Score now fills automatically from the overall leaderboard.
- Movement Plot now uses `HorzBrk` / `IndVertBrk` and also accepts `HB` / `IVB` aliases.
- Report header uses the pitcher name and includes detected org/team/level/handedness when available.
- Pitch usage is added from the `P` column when available.
- PDF and PowerPoint exports keep excluding pitch types with no matches.
- PDF is designed as a compact one-page baseball operations style report.

## Run on Mac

Open Terminal inside the app folder and run:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

If PDF export fails because ReportLab is missing:

```bash
python3 -m pip install reportlab matplotlib python-pptx
```

## CSV notes

Target CSV should include pitch characteristics such as:

- `playerId`
- `playerFullName` if available
- `Pitch Type`
- `P`
- `Vel`
- `VelMax`
- `Spin`
- `Extension`
- `IndVertBrk`
- `HorzBrk`
- `Rel. Height`
- `RSd`

Comparison CSV pitch type can be inferred from the filename, for example:

- `Fastball Pitch Characteristics.csv`
- `Slider Pitch Characteristics.csv`
- `Change Pitch Characteristics.csv`
