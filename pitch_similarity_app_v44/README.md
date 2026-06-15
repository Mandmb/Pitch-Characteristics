# Pitch Similarity App v43

Run:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

v43 updates automatic headshots by scraping normal `<img>` and `srcset` URLs from MiLB player pages and trying the exact encoded `mlbstatic.com` image route used on MiLB player pages.


V44: Headshot auto-fetch now uses requests + urllib fallback, shows diagnostics, and supports an optional direct headshot URL.
