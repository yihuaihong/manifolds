"""Fetch the data files of the concept domains that are not written out in tools/manifold_domains.py, from their
original sources, into $MANIFOLD_DATA (default <repo>/data/atlas), and check each file against the copy the paper used
(SHA-256). Nothing is re-hosted: every file comes from its source.

  file                          source                                                        terms
  gurnee_us_place.csv           Gurnee & Tegmark (2024), github.com/wesg52/world-models,      MIT
  gurnee_historical_figure.csv    data/entity_datasets/{us_place,historical_figure}.csv
  warriner.csv                  Warriner, Kuperman & Brysbaert (2013), Behavior Research       those of the article's
                                  Methods 45:1191-1207, supplementary BRM-emot-submit.csv       supplementary material
                                  (line endings CRLF -> LF)
  stars.csv                     Hipparcos main catalogue (ESA 1997; CDS I/239, hip_main.dat)   CDS
                                  x the IAU proper names in skyfield.named_stars (skyfield 1.55)
  landmarks.csv                 English Wikipedia API, primary coordinates of 76 articles      CC BY-SA

Without a file, the domain that needs it is skipped (stars, emotions, landmarks, US cities, people; see
manifold_domains.domains()). Numbers, the calendar domains, the six-prior domains (geonamescache for cities and
countries; dimensions, torus and grid are generated) and the other built-in domains need nothing from here.

    pip install skyfield==1.55        # star names; any version with the same named_star_dict gives the same file
    python tools/atlas_fetch_data.py                      # all five files
    python tools/atlas_fetch_data.py --only warriner      # one file
    python tools/atlas_fetch_data.py --hip hip_main.dat   # use a local copy of the Hipparcos catalogue

Wikipedia coordinates can be edited after 2026-10; if landmarks.csv no longer matches, the script says so.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile

from manifold_domains import DATA

UA = {"User-Agent": "manifold-atlas/0.2"}
SHA256 = {   # the copies the paper used
    "gurnee_us_place.csv": "95af51d5b80ba1a0085c431583294f4917b95628869130d65762a7e8bf6eff6e",
    "gurnee_historical_figure.csv": "0f4ccff016be3e92b6e03a2bcc59c4f6b376b01cc745b244ba355d1c7682139f",
    "warriner.csv": "78ac8107c78e116bb96538fae4faa47281a155f5f8fe39f30bbc6ea3db05b446",
    "stars.csv": "b8251b7b2144383b34395754b8b242d376e12aedc34ed9bf6b8ba333f8f31cff",
    "landmarks.csv": "3ff0aa75243ec1440b0c549a03d05a57a989eadb91df1e3ee53fc1112ac44b3c",
}
GURNEE = "https://raw.githubusercontent.com/wesg52/world-models/main/data/entity_datasets/{}.csv"
WARRINER = ("https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/"
            "13428_2012_314_MOESM1_ESM.zip")
HIPPARCOS = ["https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat",
             "https://cdsarc.u-strasbg.fr/ftp/cats/I/239/hip_main.dat",
             "http://cdsarc.u-strasbg.fr/ftp/cats/I/239/hip_main.dat.gz"]
HIP_SHA256 = "58ceabb104d647160d9437ce6e513a02a036bb4ad9f8879a5a22fd52943616e0"

LANDMARKS = [  # (prompt name, Wikipedia title)
    ("Eiffel Tower", "Eiffel Tower"), ("Colosseum", "Colosseum"), ("Big Ben", "Big Ben"), ("Stonehenge", "Stonehenge"),
    ("Parthenon", "Parthenon"), ("Leaning Tower of Pisa", "Leaning Tower of Pisa"), ("Sagrada Familia", "Sagrada Família"),
    ("Brandenburg Gate", "Brandenburg Gate"), ("Neuschwanstein Castle", "Neuschwanstein Castle"), ("Louvre", "Louvre"),
    ("Buckingham Palace", "Buckingham Palace"), ("Tower Bridge", "Tower Bridge"), ("Edinburgh Castle", "Edinburgh Castle"),
    ("Alhambra", "Alhambra"), ("Hagia Sophia", "Hagia Sophia"), ("Saint Basil's Cathedral", "Saint Basil's Cathedral"),
    ("Matterhorn", "Matterhorn"), ("Mont Blanc", "Mont Blanc"), ("Mount Vesuvius", "Mount Vesuvius"), ("Atomium", "Atomium"),
    ("Charles Bridge", "Charles Bridge"), ("Mont-Saint-Michel", "Mont-Saint-Michel"), ("Trevi Fountain", "Trevi Fountain"),
    ("Great Pyramid of Giza", "Great Pyramid of Giza"), ("Abu Simbel", "Abu Simbel"), ("Victoria Falls", "Victoria Falls"),
    ("Table Mountain", "Table Mountain"), ("Mount Kilimanjaro", "Mount Kilimanjaro"), ("Mount Kenya", "Mount Kenya"),
    ("Taj Mahal", "Taj Mahal"), ("Great Wall of China", "Great Wall of China"), ("Forbidden City", "Forbidden City"),
    ("Mount Everest", "Mount Everest"), ("Mount Fuji", "Mount Fuji"), ("Angkor Wat", "Angkor Wat"),
    ("Petronas Towers", "Petronas Towers"), ("Burj Khalifa", "Burj Khalifa"), ("Petra", "Petra"),
    ("Potala Palace", "Potala Palace"), ("Terracotta Army", "Terracotta Army"), ("Golden Temple", "Golden Temple"),
    ("Western Wall", "Western Wall"), ("Shwedagon Pagoda", "Shwedagon Pagoda"), ("Tokyo Tower", "Tokyo Tower"),
    ("Borobudur", "Borobudur"), ("Marina Bay Sands", "Marina Bay Sands"), ("Kaaba", "Kaaba"), ("Persepolis", "Persepolis"),
    ("Mount Ararat", "Mount Ararat"), ("Ha Long Bay", "Ha Long Bay"), ("Taipei 101", "Taipei 101"),
    ("Statue of Liberty", "Statue of Liberty"), ("Golden Gate Bridge", "Golden Gate Bridge"), ("Grand Canyon", "Grand Canyon"),
    ("Mount Rushmore", "Mount Rushmore"), ("Empire State Building", "Empire State Building"), ("White House", "White House"),
    ("Niagara Falls", "Niagara Falls"), ("Space Needle", "Space Needle"), ("CN Tower", "CN Tower"),
    ("Chichen Itza", "Chichen Itza"), ("Teotihuacan", "Teotihuacan"), ("Machu Picchu", "Machu Picchu"),
    ("Christ the Redeemer", "Christ the Redeemer (statue)"), ("Iguazu Falls", "Iguazu Falls"), ("Aconcagua", "Aconcagua"),
    ("Panama Canal", "Panama Canal"), ("Hollywood Sign", "Hollywood Sign"), ("Gateway Arch", "Gateway Arch"),
    ("Hoover Dam", "Hoover Dam"), ("Denali", "Denali"), ("Lake Titicaca", "Lake Titicaca"),
    ("Sydney Opera House", "Sydney Opera House"), ("Uluru", "Uluru"), ("Sydney Harbour Bridge", "Sydney Harbour Bridge"),
    ("Aoraki", "Aoraki / Mount Cook"), ("Milford Sound", "Milford Sound"), ("Mount Erebus", "Mount Erebus"),
]


def coords(titles):
    q = urllib.parse.urlencode({"action": "query", "prop": "coordinates", "titles": "|".join(titles), "format": "json", "colimit": "max",
                                "redirects": 1})
    req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{q}", headers={"User-Agent": "manifold-atlas/0.1"})
    d = json.load(urllib.request.urlopen(req, timeout=30))["query"]
    back = {}
    for m in d.get("normalized", []) + d.get("redirects", []):
        back[m["to"]] = back.get(m["from"], m["from"])
    out = {}
    for p in d["pages"].values():
        if "coordinates" in p:
            c = p["coordinates"][0]
            out[back.get(p["title"], p["title"])] = (c["lat"], c["lon"])
    return out



def get(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def gurnee(name):
    return get(GURNEE.format(name))


def warriner():
    z = zipfile.ZipFile(io.BytesIO(get(WARRINER)))
    return z.read("BRM-emot-submit.csv").replace(b"\r\n", b"\n")


def hipparcos(local=None):
    path = DATA / "hip_main.dat"
    if local:
        raw = open(local, "rb").read()
    elif path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == HIP_SHA256:
        raw = path.read_bytes()
    else:
        raw, errors = None, []
        for url in HIPPARCOS:
            try:
                raw = get(url)
                raw = gzip.decompress(raw) if url.endswith(".gz") else raw
                break
            except OSError as e:
                errors.append(f"{url}: {e}")
        if raw is None:
            raise SystemExit("Hipparcos catalogue not reachable; download hip_main.dat from CDS I/239 and pass "
                             "--hip PATH\n  " + "\n  ".join(errors))
        path.write_bytes(raw)
    if hashlib.sha256(raw).hexdigest() != HIP_SHA256:
        print("warning: hip_main.dat differs from the copy the paper used")
    return raw


def stars(local=None):
    from skyfield.named_stars import named_star_dict   # pip install skyfield
    hip = {}
    for line in hipparcos(local).decode("ascii").splitlines():
        f = line.split("|")
        if len(f) > 9 and f[8].strip() and f[9].strip():
            hip[int(f[1])] = (f[8].strip(), f[9].strip(), f[5].strip())
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["hip", "name", "ra_degrees", "dec_degrees", "magnitude"])
    for name, h in named_star_dict.items():
        if h in hip:
            w.writerow([h, name, repr(float(hip[h][0])), repr(float(hip[h][1])), repr(float(hip[h][2]))])
    return out.getvalue().encode()


def landmarks():
    got = {}
    for i in range(0, len(LANDMARKS), 40):
        got.update(coords([t for _, t in LANDMARKS[i:i + 40]]))
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n")
    w.writerow(["name", "title", "lat", "lon"])
    w.writerows((n, t, *got[t]) for n, t in LANDMARKS if t in got)
    return out.getvalue().encode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=["gurnee", "warriner", "stars", "landmarks"])
    ap.add_argument("--hip", help="local copy of hip_main.dat (CDS I/239)")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = {"gurnee": [("gurnee_us_place.csv", lambda: gurnee("us_place")),
                       ("gurnee_historical_figure.csv", lambda: gurnee("historical_figure"))],
            "warriner": [("warriner.csv", warriner)],
            "stars": [("stars.csv", lambda: stars(a.hip))],
            "landmarks": [("landmarks.csv", landmarks)]}
    for key in a.only or jobs:
        for fname, make in jobs[key]:
            data = make()
            (DATA / fname).write_bytes(data)
            same = hashlib.sha256(data).hexdigest() == SHA256[fname]
            print(f"{DATA / fname}: {len(data):,} bytes, "
                  + ("matches the paper's copy" if same else "DIFFERS from the paper's copy (the source changed)"))


if __name__ == "__main__":
    main()
