"""Concept domains for the manifold atlas (tools/manifold_atlas.py): items, prompts and candidate target geometries.

Every domain gives items (strings), a ground-truth coordinate per item and two prompt templates whose LAST token is
the item's last token ("neutral" = "Q: {item}", as in the number experiments; "cued" names the domain first). The
targets are feature matrices (items x features); the atlas measures how well each one is linearly decodable from
the states and against a permutation null. Candidate targets per kind (§5 of report/manifold_survey.md: for a
domain with symmetry group G, the natural target is G's irreps):
  cyclic  Z_n          circle (k = 1), fourier (all k), line (wrong topology, control)
  helix   numbers      helix (periods 2/5/10/100 + linear), line
  linear  ordered      line (index), curve (orthogonal polynomials up to degree 3)
  sphere  sky, cities  sphere (unit vector), harmonics (real spherical harmonics, degree <= 2), plane (lat, long)
  plane   emotions     plane (valence, arousal from Warriner et al. 2013), circumplex (angle on the VA plane)
Data sources: matplotlib CSS4 colours (hue from RGB), Hipparcos via skyfield (stars with IAU proper names),
geonamescache (cities), Warriner et al. 2013 norms (emotions); the rest are fixed sequences written out here.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# Atlas CSVs (stars, warriner, landmarks, gurnee_*): $MANIFOLD_DATA, else <repo>/data/atlas. A domain whose file is
# missing is skipped by domains() (release patch 2026-11-20; the tables of the domains that are built are unchanged).
DATA = Path(os.environ.get("MANIFOLD_DATA", Path(__file__).resolve().parent.parent / "data" / "atlas"))


def _have(fname, domain):
    if (DATA / fname).exists():
        return True
    print(f"manifold_domains: {DATA / fname} not found: domain {domain!r} skipped", file=sys.stderr)
    return False

DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega").split()
NUMWORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen twenty").split() + "thirty forty fifty sixty seventy eighty ninety".split()
NUMWORD_VALUES = list(range(1, 21)) + [30, 40, 50, 60, 70, 80, 90]
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
PLANETS = "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split()
ELEMENTS = ("hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium aluminum "
            "silicon phosphorus sulfur chlorine argon potassium calcium scandium titanium vanadium chromium manganese "
            "iron cobalt nickel copper zinc gallium germanium arsenic selenium bromine krypton").split()
# US presidents in office order ("-" = left out: the surnames shared by two presidents, Adams, Harrison, Johnson,
# Cleveland, Roosevelt and Bush, are dropped so that every name has one position)
PRESIDENTS = ("Washington - Jefferson Madison Monroe - Jackson Van_Buren - Tyler Polk Taylor Fillmore Pierce "
              "Buchanan Lincoln - Grant Hayes Garfield Arthur - - - McKinley - Taft Wilson Harding "
              "Coolidge Hoover - Truman Eisenhower Kennedy - Nixon Ford Carter Reagan - Clinton - Obama Trump Biden").split()
# pitch classes (chromatic index) and the 12 major keys as conventionally spelled
PITCH = [("C", 0), ("C#", 1), ("D", 2), ("D#", 3), ("E", 4), ("F", 5), ("F#", 6), ("G", 7), ("G#", 8), ("A", 9),
         ("A#", 10), ("B", 11)]
KEYS = [("C", 0), ("Db", 1), ("D", 2), ("Eb", 3), ("E", 4), ("F", 5), ("F#", 6), ("G", 7), ("Ab", 8), ("A", 9),
        ("Bb", 10), ("B", 11)]
COLOURS = ("red orange yellow gold chartreuse lime green olive teal turquoise cyan aqua blue navy indigo violet purple "
           "magenta orchid plum pink crimson coral salmon tomato maroon").split()
EMOTIONS = ("happy sad angry afraid calm excited bored relaxed anxious content depressed elated furious gloomy joyful "
            "lonely nervous peaceful proud scared serene tense tired upset cheerful delighted enthusiastic frustrated "
            "grateful guilty hopeful hostile irritated jealous miserable optimistic pleased sleepy surprised terrified "
            "thrilled worried ashamed disgusted ecstatic embarrassed energetic inspired lively panicked satisfied "
            "tranquil triumphant uneasy weary alert amused annoyed").split()

# ---- atlas v2 (2026-10-13): general-knowledge and tokenizer-independent domains (report/manifold_survey.md §9)
# famous people; birth years are looked up in Gurnee & Tegmark's historical_figure table (names not found are dropped)
PEOPLE = ["Confucius", "Sun Tzu", "Pericles", "Socrates", "Plato", "Aristotle", "Alexander the Great", "Hannibal", "Spartacus",
          "Cleopatra", "Attila", "Justinian I", "Charlemagne", "Richard I of England", "Genghis Khan", "Kublai Khan",
          "Thomas Aquinas", "Marco Polo", "Dante Alighieri", "Petrarch", "Ibn Battuta", "Giovanni Boccaccio", "Geoffrey Chaucer",
          "Johannes Gutenberg", "Joan of Arc", "Christopher Columbus", "Leonardo da Vinci", "Vasco da Gama",
          "Niccolò Machiavelli", "Nicolaus Copernicus", "Michelangelo", "Ferdinand Magellan", "Martin Luther", "Hernán Cortés",
          "Henry VIII of England", "Suleiman the Magnificent", "Akbar", "Galileo Galilei", "William Shakespeare",
          "Johannes Kepler", "René Descartes", "Rembrandt", "Blaise Pascal", "Isaac Newton", "Peter the Great",
          "Johann Sebastian Bach", "Benjamin Franklin", "Adam Smith", "Immanuel Kant", "Joseph Haydn", "George Washington",
          "Thomas Jefferson", "Johann Wolfgang von Goethe", "Wolfgang Amadeus Mozart", "Napoleon", "Ludwig van Beethoven",
          "Jane Austen", "Simón Bolívar", "Abraham Lincoln", "Charles Darwin", "Charles Dickens", "Karl Marx", "Queen Victoria",
          "Florence Nightingale", "Leo Tolstoy", "Mark Twain", "Claude Monet", "Vincent van Gogh", "Oscar Wilde",
          "Sigmund Freud", "Nikola Tesla", "Theodore Roosevelt", "Marie Curie", "Mahatma Gandhi", "Vladimir Lenin",
          "Winston Churchill", "Joseph Stalin", "Albert Einstein", "Pablo Picasso", "Virginia Woolf", "Charlie Chaplin",
          "Mao Zedong", "Ernest Hemingway", "Walt Disney", "George Orwell", "Frida Kahlo", "Alan Turing", "John F. Kennedy",
          "Nelson Mandela", "Marilyn Monroe", "Andy Warhol", "Che Guevara", "Martin Luther King Jr.", "Audrey Hepburn",
          "Neil Armstrong", "Elvis Presley", "John Lennon", "Bruce Lee", "Stephen Hawking", "Jimi Hendrix", "Muhammad Ali",
          "Bob Marley", "Freddie Mercury", "Steve Jobs", "Michael Jackson", "Kurt Cobain"]
GEO_PERIODS = ("Cambrian Ordovician Silurian Devonian Carboniferous Permian Triassic Jurassic Cretaceous Paleogene Neogene "
               "Quaternary").split()
# (animal, class, order, typical adult mass in kg). Orders are the traditional ones (whales as Cetacea); masses are
# order-of-magnitude values (the targets use log10 mass; the test only compares animals >= 10x apart)
ANIMALS = [
    ("mouse", "mammal", "rodent", 0.02), ("rat", "mammal", "rodent", 0.3), ("squirrel", "mammal", "rodent", 0.5),
    ("beaver", "mammal", "rodent", 20), ("rabbit", "mammal", "lagomorph", 2), ("bat", "mammal", "bat", 0.02),
    ("cat", "mammal", "carnivore", 4), ("fox", "mammal", "carnivore", 6), ("dog", "mammal", "carnivore", 25),
    ("wolf", "mammal", "carnivore", 40), ("lion", "mammal", "carnivore", 190), ("tiger", "mammal", "carnivore", 200),
    ("bear", "mammal", "carnivore", 300), ("horse", "mammal", "odd-toed", 500), ("zebra", "mammal", "odd-toed", 350),
    ("rhinoceros", "mammal", "odd-toed", 2000), ("sheep", "mammal", "even-toed", 70), ("goat", "mammal", "even-toed", 50),
    ("pig", "mammal", "even-toed", 150), ("deer", "mammal", "even-toed", 100), ("cow", "mammal", "even-toed", 700),
    ("camel", "mammal", "even-toed", 500), ("giraffe", "mammal", "even-toed", 1000),
    ("hippopotamus", "mammal", "even-toed", 1500), ("dolphin", "mammal", "cetacean", 150),
    ("whale", "mammal", "cetacean", 40000), ("elephant", "mammal", "proboscidean", 5000),
    ("monkey", "mammal", "primate", 8), ("chimpanzee", "mammal", "primate", 50), ("gorilla", "mammal", "primate", 150),
    ("kangaroo", "mammal", "marsupial", 60), ("sparrow", "bird", "songbird", 0.03), ("crow", "bird", "songbird", 0.5),
    ("pigeon", "bird", "pigeon", 0.35), ("parrot", "bird", "parrot", 0.4), ("owl", "bird", "owl", 1.5),
    ("hawk", "bird", "raptor", 1), ("eagle", "bird", "raptor", 5), ("chicken", "bird", "fowl", 2.5),
    ("turkey", "bird", "fowl", 8), ("duck", "bird", "waterfowl", 1.2), ("goose", "bird", "waterfowl", 4),
    ("penguin", "bird", "penguin", 20), ("ostrich", "bird", "ratite", 100), ("lizard", "reptile", "squamate", 0.1),
    ("snake", "reptile", "squamate", 2), ("turtle", "reptile", "testudine", 10), ("crocodile", "reptile", "crocodilian", 400),
    ("frog", "amphibian", "anuran", 0.05), ("toad", "amphibian", "anuran", 0.08), ("salamander", "amphibian", "urodele", 0.02),
    ("goldfish", "fish", "cypriniform", 0.1), ("carp", "fish", "cypriniform", 5), ("salmon", "fish", "salmoniform", 5),
    ("trout", "fish", "salmoniform", 2), ("tuna", "fish", "scombriform", 200), ("shark", "fish", "shark", 500),
    ("ant", "insect", "hymenopteran", 3e-6), ("bee", "insect", "hymenopteran", 1e-4), ("wasp", "insect", "hymenopteran", 1e-4),
    ("butterfly", "insect", "lepidopteran", 5e-4), ("moth", "insect", "lepidopteran", 3e-4), ("beetle", "insect", "beetle", 1e-3),
    ("fly", "insect", "dipteran", 1e-5), ("mosquito", "insect", "dipteran", 2.5e-6)]
# (language, family, branch); single-branch families use the family name as the branch
LANGUAGES = [(l, "Indo-European", b) for b, ls in (
    ("Germanic", "English German Dutch Swedish Danish Norwegian Icelandic"),
    ("Romance", "French Spanish Italian Portuguese Romanian Catalan"),
    ("Slavic", "Russian Polish Czech Ukrainian Serbian Bulgarian"),
    ("Indo-Aryan", "Hindi Bengali Urdu Punjabi Marathi"), ("Iranian", "Persian Kurdish Pashto"),
    ("Celtic", "Irish Welsh"), ("Hellenic", "Greek")) for l in ls.split()] + [
    (l, f, f) for f, ls in (("Uralic", "Finnish Hungarian Estonian"), ("Turkic", "Turkish Kazakh Uzbek Azerbaijani"),
                            ("Semitic", "Arabic Hebrew Amharic"), ("Sino-Tibetan", "Chinese Burmese Tibetan"),
                            ("Japonic", "Japanese"), ("Koreanic", "Korean"),
                            ("Austronesian", "Indonesian Malay Tagalog Javanese Hawaiian"),
                            ("Niger-Congo", "Swahili Zulu Xhosa Yoruba"), ("Dravidian", "Tamil Telugu Kannada Malayalam"),
                            ("Austroasiatic", "Vietnamese Khmer"), ("Kra-Dai", "Thai Lao")) for l in ls.split()]
# (term, generation relative to ego, sex +1 male / -1 female / 0 either, lineal 1 / collateral 0 / by marriage -1)
KINSHIP = [("great-grandfather", 3, 1, 1), ("great-grandmother", 3, -1, 1), ("grandfather", 2, 1, 1),
           ("grandmother", 2, -1, 1), ("father", 1, 1, 1), ("mother", 1, -1, 1), ("uncle", 1, 1, 0), ("aunt", 1, -1, 0),
           ("brother", 0, 1, 0), ("sister", 0, -1, 0), ("cousin", 0, 0, 0), ("husband", 0, 1, -1), ("wife", 0, -1, -1),
           ("son", -1, 1, 1), ("daughter", -1, -1, 1), ("nephew", -1, 1, 0), ("niece", -1, -1, 0),
           ("grandson", -2, 1, 1), ("granddaughter", -2, -1, 1), ("great-grandson", -3, 1, 1),
           ("great-granddaughter", -3, -1, 1)]
# scalar adjectives / adverbs: (scale, comparison question, words from low to high)
SCALES = [("temperature", "Which is hotter", "freezing cold cool lukewarm warm hot scorching"),
          ("size", "Which is bigger", "tiny small medium large huge gigantic"),
          ("quality", "Which is better", "awful bad mediocre decent good great excellent"),
          ("frequency", "Which happens more often", "never rarely sometimes often usually always"),
          ("likelihood", "Which is more likely", "impossible unlikely possible likely certain")]
RANKS = "private corporal sergeant lieutenant captain major colonel general".split()
LIFE_STAGES = "newborn infant toddler child teenager adult elderly".split()
ZODIAC = "Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius Capricorn Aquarius Pisces".split()
CHINESE_ZODIAC = "Rat Ox Tiger Rabbit Dragon Snake Horse Goat Monkey Rooster Dog Pig".split()
COMPASS = "north northeast east southeast south southwest west northwest".split()
# units: (unit, dimension, comparison question, value in SI base units)
UNITS = [(u, "time", "Which lasts longer", v) for u, v in (
    ("second", 1), ("minute", 60), ("hour", 3600), ("day", 86400), ("week", 604800), ("month", 2.63e6), ("year", 3.156e7),
    ("decade", 3.156e8), ("century", 3.156e9), ("millennium", 3.156e10))] + [
    (u, "length", "Which is longer", v) for u, v in (
    ("millimeter", 1e-3), ("centimeter", 1e-2), ("inch", 0.0254), ("foot", 0.3048), ("yard", 0.9144), ("meter", 1),
    ("kilometer", 1e3), ("mile", 1609))] + [
    (u, "mass", "Which is heavier", v) for u, v in (
    ("milligram", 1e-6), ("gram", 1e-3), ("ounce", 0.02835), ("pound", 0.4536), ("kilogram", 1), ("ton", 1000))]
PREFIXES = [("pico", -12), ("nano", -9), ("micro", -6), ("milli", -3), ("centi", -2), ("deci", -1), ("kilo", 3), ("mega", 6),
            ("giga", 9), ("tera", 12), ("peta", 15)]
# group (1..18) of elements 1..36, for the periodic-table targets of the existing elements domain
ELEMENT_GROUPS = [1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17, 18]


# ------------------------------------------------------------------------------------------------ target features

def _std(F):
    """Centre and scale every column; constant columns are DROPPED. (Before 2026-10-13 a column that is zero up to
    rounding, e.g. sin(pi x) of the period-2 helix term on integers (std ~1e-14), was scaled up into unit-variance
    noise: an unpredictable column that lowered every helix R^2 -- found by the living-lab-page audit; the training
    code, numlib.features / centred_gram, always dropped it.)"""
    F = np.asarray(F, dtype=np.float64)
    F = F - F.mean(0)
    s = F.std(0)
    keep = s > 1e-9 * max(float(s.max()), 1e-300)
    return F[:, keep] / s[keep]


def fourier(idx, n, ks):
    th = 2 * np.pi * np.asarray(idx)[:, None] / n
    return _std(np.concatenate([np.cos(th * k) for k in ks] + [np.sin(th * k) for k in ks if 2 * k != n], 1))


def line(v):
    return _std(np.asarray(v, dtype=np.float64)[:, None])


def curve(v, deg=3):
    x = np.asarray(v, dtype=np.float64)
    x = 2 * (x - x.min()) / (np.ptp(x) or 1) - 1
    return _std(np.polynomial.legendre.legvander(x, deg)[:, 1:])


def helix(v, periods=(2, 5, 10, 100)):
    v = np.asarray(v, dtype=np.float64)
    cols = [v]
    for T in periods:
        cols += [np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)]
    return _std(np.stack(cols, 1))


def unit_vec(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def harmonics(lat_deg, lon_deg):
    """Real spherical harmonics of degree 1 and 2 (8 functions) on the unit sphere."""
    x, y, z = unit_vec(lat_deg, lon_deg).T
    return _std(np.stack([x, y, z, x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1], 1))


# ------------------------------------------------------------------------------------------------ data-backed lists

def _colours():
    import colorsys
    from matplotlib.colors import CSS4_COLORS, to_rgb
    out = []
    for c in COLOURS:
        h, s, v = colorsys.rgb_to_hsv(*to_rgb(CSS4_COLORS[c]))
        if s > 0.3 and v > 0.25:
            out.append((c, 360 * h))
    return out


def _stars(max_mag=2.6):
    seen, out = set(), []
    with open(DATA / "stars.csv") as f:
        for r in csv.DictReader(f):
            if r["hip"] in seen or float(r["magnitude"]) > max_mag or " " in r["name"]:
                continue
            seen.add(r["hip"])
            out.append((r["name"], float(r["dec_degrees"]), float(r["ra_degrees"])))
    return out


def _cities(n=60, per_country=2):
    import geonamescache
    cs = sorted(geonamescache.GeonamesCache().get_cities().values(), key=lambda c: -c["population"])
    out, cnt, names = [], {}, set()
    for c in cs:
        nm = c["name"]
        if not nm.isascii() or nm in names or cnt.get(c["countrycode"], 0) >= per_country:
            continue
        cnt[c["countrycode"]] = cnt.get(c["countrycode"], 0) + 1
        names.add(nm)
        out.append((nm, float(c["latitude"]), float(c["longitude"])))
        if len(out) == n:
            break
    return out


def _emotions():
    va = {}
    with open(DATA / "warriner.csv") as f:
        for r in csv.DictReader(f):
            va[r["Word"]] = (float(r["V.Mean.Sum"]), float(r["A.Mean.Sum"]))
    return [(w, *va[w]) for w in EMOTIONS if w in va]


def onehot(labels, min_count=2):
    """Centred one-hot code; categories with fewer than min_count items are dropped (a held-out singleton can never
    be predicted, so its column would only add a constant penalty to the LOO R^2)."""
    labels = list(labels)
    cats = [c for c in dict.fromkeys(labels) if labels.count(c) >= min_count]
    F = np.array([[float(l == c) for c in cats] for l in labels])
    return F - F.mean(0)


def tree(levels, weights=(1.0, 0.7), min_count=2):
    """Hierarchy as features: weighted one-hot codes per level (e.g. class, order). Squared distances are then
    ultrametric-like: same order < same class < different class. Targets drop singleton categories (see onehot);
    the test sets use min_count=1 so that two singletons (Japanese, Korean) are not counted as related. A category
    with the same members as one of the level above (a single-branch family) is not repeated: it would add a
    duplicate column (rank deficiency) and count that split twice.
    Note: the atlas's per-column R^2 ignores the weights; its pooled R^2 (r2p) and CKA use them."""
    cols, seen = [], set()
    for lv, w in zip(levels, weights):
        lv = list(lv)
        for cat in dict.fromkeys(lv):
            members = frozenset(i for i, x in enumerate(lv) if x == cat)
            if len(members) < min_count or members in seen:
                continue
            seen.add(members)
            col = np.array([float(i in members) for i in range(len(lv))])
            cols.append(w * (col - col.mean()))
    return np.stack(cols, 1)


def within(groups, values):
    """One column per group: the value z-scored within that group, 0 for items of other groups (separate scales)."""
    groups, values = np.asarray(groups), np.asarray(values, dtype=np.float64)
    cols = []
    for g in dict.fromkeys(groups):
        m = groups == g
        c = np.zeros(len(values))
        c[m] = (values[m] - values[m].mean()) / (values[m].std() or 1)
        cols.append(c)
    return np.stack(cols, 1)


def lab(rgb):
    """sRGB (0..1) -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _countries(n=70):
    """The n most populous countries with an ASCII name and a capital found among geonamescache's cities:
    (name, capital lat, capital lon, population, area km2, continent)."""
    import geonamescache
    gc = geonamescache.GeonamesCache()
    import unicodedata
    fold = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    exact, folded, alt = {}, {}, {}   # capital lookup: exact name, then accent-folded name, then alternate names
    for c in sorted(gc.get_cities().values(), key=lambda c: -c["population"]):
        ll = (float(c["latitude"]), float(c["longitude"]))
        exact.setdefault((c["countrycode"], c["name"]), ll)
        folded.setdefault((c["countrycode"], fold(c["name"])), ll)
        for nm in c.get("alternatenames") or []:
            alt.setdefault((c["countrycode"], fold(nm)), ll)
    def capital(iso, name):
        return exact.get((iso, name)) or folded.get((iso, fold(name))) or alt.get((iso, fold(name)))
    out = []
    for c in sorted(gc.get_countries().values(), key=lambda c: -c["population"]):
        ll = capital(c["iso"], c["capital"])
        if not c["name"].isascii() or ll is None or not c["areakm2"]:
            continue
        out.append((c["name"], *ll, float(c["population"]), float(c["areakm2"]), c["continentcode"]))
        if len(out) == n:
            break
    return out


def _us_cities(n=45):
    """The largest city of each state (Gurnee & Tegmark us_place table), one per name, DC left out: (name, lat, lon)."""
    seen_state, seen_name, out = set(), set(), []
    rows = sorted(csv.DictReader(open(DATA / "gurnee_us_place.csv")), key=lambda r: -float(r["population"] or 0))
    for r in rows:
        if r["entity_type"] != "city" or r["state_id"] in seen_state or r["state_id"] == "DC" or r["name"] in seen_name:
            continue
        seen_state.add(r["state_id"]); seen_name.add(r["name"])
        out.append((r["name"], float(r["latitude"]), float(r["longitude"])))
        if len(out) == n:
            break
    return out


def _landmarks():
    return [(r["name"], float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(DATA / "landmarks.csv"))]


def _people():
    born = {}
    for r in csv.DictReader(open(DATA / "gurnee_historical_figure.csv")):
        if r["name"] in PEOPLE:   # first row per name in Wikidata-id order (the table is sorted that way)
            born.setdefault(r["name"], int(float(r["birth_year"])))
    show = {"Martin Luther King Jr.": "Martin Luther King"}   # prompt name (the last token would be ".")
    return [(show.get(p, p), born[p]) for p in PEOPLE if p in born]


# ------------------------------------------------------------------------------------------------ domains

def domains() -> dict:
    D = {}

    def add(name, kind, items, cued, targets, coord):
        D[name] = dict(kind=kind, items=list(items), cued=cued, targets=targets, coord=np.asarray(coord))

    x = np.arange(10, 100)
    add("numbers", "helix", [str(v) for v in x], "The number {}", {"helix": helix(x), "line": line(x)}, x)
    v = np.array(NUMWORD_VALUES)
    add("number_words", "helix", NUMWORDS, "The number {}", {"helix": helix(v, (2, 5, 10)), "line": line(v)}, v)
    i = np.arange(1, 21)
    add("ordinals", "helix", ORDINALS, "It came {}", {"helix": helix(i, (2, 5, 10)), "line": line(i)}, i)
    y = np.arange(1900, 2000)
    add("years", "helix", [str(t) for t in y], "In the year {}", {"helix": helix(y - 1900, (10, 100)), "line": line(y)}, y)
    for name, items, cued in (("days", DAYS, "The day was {}"), ("months", MONTHS, "The month of {}")):
        n = len(items); i = np.arange(n)
        add(name, "cyclic", items, cued, {"circle": fourier(i, n, (1,)), "fourier": fourier(i, n, range(1, n // 2 + 1)),
                                          "line": line(i)}, i)
    h = np.arange(24)
    add("hours", "cyclic", [f"{t}:00" for t in h], "The time was {}",
        {"circle": fourier(h, 24, (1,)), "fourier": fourier(h, 24, range(1, 13)), "base10": helix(h, (2, 5, 10)),
         "line": line(h)}, h)
    i = np.arange(26)
    add("letters", "linear", [chr(65 + k) for k in i], "The letter {}",
        {"line": line(i), "curve": curve(i), "circle": fourier(i, 26, (1,))}, i)
    i = np.arange(24)
    add("greek", "linear", GREEK, "The Greek letter {}", {"line": line(i), "curve": curve(i), "circle": fourier(i, 24, (1,))}, i)
    for name, items, cued in (("planets", PLANETS, "The planet {}"), ("elements", ELEMENTS, "The element {}")):
        i = np.arange(len(items))
        add(name, "linear", items, cued, {"line": line(i), "curve": curve(i)}, i)
    pres = [(p.replace("_", " "), k) for k, p in enumerate(PRESIDENTS, 1) if p != "-"]
    i = np.array([k for _, k in pres])
    add("presidents", "linear", [p for p, _ in pres], "President {}", {"line": line(i), "curve": curve(i)}, i)
    for name, items, cued in (("pitch", PITCH, "The note {}"), ("keys", KEYS, "A piece in the key of {}")):
        c = np.array([k for _, k in items])
        fifths = (7 * c) % 12   # position on the circle of fifths
        add(name, "cyclic", [n for n, _ in items], cued,
            {"chromatic": fourier(c, 12, (1,)), "fifths": fourier(fifths, 12, (1,)), "fourier": fourier(c, 12, range(1, 7)),
             "line": line(c)}, c)
    col = _colours(); hue = np.array([hh for _, hh in col])
    add("colours", "cyclic", [c for c, _ in col], "The colour {}",
        {"hue": _std(np.stack([np.cos(np.radians(hue)), np.sin(np.radians(hue))], 1)), "line": line(hue)}, hue)
    if _have("stars.csv", "stars"):
        st = _stars(); dec = np.array([s[1] for s in st]); ra = np.array([s[2] for s in st])
        add("stars", "sphere", [s[0] for s in st], "In the night sky, the star {}",
            {"sphere": _std(unit_vec(dec, ra)), "harmonics": harmonics(dec, ra), "plane": _std(np.stack([dec, ra], 1))},
            np.stack([dec, ra], 1))
    ci = _cities(); lat = np.array([c[1] for c in ci]); lon = np.array([c[2] for c in ci])
    add("cities", "sphere", [c[0] for c in ci], "The city of {}",
        {"sphere": _std(unit_vec(lat, lon)), "harmonics": harmonics(lat, lon), "plane": _std(np.stack([lat, lon], 1))},
        np.stack([lat, lon], 1))
    if _have("warriner.csv", "emotions"):
        em = _emotions(); V = np.array([e[1] for e in em]); A = np.array([e[2] for e in em])
        ang = np.arctan2(_std(A[:, None])[:, 0], _std(V[:, None])[:, 0])
        add("emotions", "plane", [e[0] for e in em], "I feel {}",
            {"plane": _std(np.stack([V, A], 1)), "valence": line(V), "arousal": line(A),
             "circumplex": _std(np.stack([np.cos(ang), np.sin(ang)], 1))}, np.stack([V, A], 1))
    # extra targets for two v1 domains (primary unchanged): the periodic table (Lei & Cooper 2502.10871 report a 3-D
    # spiral) and perceptual colour space
    Z = np.arange(1, len(ELEMENTS) + 1); g = np.array(ELEMENT_GROUPS); per = np.searchsorted([2, 10, 18, 36], Z) + 1
    D["elements"]["targets"].update({"table": _std(np.stack([per, g], 1)),
                                     "spiral": _std(np.stack([Z, np.cos(2 * np.pi * (g - 1) / 18), np.sin(2 * np.pi * (g - 1) / 18)], 1))})
    from matplotlib.colors import CSS4_COLORS, to_rgb
    D["colours"]["targets"]["lab"] = _std(lab([to_rgb(CSS4_COLORS[c]) for c in D["colours"]["items"]]))

    # ---- atlas v2 domains (appended, so the v1 domains' test sets are drawn exactly as before)
    co = _countries(); lat, lon = np.array([c[1] for c in co]), np.array([c[2] for c in co])
    pop, area = np.log10([c[3] for c in co]), np.log10([c[4] for c in co])
    add("countries", "sphere", [c[0] for c in co], "The country of {}",
        {"sphere": _std(unit_vec(lat, lon)), "harmonics": harmonics(lat, lon), "plane": _std(np.stack([lat, lon], 1)),
         "continent": onehot([c[5] for c in co])}, np.stack([lat, lon], 1))
    add("country_size", "magnitude", [c[0] for c in co], "The population of {}",
        {"log": line(pop), "logarea": line(area), "harmonics": harmonics(lat, lon)}, pop)
    if _have("landmarks.csv", "landmarks"):
        lm = _landmarks(); lat, lon = np.array([c[1] for c in lm]), np.array([c[2] for c in lm])
        add("landmarks", "sphere", [c[0] for c in lm], "The famous landmark {}",
            {"sphere": _std(unit_vec(lat, lon)), "harmonics": harmonics(lat, lon), "plane": _std(np.stack([lat, lon], 1))},
            np.stack([lat, lon], 1))
    if _have("gurnee_us_place.csv", "us_cities"):
        us = _us_cities(); lat, lon = np.array([c[1] for c in us]), np.array([c[2] for c in us])
        add("us_cities", "plane", [c[0] for c in us], "The American city of {}",
            {"plane": _std(np.stack([lat, lon], 1)), "sphere": _std(unit_vec(lat, lon))}, np.stack([lat, lon], 1))
    if _have("gurnee_historical_figure.csv", "people"):
        pe = _people(); yb = np.array([p[1] for p in pe], dtype=np.float64)
        add("people", "linear", [p[0] for p in pe], "The historical figure {}",
            {"rank": line(np.argsort(np.argsort(yb, kind="stable"), kind="stable")), "line": line(yb), "curve": curve(yb),
             "logtime": line(np.log10(2030 - yb))}, yb)
    i = np.arange(len(GEO_PERIODS))
    add("geo_periods", "linear", GEO_PERIODS, "The geologic period {}", {"line": line(i), "curve": curve(i)}, i)
    cls, order, mass = [a[1] for a in ANIMALS], [a[2] for a in ANIMALS], np.log10([a[3] for a in ANIMALS])
    add("animals", "tree", [a[0] for a in ANIMALS], "The animal {}",
        {"tree": tree([cls, order]), "class": onehot(cls), "logmass": line(mass)}, tree([cls, order], min_count=1))
    D["animals"]["label"] = np.array(cls)
    add("animal_size", "magnitude", [a[0] for a in ANIMALS], "The weight of a {}",
        {"log": line(mass), "tree": tree([cls, order])}, mass)
    fam, br = [l[1] for l in LANGUAGES], [l[2] for l in LANGUAGES]
    add("languages", "tree", [l[0] for l in LANGUAGES], "The language {}",
        {"tree": tree([fam, br]), "family": onehot(fam)}, tree([fam, br], min_count=1))
    D["languages"]["label"] = np.array(fam)
    gen, sex, lin = (np.array([k[j] for k in KINSHIP], dtype=np.float64) for j in (1, 2, 3))
    add("kinship", "kin", [k[0] for k in KINSHIP], "My {}",
        {"generation": line(gen), "plane": _std(np.stack([gen, sex], 1)), "full": _std(np.stack([gen, sex, lin], 1))}, gen)
    sw = [(w, s, r) for s, _, ws in SCALES for r, w in enumerate(ws.split())]
    sg, sr = [w[1] for w in sw], np.array([w[2] for w in sw], dtype=np.float64)
    add("scalar_adj", "scalar", [w[0] for w in sw], "It was {}",
        {"within": within(sg, sr), "intensity": line(within(sg, sr).sum(1)), "scale": onehot(sg)}, sr)
    D["scalar_adj"]["group"] = np.array(sg); D["scalar_adj"]["question"] = {s: q for s, q, _ in SCALES}
    for name, items, cued in (("ranks", RANKS, "The military rank of {}"), ("life_stages", LIFE_STAGES, "The life stage {}")):
        i = np.arange(len(items))
        add(name, "linear", items, cued, {"line": line(i), "curve": curve(i)}, i)
    for name, items, cued in (("zodiac", ZODIAC, "The zodiac sign {}"), ("chinese_zodiac", CHINESE_ZODIAC, "The Year of the {}"),
                              ("compass", COMPASS, "The wind blew from the {}")):
        n = len(items); i = np.arange(n)
        add(name, "cyclic", items, cued, {"circle": fourier(i, n, (1,)), "fourier": fourier(i, n, range(1, n // 2 + 1)),
                                          "line": line(i)}, i)
    ug, uv = [u[1] for u in UNITS], np.log10([u[3] for u in UNITS])
    add("units", "scalar", [u[0] for u in UNITS], "One {}",
        {"within": within(ug, uv), "scale": onehot(ug), "logsi": line(uv)}, uv)
    D["units"]["group"] = np.array(ug); D["units"]["question"] = {u[1]: u[2] for u in UNITS}
    ex = np.array([p[1] for p in PREFIXES], dtype=np.float64)
    add("prefixes", "linear", [p[0] for p in PREFIXES], "The metric prefix {}", {"line": line(ex), "curve": curve(ex)}, ex)
    return D


if __name__ == "__main__":
    for k, d in domains().items():
        print(f"{k:13s} {d['kind']:7s} n={len(d['items']):3d}  targets {list(d['targets'])}  e.g. {d['items'][:4]}")


# ------------------------------------------------------------------------------------------------ test sets
# One task family per domain, scored by choosing the answer among candidates (summed log-prob of " candidate"
# after a 3-shot prompt). offset = "k steps after x" over the domain's own items (candidates = all items, chance 1/n);
# pair = a two-way comparison that needs the domain's geometry (candidates = the two options, chance 1/2).
OFFSET = {
    "numbers": ("{x} + {k} =", "add", range(10, 50)),
    "number_words": ("{x} plus {k} is", "linear", range(1, 8)),
    "ordinals": ("{k} places after {x} is", "linear", range(1, 6)),
    "years": ("{k} years after {x} is", "linear", range(1, 21)),
    "days": ("{k} days after {x} is", "cyclic", range(1, 7)),
    "months": ("{k} months after {x} is", "cyclic", range(1, 12)),
    "hours": ("{k} hours after {x} is", "cyclic", range(1, 13)),
    "letters": ("{k} letters after {x} is", "linear", range(1, 6)),
    "greek": ("in the Greek alphabet, {k} letters after {x} is", "linear", range(1, 5)),
    "planets": ("counting outward from the Sun, {k} planets after {x} is", "linear", range(1, 4)),
    "elements": ("in the periodic table, {k} elements after {x} is", "linear", range(1, 4)),
    "pitch": ("{k} semitones above {x} is", "cyclic", range(1, 12)),
    "keys": ("{k} semitones above the key of {x} is the key of", "cyclic", range(1, 12)),
    # atlas v2
    "geo_periods": ("in the geologic time scale, {k} periods after the {x} comes the", "linear", range(1, 4)),
    "zodiac": ("{k} signs after {x} in the zodiac is", "cyclic", range(1, 12)),
    "chinese_zodiac": ("in the Chinese zodiac, {k} years after the Year of the {x} comes the Year of the", "cyclic", range(1, 12)),
}
# atlas v2 comparisons: (template, rule, margin). Rules: sphere = b or c closer to a (great-circle degrees on coord =
# (lat, lon)); tree = closer to a in the hierarchy features; angle = closer to a on the cycle (degrees); less / more =
# smaller / larger coord; within = larger coord within the same group (template filled with the group's question)
PAIR2 = {
    "countries": ("Which country is closer to {a}, {b} or {c}? Answer:", "sphere", 35),   # capitals stand for countries
    "country_size": ("Which country has more people, {b} or {c}? Answer:", "more", math.log10(3)),
    "landmarks": ("Which landmark is closer to {a}, {b} or {c}? Answer:", "sphere", 25),
    "us_cities": ("Which city is closer to {a}, {b} or {c}? Answer:", "sphere", 5),
    "people": ("Who was born earlier, {b} or {c}? Answer:", "less", 50),
    "animals": ("Which animal is more closely related to {a}, {b} or {c}? Answer:", "tree", 0.5),
    "animal_size": ("Which animal is heavier, {b} or {c}? Answer:", "more", 1.0),
    "languages": ("Which language is more closely related to {a}, {b} or {c}? Answer:", "tree", 0.5),
    "kinship": ("Who belongs to an older generation, my {b} or my {c}? Answer:", "more", 1),
    "scalar_adj": ("{q}, {b} or {c}? Answer:", "within", 2),
    "ranks": ("Which military rank is higher, {b} or {c}? Answer:", "more", 2),
    "life_stages": ("Which stage of life comes later, {b} or {c}? Answer:", "more", 2),
    "compass": ("Which direction is closer to {a}, {b} or {c}? Answer:", "angle", 45),
    "units": ("{q}: one {b} or one {c}? Answer:", "within", math.log10(3)),
    "prefixes": ("Which metric prefix stands for a larger amount, {b} or {c}? Answer:", "more", 2),
}
PAIR = {
    "presidents": "Who was US president earlier, {b} or {c}? Answer:",
    "colours": "Which colour is closer in hue to {a}, {b} or {c}? Answer:",
    "stars": "Which star is closer to {a} in the night sky, {b} or {c}? Answer:",
    "cities": "Which city is closer to {a}, {b} or {c}? Answer:",
    "emotions": "Which feeling is more pleasant, {b} or {c}? Answer:",
}
WORD_NUM = dict(zip(("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                     "seventeen eighteen nineteen twenty").split(), range(1, 21)))


def _sphere_dist(p, q):
    a, b = unit_vec(*np.asarray(p)[:, None]), unit_vec(*np.asarray(q)[:, None])
    return float(np.degrees(np.arccos(np.clip((a * b).sum(), -1, 1))))


def _singular(tmpl, x, kw, k):
    """Fill an offset template; for k = 1 the unit word after the {k} slot becomes singular ('one letters after' ->
    'one letter after'). The slot is found in the template, not by searching the filled text for kw (2026-10-14: for
    number_words the first 'one' in 'one plus one is' is the item, and the question read 'one plu one is')."""
    w = tmpl.split(" ")
    at = w.index("{k}") + 1
    if k == 1 and at < len(w) and w[at].endswith("s") and w[at] != "is":
        w[at] = w[at][:-1]
    return " ".join(w).format(x=x, k=kw)


def _offset_questions(name, d):
    """Every valid (question, content-free question, answer index) of an offset domain, in a fixed order."""
    tmpl, mode, ks = OFFSET[name]
    items, nI = d["items"], len(d["items"])
    out = []
    for i in range(nI):
        for k in ks:
            if mode == "add":
                a = int(d["coord"][i])
                if a + k <= 99:
                    out.append((tmpl.format(x=a, k=k), tmpl.format(x="N/A", k="N/A"), items.index(str(a + k))))
                continue
            j = (i + k) % nI if mode == "cyclic" else i + k
            if j >= nI or (name == "number_words" and items[j] not in WORD_NUM):   # stay within one..twenty
                continue
            kw = str(k) if name in ("years", "hours") else list(WORD_NUM)[k - 1]
            out.append((_singular(tmpl, items[i], kw, k), tmpl.format(x="N/A", k="N/A"), j))
    return out


def _pair_answer(name, d, a, b, e):
    """For the comparison 'a: b or e', the index (0 = b, 1 = e) of the right answer, or None if the comparison is
    not clear enough (margin) or not allowed; also the question text for grouped domains."""
    c = d["coord"]
    if name in PAIR2:
        tmpl, rule, margin = PAIR2[name]
        c = np.asarray(c, dtype=np.float64)
        if rule in ("sphere", "tree", "angle"):
            if rule == "sphere":
                db, de = _sphere_dist(c[a], c[b]), _sphere_dist(c[a], c[e])
            elif rule == "tree":
                db, de = float(((c[a] - c[b]) ** 2).sum()), float(((c[a] - c[e]) ** 2).sum())
            else:
                n = len(c)
                cyc = lambda u, v: 360 / n * min(abs(c[u] - c[v]), n - abs(c[u] - c[v]))
                db, de = cyc(a, b), cyc(a, e)
            return (None, "") if abs(db - de) < margin else (0 if db < de else 1, "")
        if rule == "within":
            g = d["group"]
            if g[b] != g[e] or abs(c[b] - c[e]) < margin:
                return None, ""
            return (0 if c[b] > c[e] else 1), d["question"][g[b]]
        if abs(c[b] - c[e]) < margin:
            return None, ""
        return (0 if (c[b] < c[e]) == (rule == "less") else 1), ""
    if name == "presidents":
        return (None, "") if abs(c[b] - c[e]) < 4 else (0 if c[b] < c[e] else 1, "")
    if name == "emotions":
        return (None, "") if abs(c[b][0] - c[e][0]) < 2 else (0 if c[b][0] > c[e][0] else 1, "")
    dist = (lambda u, v: min(abs(c[u] - c[v]), 360 - abs(c[u] - c[v]))) if name == "colours" else \
        (lambda u, v: _sphere_dist(c[u], c[v]))
    db, de = dist(a, b), dist(a, e)
    return (None, "") if abs(db - de) < (30 if name == "colours" else 25) else (0 if db < de else 1, "")


def tasks(D: dict, n: int = 60, shots: int = 3, seed: int = 0) -> dict:
    """{domain: [(prompt, candidates, answer_index, content_free_prompt), ...]}.

    Atlas v2 (after the 2026-10-13 audit, runs/atlas2/AUDIT.md):
      - every test question is unique, and the shots are separate questions (no question or comparison pair of the
        test set appears among the shots);
      - offset domains: questions drawn without replacement from all valid (item, k); small domains get fewer than n;
      - comparison domains: distinct comparisons, each asked in BOTH orders ("b or c" and "c or b"), so a preference
        for the first / second option cancels; up to n rows (n / 2 comparisons);
      - content_free_prompt = the same shots and question with every item replaced by "N/A": scoring the candidates
        there measures the model's prior for each candidate (used for calibrated accuracy, Zhao et al. 2021)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, d in D.items():
        items, nI = d["items"], len(d["items"])
        if name in OFFSET:
            qs = _offset_questions(name, d)
            order = rng.permutation(len(qs))
            sel = [qs[i] for i in order[:shots + n]]
            shot_rows = [(q, items, j) for q, _, j in sel[:shots]]
            test = [(q, list(items), j, cf) for q, cf, j in sel[shots:]]
        else:
            tmpl = PAIR2[name][0] if name in PAIR2 else PAIR[name]
            uses_a = "{a}" in tmpl
            seen, picked, tries = set(), [], 0
            while len(picked) < shots + (n + 1) // 2 and tries < 20000:
                tries += 1
                a, b, e = (int(t) for t in rng.choice(nI, 3, replace=False))
                key = (a if uses_a else -1, min(b, e), max(b, e))
                if key in seen:
                    continue
                ans, q = _pair_answer(name, d, a, b, e)
                if ans is None:
                    continue
                seen.add(key)
                picked.append((a, b, e, ans, q))
            assert len(picked) > shots + 5, f"{name}: only {len(picked)} distinct comparisons"
            fill = lambda a, b, e, q: tmpl.format(a=items[a], b=items[b], c=items[e], q=q)
            cf = lambda q: tmpl.format(a="N/A", b="N/A", c="N/A", q=q)
            shot_rows = [(fill(a, b, e, q), (items[b], items[e]), ans) for a, b, e, ans, q in picked[:shots]]
            shot_pairs = {frozenset((b, e)) for _, b, e, _, _ in picked[:shots]}
            test = []
            for a, b, e, ans, q in picked[shots:]:
                if frozenset((b, e)) in shot_pairs and not uses_a:
                    continue
                test.append((fill(a, b, e, q), [items[b], items[e]], ans, cf(q)))
                test.append((fill(a, e, b, q), [items[e], items[b]], 1 - ans, cf(q)))
            test = test[:n]
        shot_txt = "".join(f"{p} {cs[ai]}\n" for p, cs, ai in shot_rows)
        out[name] = [(shot_txt + p, cs, ai, shot_txt + cf) for p, cs, ai, cf in test]
    return out
