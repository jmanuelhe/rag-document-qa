"""Normalización y tokenización bilingüe (español/inglés)."""
import re
import unicodedata

STOPWORDS = set("""
a al algo algún alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde
dos el ella ellas ellos en entre era es esa ese eso esta este esto estos fue fueron ha han hay la las le les lo
los mas más me mi muy no nos o otra otro para pero por porque que qué se sea ser si sin sobre su sus tal también
te tiene tu un una uno unos y ya cuál cuáles cuándo dónde quién qué hubo sobre cual
the a an and or of to in for on with from by is are was were be how i i'd i'm your you that this these it its
at as what which who into here's will can my me do does did
""".split())


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    t = strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", t)
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
