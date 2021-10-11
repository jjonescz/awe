from dataclasses import dataclass
import os
import re

SWDE_URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
DATA_DIR = 'data'
SWDE_ZIP = f'{DATA_DIR}/swde.zip'
SWDE_DIR = f'{DATA_DIR}/swde'
SWDE_DATA_DIR = f'{SWDE_DIR}/data'

VERTICALS = [
    "auto",
    "book",
    "camera",
    "job",
    "movie",
    "nbaplayer",
    "restaurant",
    "university"
]

WEBSITE_REGEX = r"^(\w+)-(\w+)\((\d+)\)$"

@dataclass
class Website:
    vertical: str
    name: str
    page_count: int

def parse_website(dir_name):
    match = re.search(WEBSITE_REGEX, dir_name)
    return Website(match.group(1), match.group(2), int(match.group(3)))

def get_websites(vertical):
    for subdir in os.listdir(f'{SWDE_DATA_DIR}/{vertical}'):
        yield parse_website(subdir)

WEBSITES = {
    "auto": [
        "msn", "aol", "kbb", "cars", "yahoo", "autoweb", "autobytel",
        "automotive", "carquotes", "motortrend"
    ],
    "book": [
        "abebooks", "amazon", "barnesandnoble", "bookdepository",
        "booksamillion", "borders", "buy", "christianbook", "deepdiscount",
        "waterstones"
    ],
    "camera": [
        "amazon", "beachaudio", "buy", "compsource", "ecost", "jr", "newegg",
        "onsale", "pcnation", "thenerds"
    ],
    "job": [
        "careerbuilder", "dice", "hotjobs", "job", "jobcircle", "jobtarget",
        "monster", "nettemps", "rightitjobs", "techcentric"
    ],
    "movie": [
        "allmovie", "amctv", "boxofficemojo", "hollywood", "iheartmovies",
        "imdb", "metacritic", "msn", "rottentomatoes", "yahoo"
    ],
    "nbaplayer": [
        "espn", "fanhouse", "foxsports", "msnca", "nba", "si", "slam",
        "usatoday", "wiki", "yahoo"
    ],
    "restaurant": [
        "fodors", "frommers", "gayot", "opentable", "pickarestaurant",
        "restaurantica", "tripadvisor", "urbanspoon", "usdiners", "zagat"
    ],
    "university": [
        "collegeboard", "collegenavigator", "collegeprowler", "collegetoolkit",
        "ecampustours", "embark", "matchcollege", "princetonreview",
        "studentaid", "usnews"
    ]
}
