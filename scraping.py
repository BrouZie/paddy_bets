import requests
from bs4 import BeautifulSoup
import pandas as pd


FBREF = "https://fbref.com/en/comps/9/Premier-League-Stats"

data = requests.get(FBREF)
print(data)

soup = BeautifulSoup(data.text, "html.parser")

# CSS selectors
standings_table = soup.select('table.stats_table')[0]

a_tags = standings_table.find_all('a')
links = [lst.get("href") for lst in a_tags]
club_links = [link for link in links if link.startswith("/en/squads/")]

club_urls = [f"https://fbref.com{link}" for link in club_links]

# for team in club_urls:
#     matches = pd.read_html(requests.get(club_urls[team]).text, match="Scores & Fixtures")

matches = pd.read_html(requests.get(club_urls[0]).text, match="Scores & Fixtures")
print(type(matches[0]))
