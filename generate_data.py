"""Generate a realistic synthetic Atlantic Top-50 playlist dataset
matching the schema and statistics described in the analysis notebook.

Columns: date, position, song, artist, popularity, duration_ms,
         album_type, total_tracks, is_explicit, album_cover_url
"""
import csv
import random
from datetime import date, timedelta

random.seed(42)

ARTISTS = [
    "Taylor Swift", "Morgan Wallen", "Drake", "SZA", "Bad Bunny",
    "Olivia Rodrigo", "The Weeknd", "Post Malone", "Luke Combs",
    "Billie Eilish", "Travis Scott", "Sabrina Carpenter", "Shaboozey",
    "Benson Boone", "Kendrick Lamar", "Chappell Roan", "Lady Gaga",
    "Bruno Mars", "Miley Cyrus", "Dua Lipa",
    "Tate McRae", "Zach Bryan", "Jack Harlow", "21 Savage", "Doja Cat",
    "Ariana Grande", "Jelly Roll", "Tyler The Creator", "Megan Thee Stallion",
    "Rihanna", "Ed Sheeran", "Justin Bieber", "Adele", "Beyonce",
    "Coldplay", "Imagine Dragons", "Kendrick Lamar", "Lil Durk",
    "Eslabon Armado", "Grupo Frontera", "Peso Pluma", "Karol G",
    "Shakira", "Rauw Alejandro", "Feid", "Myke Towers", "Young Miko",
    "Noah Kahan", "Hozier", "Teddy Swims", "Benson Boone",
]

SONG_POOL = [
    "Anti-Hero", "Cruel Summer", "Last Night", "Rich Men North of Richmond",
    "First Person Shooter", "Snooze", "Kill Bill", "Where Does The Good Go",
    "Flowers", "Vampire", "greedy", "I Remember Everything",
    "Cruel Summer", "Espresso", "A Bar Song (Tipsy)", "Beautiful Things",
    "Not Like Us", "Good Luck, Babe!", "Die With A Smile", "I Had Some Help",
    "Used To Be Young", "Houdini", "Calm Down", "Too Sweet",
    "Lose Control", "Million Dollar Baby", "Hiss", "yeah right",
    "Fortnight", "Down Bad", "I Can Do It With A Broken Heart",
    "Taste", "Please Please Please", "Good Luck, Babe", "360",
    "Pink Pony Club", "Hot To Go", "The Giver", "Bad Habit",
    "Unholy", "Calm Down", "Snooze", "Ella Baila Sola",
    "Lady Gaga", "Bongos", "Rush", "Dance The Night",
    "All My Life", "un x100to", "La Bebe", "Lady Gaga",
    "Mamiii", "Gracias a Ti", "Por el Contrario", "Qlona",
    "Mi Ex Tenia Razon", "Beso", "Vampiros", "Lady Gaga",
]

ALBUM_TYPES = ["album", "single", "compilation"]
ALBUM_WEIGHTS = [0.55, 0.40, 0.05]

START = date(2024, 5, 18)
END = date(2025, 11, 27)
TOTAL_DAYS = (END - START).days + 1  # 554 days

all_dates = [START + timedelta(days=d) for d in range(TOTAL_DAYS)]
# Remove a few days to mimic the notebook's missing-calendar-day note
skip = {date(2025, 3, 14), date(2025, 3, 25), date(2025, 7, 11), date(2025, 8, 13)}
dates = [d for d in all_dates if d not in skip]
# ~550 dates * 50 rows = ~27,500 rows (matches notebook's 27,750 clean rows)

def make_song_entry(day_idx):
    artist = random.choice(ARTISTS)
    song = random.choice(SONG_POOL)
    duration_ms = random.randint(120_000, 300_000)
    album_type = random.choices(ALBUM_TYPES, weights=ALBUM_WEIGHTS, k=1)[0]
    total_tracks = 1 if album_type == "single" else random.randint(2, 40)
    is_explicit = random.random() < 0.45
    popularity = max(0, min(100, int(random.gauss(75, 18))))
    cover = f"https://i.scdn.co/image/ab67616d0000b273{random.randint(0, 0xFFFFFF):06x}"
    return [song, artist, duration_ms, album_type, total_tracks, is_explicit, popularity, cover]

def write_csv():
    rows = []
    for d in dates:
        # Pick 50 unique (song, artist) combos for the day
        seen = set()
        picks = []
        attempts = 0
        while len(picks) < 50 and attempts < 200:
            entry = make_song_entry(0)
            key = (entry[0], entry[1])
            if key not in seen:
                seen.add(key)
                picks.append(entry)
            attempts += 1
        # Pad if needed
        while len(picks) < 50:
            entry = make_song_entry(0)
            picks.append(entry)

        # Sort by descending popularity to assign positions (1 = most popular)
        picks.sort(key=lambda e: -e[6])
        for i, e in enumerate(picks, start=1):
            dstr = d.strftime("%d-%m-%Y")
            rows.append([dstr, i, e[0], e[1], e[6], e[2], e[3], e[4], e[5], e[7]])

    with open("atlantic_clean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "position", "song", "artist", "popularity",
                    "duration_ms", "album_type", "total_tracks", "is_explicit",
                    "album_cover_url"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to atlantic_clean.csv")

if __name__ == "__main__":
    write_csv()
