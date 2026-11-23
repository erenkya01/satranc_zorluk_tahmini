import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "puzzles.db"
CHUNK_SIZE = 100_000
MAX_RATING_DEVIATION = 90
RANDOM_SEED = 42


def find_input_file() -> Path:
    candidates = list(DATA_DIR.glob("*.csv.zst")) + list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "data/ klasöründe .csv veya .csv.zst uzantılı veri dosyası bulunamadı!"
        )
    return candidates[0]


def main():
    input_file = find_input_file()
    print(f"Kaynak dosya bulundu: {input_file.name}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Hızlı yazma için SQLite performans ayarları
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    cursor.execute("""
        CREATE TABLE raw_puzzles (
            puzzle_id TEXT PRIMARY KEY,
            fen TEXT NOT NULL,
            moves TEXT NOT NULL,
            rating INTEGER NOT NULL,
            rating_deviation INTEGER NOT NULL,
            popularity INTEGER,
            nb_plays INTEGER,
            themes TEXT,
            split TEXT NOT NULL
        )
    """)
    conn.commit()

    rng = np.random.default_rng(RANDOM_SEED)
    total_read = 0
    total_kept = 0

    reader = pd.read_csv(input_file, chunksize=CHUNK_SIZE)

    for chunk_idx, chunk in enumerate(reader, start=1):
        total_read += len(chunk)

        # Rapor Bölüm 6.2: Yüksek derece sapmasına sahip gürültülü bulmacaları ele
        clean = chunk[chunk["RatingDeviation"] <= MAX_RATING_DEVIATION].copy()
        if clean.empty:
            continue

        # Sızıntısız (leak-free) %80 train, %10 val, %10 test ataması
        rand_vals = rng.random(len(clean))
        splits = np.where(
            rand_vals < 0.80, "train", np.where(rand_vals < 0.90, "val", "test")
        )

        clean_df = pd.DataFrame({
            "puzzle_id": clean["PuzzleId"].astype(str),
            "fen": clean["FEN"].astype(str),
            "moves": clean["Moves"].astype(str),
            "rating": clean["Rating"].astype(int),
            "rating_deviation": clean["RatingDeviation"].astype(int),
            "popularity": clean["Popularity"].fillna(0).astype(int),
            "nb_plays": clean["NbPlays"].fillna(0).astype(int),
            "themes": clean["Themes"].fillna("").astype(str),
            "split": splits,
        })

        clean_df.to_sql("raw_puzzles", conn, if_exists="append", index=False)
        total_kept += len(clean_df)

        if chunk_idx % 5 == 0:
            print(f"İşlenen satır: {total_read:,} | Temiz kaydedilen: {total_kept:,}")

    print("İndeksler oluşturuluyor...")
    cursor.execute("CREATE INDEX idx_raw_split ON raw_puzzles(split);")
    conn.commit()

    # Özet istatistikleri yazdır
    summary = pd.read_sql_query(
        """
        SELECT split, COUNT(*) as adet, ROUND(AVG(rating), 1) as ort_puan
        FROM raw_puzzles
        GROUP BY split
        """,
        conn,
    )
    conn.close()

    print("\n--- VERİTABANI ÖZETİ ---")
    print(f"Toplam okunan : {total_read:,}")
    print(f"Filtre sonrası : {total_kept:,} (RatingDeviation <= {MAX_RATING_DEVIATION})")
    print(summary.to_string(index=False))
    print(f"\nVeritabanı hazır: {DB_PATH}")


if __name__ == "__main__":
    main()