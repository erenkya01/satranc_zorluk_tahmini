import base64
import sqlite3
import sys
from pathlib import Path

import chess
import chess.svg
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import PIECE_NAMES_TR, PuzzleDifficultyPredictor

DB_PATH = ROOT_DIR / "data" / "puzzles.db"

st.set_page_config(
    page_title="Satranç Bulmacası Zorluk Tahmini",
    page_icon="♟️",
    layout="wide",
)


@st.cache_resource
def get_predictor():
    return PuzzleDifficultyPredictor()


def fetch_random_test_puzzle(min_rating: int, max_rating: int):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT puzzle_id, fen, moves, rating, themes
        FROM raw_puzzles
        WHERE split = 'test' AND rating BETWEEN ? AND ?
        ORDER BY RANDOM()
        LIMIT 1
        """,
        (min_rating, max_rating),
    ).fetchone()
    conn.close()
    return row


def render_svg_board(fen: str, last_move: chess.Move | None, orientation_white: bool):
    board = chess.Board(fen)
    orientation = chess.WHITE if orientation_white else chess.BLACK
    svg_code = chess.svg.board(
        board=board,
        lastmove=last_move,
        orientation=orientation,
        size=430,
    )
    b64 = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<div style="display:flex; justify-content:center;"><img src="data:image/svg+xml;base64,{b64}"/></div>',
        unsafe_allow_html=True,
    )


def difficulty_tier(rating: int) -> str:
    if rating < 1100:
        return "Başlangıç (Novice)"
    if rating < 1500:
        return "Orta (Intermediate)"
    if rating < 1900:
        return "İleri (Advanced)"
    if rating < 2300:
        return "Usta Adayı (Expert)"
    return "Usta Seviyesi (Master)"


def main():
    st.title("♟️ Satranç Bulmacalarının Zorluk Derecesi Tahmin Sistemi")
    st.caption(
        "Sivas Cumhuriyet Üniversitesi Bilgisayar Mühendisliği Bitirme Projesi | "
        "Hibrit GBDT Topluluğu + Artık Öğrenme + Yapısal Özellik Mühendisliği"
    )

    predictor = get_predictor()

    with st.sidebar:
        st.header("Veri Kaynağı Seçimi")
        mode = st.radio(
            "Çalışma Modu:",
            ["Test Kümesinden Bulmaca Çek", "Özel Pozisyon Gir (FEN + PGN/UCI)"],
        )

        st.divider()
        actual_rating = None
        puzzle_id = None
        themes = ""

        if mode == "Test Kümesinden Bulmaca Çek":
            rating_range = st.slider(
                "Hedef Zorluk Aralığı (Glicko-2):", 600, 2800, (1000, 2200), step=100
            )
            if "sample_puzzle" not in st.session_state or st.button(
                "🎲 Yeni Test Bulmacası Getir", use_container_width=True
            ):
                if DB_PATH.exists():
                    st.session_state["sample_puzzle"] = fetch_random_test_puzzle(*rating_range)
                else:
                    st.error("data/puzzles.db bulunamadı!")

            sample = st.session_state.get("sample_puzzle")
            if sample:
                puzzle_id, fen_input, moves_input, actual_rating, themes = sample
                st.info(f"**Bulmaca ID:** `{puzzle_id}`\n\n**Gerçek Lichess Puanı:** `{actual_rating}`")
            else:
                fen_input = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
                moves_input = "g8f6 h5f7"
        else:
            st.markdown("Dışarıdan herhangi bir FEN ve çözüm hamlelerini (UCI veya PGN) yapıştırın:")
            fen_input = st.text_input(
                "FEN Pozisyonu:",
                value="r2q1rk1/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 b - - 0 8",
            )
            moves_input = st.text_area(
                "Hamle Dizisi (1. hamle rakip hatası, devamı çözüm):",
                value="c6d4 f3d4 c5d4",
                height=90,
            )

        st.divider()
        st.caption(f"Aktif Model Dizini: `{predictor.model_dir.name}`")

    try:
        result = predictor.predict(fen_input, moves_input)
    except Exception as exc:
        st.error(f"Pozisyon veya hamle dizisi çözümlenemedi: {exc}")
        return

    pred_rating = result["predicted_rating"]
    margin = result["expected_margin"]
    states = result["states"]
    feats = result["features"]

    # Üst Metrik Kartları
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Tahmini Glicko-2 Zorluğu", f"{pred_rating} Elo")
    col_m2.metric("Belirsizlik Aralığı", f"{result['ci_low']} – {result['ci_high']}", delta=f"±{margin} Elo", delta_color="off")
    col_m3.metric("Zorluk Kategorisi", difficulty_tier(pred_rating))

    if actual_rating is not None:
        diff = pred_rating - actual_rating
        col_m4.metric("Gerçek Lichess Puanı", f"{actual_rating} Elo", delta=f"Fark: {diff:+d} Elo")
    else:
        col_m4.metric("Çözüm Derinliği", f"{feats['solver_moves_count']} Hamle ({feats['solution_ply']} ply)")

    st.divider()

    # Sol: Etkileşimli Satranç Tahtası | Sağ: Model Kırılımı ve Yapısal Analiz
    left_col, right_col = st.columns([1.05, 1.15])

    with left_col:
        st.subheader(f"Etkileşimli Tahta ({result['solver_color']} Oynar)")
        step_idx = st.slider(
            "Hamle Adımı (0: Ham FEN, 1: Rakip Hatası Sonrası Başlangıç, 2+: Çözüm):",
            min_value=0,
            max_value=len(states) - 1,
            value=1,
            format="Adım %d",
        )
        current_state = states[step_idx]
        st.markdown(f"**Mevcut Konum:** `{current_state['label']}`")
        render_svg_board(
            fen=current_state["fen"],
            last_move=current_state["last_move"],
            orientation_white=(result["solver_color"] == "Beyaz"),
        )
        if themes:
            st.caption(f"**Lichess Temaları:** `{themes}`")

    with right_col:
        st.subheader("Çok Aşamalı Model Çıktıları")
        model_df = pd.DataFrame(
            list(result["models"].items()), columns=["Model / Aşama", "Puan / Katkı"]
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        st.subheader("Zorluğu Belirleyen Yapısal Etkenler")
        piece_name = PIECE_NAMES_TR.get(feats["first_move_piece"], "-")
        drivers = [
            ("Başlangıç Yasal Hamle Sayısı (Dallanma)", f"{feats['legal_moves_start']} hamle"),
            ("Çözüm Uzunluğu", f"{feats['solver_moves_count']} çözücü hamlesi ({feats['solution_ply']} ply)"),
            ("İlk Hamle Karakteri", "Sessiz Hamle (Quiet)" if feats["first_move_is_quiet"] else ("Şah Çekiş" if feats["first_move_is_check"] else "Taş Alma")),
            ("İlk Hamle Taşı ve Mesafesi", f"{piece_name} ({feats['first_move_dist']} kare Manhattan)"),
            ("Taş Fedası (Sacrifice)", "Evet" if feats["is_sacrifice"] else "Hayır"),
            ("Mat ile Bitiş", "Evet" if feats["ends_in_mate"] else "Hayır"),
            ("Net Materyal Kazancı", f"{feats['material_gain']:+d} puan"),
            ("Rakip Şah Çevresi Baskısı", f"{feats['opp_king_attackers']} saldırı birimi"),
        ]
        drivers_df = pd.DataFrame(drivers, columns=["Yapısal Öznitelik", "Değer"])
        st.dataframe(drivers_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()