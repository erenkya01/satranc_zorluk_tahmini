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
def get_predictor(model_dir_str: str):
    return PuzzleDifficultyPredictor(model_dir=model_dir_str)


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


def fetch_puzzle_by_id(puzzle_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT puzzle_id, fen, moves, rating, themes
        FROM raw_puzzles
        WHERE puzzle_id = ?
        LIMIT 1
        """,
        (puzzle_id.strip(),),
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
        "Çok Adımlı Maia-2 İnsan Politikası + 84 Öznitelikli Hibrit GBDT + Artık Öğrenme"
    )

    with st.sidebar:
        st.header("Model Mimarisi Seçimi")
        model_options = {}
        if (ROOT_DIR / "models_v3_full").exists():
            model_options["v3: Tam Hibrit (84 Özellik + Ply 1&3 Maia-2 | 241.4 RMSE)"] = str(
                ROOT_DIR / "models_v3_full"
            )
        if (ROOT_DIR / "models_hybrid_maia2").exists():
            model_options["v2: Hibrit (70 Özellik + Ply 1 Maia-2 | 261.7 RMSE)"] = str(
                ROOT_DIR / "models_hybrid_maia2"
            )
        if (ROOT_DIR / "models_structural_only").exists():
            model_options["v1: Saf Yapısal (56 Özellik | 282.7 RMSE)"] = str(
                ROOT_DIR / "models_structural_only"
            )

        selected_model_label = st.selectbox(
            "Aktif Tahmin Modeli:",
            list(model_options.keys()),
        )
        predictor = get_predictor(model_options[selected_model_label])

        st.divider()
        st.header("Veri Kaynağı Seçimi")
        mode = st.radio(
            "Çalışma Modu:",
            [
                "Test Kümesinden Bulmaca Çek",
                "Bulmaca ID ile Sorgula",
                "Özel Pozisyon Gir (FEN + PGN/UCI)",
            ],
        )

        st.divider()
        actual_rating = None
        puzzle_id = None
        themes = ""

        if mode == "Test Kümesinden Bulmaca Çek":
            tier_preset = st.selectbox(
                "Hızlı Zorluk Filtresi:",
                [
                    "Tümü (1000 – 2200)",
                    "Başlangıç (< 1100)",
                    "Orta (1100 – 1500)",
                    "İleri (1500 – 1900)",
                    "Usta Adayı (1900 – 2300)",
                    "Usta Seviyesi (2300+)",
                    "Özel Aralık",
                ],
            )
            preset_bounds = {
                "Tümü (1000 – 2200)": (1000, 2200),
                "Başlangıç (< 1100)": (600, 1099),
                "Orta (1100 – 1500)": (1100, 1500),
                "İleri (1500 – 1900)": (1500, 1900),
                "Usta Adayı (1900 – 2300)": (1900, 2300),
                "Usta Seviyesi (2300+)": (2300, 2900),
            }
            if tier_preset == "Özel Aralık":
                rating_range = st.slider(
                    "Hedef Zorluk Aralığı (Glicko-2):", 600, 2900, (1000, 2200), step=100
                )
            else:
                rating_range = preset_bounds[tier_preset]

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
                st.info(
                    f"**Bulmaca ID:** `{puzzle_id}`\n\n"
                    f"**Gerçek Lichess Puanı:** `{actual_rating}` ({difficulty_tier(actual_rating)})"
                )
            else:
                fen_input = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
                moves_input = "g8f6 h5f7"

        elif mode == "Bulmaca ID ile Sorgula":
            query_id = st.text_input("Lichess Bulmaca ID:", value="00Kmw")
            row = fetch_puzzle_by_id(query_id) if DB_PATH.exists() else None
            if row:
                puzzle_id, fen_input, moves_input, actual_rating, themes = row
                st.info(
                    f"**Bulmaca ID:** `{puzzle_id}`\n\n"
                    f"**Gerçek Lichess Puanı:** `{actual_rating}` ({difficulty_tier(actual_rating)})"
                )
            else:
                st.warning("Bu ID veritabanında bulunamadı.")
                fen_input = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
                moves_input = "g8f6 h5f7"
        else:
            st.markdown("Dışarıdan herhangi bir FEN ve çözüm hamlelerini (UCI veya PGN) yapıştırın:")
            fen_input = st.text_input(
                "FEN Pozisyonu:",
                value="8/8/3k1K1p/4P1pP/6P1/1p6/8/8 b - - 0 1",
            )
            moves_input = st.text_area(
                "Hamle Dizisi (1. hamle rakip hatası, devamı çözüm):",
                value="d6d7 e5e6 d7e8 f6g7 b3b2 g7h6",
                height=90,
            )

        st.divider()
        st.caption(f"Aktif Dizin: `{predictor.model_dir.name}`")

    try:
        result = predictor.predict(fen_input, moves_input, puzzle_id=puzzle_id)
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
    col_m2.metric(
        "Belirsizlik Aralığı",
        f"{result['ci_low']} – {result['ci_high']}",
        delta=f"±{margin} Elo",
        delta_color="off",
    )
    col_m3.metric("Tahmini Zorluk Sınıfı", difficulty_tier(pred_rating))

    if actual_rating is not None:
        diff = pred_rating - actual_rating
        col_m4.metric(
            f"Gerçek Sınıf ({difficulty_tier(actual_rating).split()[0]})",
            f"{actual_rating} Elo",
            delta=f"Fark: {diff:+d} Elo",
        )
    else:
        col_m4.metric("Çözüm Derinliği", f"{feats['solver_moves_count']} Hamle ({feats['solution_ply']} ply)")

    st.divider()

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

        if result["is_hybrid"] and "maia_prob_1500" in feats:
            st.subheader("Maia-2 İnsan Politikası Analizi")
            maia_rows = [
                ("1. Hamle (Ply 1) İnsan Olasılığı (1500 Elo)", f"%{feats['maia_prob_1500'] * 100:.1f} (Sıra: {int(feats['maia_move_rank_1500'])})"),
                ("1. Hamle (Ply 1) Usta Olasılığı (1900 Elo)", f"%{feats['maia_prob_1900'] * 100:.1f}"),
            ]
            if result.get("is_v3") and feats.get("has_ply3", 0.0) >= 0.5:
                maia_rows.extend(
                    [
                        ("2. Hamle (Ply 3) İnsan Olasılığı (1500 Elo)", f"%{feats['maia_p3_prob_1500'] * 100:.1f} (Sıra: {int(feats['maia_p3_rank_1500'])})"),
                        ("Zincirleme Bulunma Olasılığı (Ply 1 × Ply 3)", f"%{feats['maia_chain_prob_1500'] * 100:.1f}"),
                    ]
                )
            maia_rows.extend(
                [
                    ("Yetenek Ayrım Eğimi (1900 − 1100)", f"{feats['maia_prob_slope'] * 100:+.1f} puan"),
                    ("Karar Belirsizliği Entropisi (1500 Elo)", f"{feats['maia_entropy_1500']:.2f} nat"),
                ]
            )
            maia_df = pd.DataFrame(maia_rows, columns=["Maia-2 Bilişsel Öznitelik", "Değer"])
            st.dataframe(maia_df, use_container_width=True, hide_index=True)

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