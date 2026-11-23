import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import torch
from maia2 import inference, model
from tqdm import tqdm

DB_PATH = Path("data") / "puzzles.db"
MAIA2_SAVE_ROOT = Path("maia2_models")
ELO_LEVELS = [1100, 1300, 1500, 1700, 1900]

CHUNK_READ_SIZE = 32_768
GPU_BATCH_SIZE = 1024

MAIA2_FEATURE_COLUMNS = [
    "puzzle_id",
    "maia_prob_1100",
    "maia_prob_1300",
    "maia_prob_1500",
    "maia_prob_1700",
    "maia_prob_1900",
    "maia_prob_mean",
    "maia_prob_slope",
    "maia_top1_prob_1500",
    "maia_top1_matches_1500",
    "maia_top1_matches_1900",
    "maia_move_rank_1500",
    "maia_prob_ratio_1500",
    "maia_entropy_1500",
    "maia_entropy_1900",
]

_WORKER_ALL_MOVES = None
_WORKER_ELO_DICT = None


def _init_worker(all_moves_dict, elo_dict):
    global _WORKER_ALL_MOVES, _WORKER_ELO_DICT
    _WORKER_ALL_MOVES = all_moves_dict
    _WORKER_ELO_DICT = elo_dict


def _preprocess_row_strict(row: tuple):
    puzzle_id, fen, moves_str = row
    tokens = moves_str.split()
    if len(tokens) < 2:
        raise ValueError(f"Geçersiz hamle dizisi: {moves_str}")

    board = chess.Board(fen)
    setup_mv = chess.Move.from_uci(tokens[0])
    board.push(setup_mv)

    puzzle_fen = board.fen()
    sol_uci = tokens[1]
    if board.turn == chess.BLACK:
        sol_uci = inference.mirror_move(sol_uci)

    move_idx = _WORKER_ALL_MOVES.get(sol_uci, -1)
    if move_idx < 0:
        raise ValueError(f"Hamle sözlükte bulunamadı: {sol_uci}")

    board_input, _, _, legal_mask = inference.preprocessing(
        puzzle_fen, 1500, 1500, _WORKER_ELO_DICT, _WORKER_ALL_MOVES
    )
    board_arr = board_input.numpy() if isinstance(board_input, torch.Tensor) else np.asarray(board_input)
    legal_arr = legal_mask.numpy() if isinstance(legal_mask, torch.Tensor) else np.asarray(legal_mask)
    return puzzle_id, board_arr, legal_arr.astype(np.float32), int(move_idx)


def _preprocess_row(row: tuple):
    try:
        return _preprocess_row_strict(row)
    except Exception:
        return None


def load_maia2_on_gpu(device: torch.device):
    MAIA2_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    last_err = None
    for model_type in ("rapid", "blitz"):
        try:
            print(f"Maia-2 ({model_type}) ağırlıkları yükleniyor...")
            m = model.from_pretrained(
                type=model_type,
                device=str(device),
                save_root=str(MAIA2_SAVE_ROOT),
            )
            m.eval()
            print(f"Maia-2 ({model_type}) başarıyla {device} cihazına yüklendi.")
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Maia-2 modeli yüklenemedi: {last_err}")


def verify_coordinate_alignment(all_moves_dict, elo_dict, conn):
    """Beyaz ve Siyah hamle sırası için ayna (mirror) indekslemesini zorunlu doğrular."""
    sample_rows = conn.execute("SELECT puzzle_id, fen, moves FROM raw_puzzles LIMIT 20;").fetchall()
    _init_worker(all_moves_dict, elo_dict)
    for r in sample_rows:
        _, _, legal_arr, move_idx = _preprocess_row_strict(r)
        if legal_arr[move_idx] <= 0.5:
            raise RuntimeError(
                f"Koordinat hizalama hatası tespit edildi! puzzle_id={r[0]}, move_idx={move_idx}"
            )
    print(f"Koordinat ve ayna (mirror) hizalama testi geçti ({len(sample_rows)}/20 örnek doğrulandı).")


@torch.inference_mode()
def evaluate_gpu_batch(maia_model, device, elo_cats, boards_np, legal_np, move_indices):
    B = len(move_indices)
    boards_t = torch.from_numpy(boards_np).to(device, non_blocking=True)
    legal_t = torch.from_numpy(legal_np).to(device, non_blocking=True)
    move_idx_t = torch.from_numpy(move_indices).to(device, non_blocking=True).unsqueeze(1)
    neg_inf = torch.tensor(-1e4, device=device, dtype=torch.float32)

    probs_per_elo = []
    top1_prob_1500 = None
    top1_match_1500 = None
    top1_match_1900 = None
    rank_1500 = None
    entropy_1500 = None
    entropy_1900 = None

    for elo_val, elo_cat in zip(ELO_LEVELS, elo_cats):
        elo_tensor = torch.full((B,), elo_cat, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = maia_model(boards_t, elo_tensor, elo_tensor)
            logits = out[0] if isinstance(out, (tuple, list)) else out

        logits_f32 = logits.float()
        masked_logits = torch.where(legal_t > 0.5, logits_f32, neg_inf)
        probs = torch.softmax(masked_logits, dim=-1)

        sol_prob = probs.gather(1, move_idx_t).squeeze(1)
        probs_per_elo.append(sol_prob)

        if elo_val in (1500, 1900):
            top1_prob, top1_idx = probs.max(dim=-1)
            matches = (top1_idx == move_idx_t.squeeze(1)).float()
            safe_p = torch.clamp(probs, min=1e-12)
            ent = -torch.sum(
                torch.where(legal_t > 0.5, probs * torch.log(safe_p), torch.zeros_like(probs)),
                dim=-1,
            )

            if elo_val == 1500:
                top1_prob_1500 = top1_prob
                top1_match_1500 = matches
                entropy_1500 = ent
                rank_1500 = (probs > sol_prob.unsqueeze(1)).sum(dim=-1).float() + 1.0
            else:
                top1_match_1900 = matches
                entropy_1900 = ent

    p1100, p1300, p1500, p1700, p1900 = probs_per_elo
    p_mean = (p1100 + p1300 + p1500 + p1700 + p1900) / 5.0
    p_slope = p1900 - p1100
    p_ratio_1500 = p1500 / torch.clamp(top1_prob_1500, min=1e-6)

    feats = torch.stack(
        [
            p1100,
            p1300,
            p1500,
            p1700,
            p1900,
            p_mean,
            p_slope,
            top1_prob_1500,
            top1_match_1500,
            top1_match_1900,
            rank_1500,
            p_ratio_1500,
            entropy_1500,
            entropy_1900,
        ],
        dim=1,
    )
    return feats.cpu().numpy()


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Maia-2 GPU Öznitelik Çıkarımı Başlatılıyor | Cihaz: {device}")

    raw_moves = inference.get_all_possible_moves()
    all_moves_dict = (
        {m: i for i, m in enumerate(raw_moves)}
        if isinstance(raw_moves, list)
        else raw_moves
    )
    elo_dict = inference.create_elo_dict()
    elo_cats = [inference.map_to_category(e, elo_dict) for e in ELO_LEVELS]

    maia_model = load_maia2_on_gpu(device)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    verify_coordinate_alignment(all_moves_dict, elo_dict, conn)

    cols_sql = ",\n            ".join(
        ["puzzle_id TEXT PRIMARY KEY"]
        + [f"{col} REAL NOT NULL" for col in MAIA2_FEATURE_COLUMNS[1:]]
    )
    cursor.execute(f"CREATE TABLE IF NOT EXISTS maia2_features (\n            {cols_sql}\n        );")
    conn.commit()

    already_done = cursor.execute("SELECT COUNT(*) FROM maia2_features;").fetchone()[0]
    total_all = cursor.execute("SELECT COUNT(*) FROM raw_puzzles;").fetchone()[0]
    remaining = total_all - already_done
    print(f"Toplam bulmaca: {total_all:,} | Tamamlanan: {already_done:,} | Kalan: {remaining:,}")

    if remaining <= 0:
        print("Tüm bulmacalar için Maia-2 öznitelikleri zaten çıkarılmış!")
        conn.close()
        return

    placeholders = ", ".join(["?"] * len(MAIA2_FEATURE_COLUMNS))
    insert_sql = f"INSERT OR IGNORE INTO maia2_features VALUES ({placeholders})"

    read_cursor = conn.cursor()
    read_cursor.execute("""
        SELECT r.puzzle_id, r.fen, r.moves
        FROM raw_puzzles r
        LEFT JOIN maia2_features m ON r.puzzle_id = m.puzzle_id
        WHERE m.puzzle_id IS NULL
    """)

    num_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(all_moves_dict, elo_dict),
    ) as executor, tqdm(total=remaining, unit="bulmaca", desc="Maia-2 RTX 3050") as pbar:
        while True:
            raw_chunk = read_cursor.fetchmany(CHUNK_READ_SIZE)
            if not raw_chunk:
                break

            preprocessed = [
                item
                for item in executor.map(_preprocess_row, raw_chunk, chunksize=512)
                if item is not None
            ]

            if not preprocessed:
                raise RuntimeError("Ön işleme bloğu hiçbir geçerli bulmaca üretemedi!")

            rows_to_insert = []
            for start_i in range(0, len(preprocessed), GPU_BATCH_SIZE):
                sub = preprocessed[start_i : start_i + GPU_BATCH_SIZE]
                p_ids = [x[0] for x in sub]
                boards_np = np.stack([x[1] for x in sub], axis=0)
                legal_np = np.stack([x[2] for x in sub], axis=0)
                move_indices = np.array([x[3] for x in sub], dtype=np.int64)

                batch_feats = evaluate_gpu_batch(
                    maia_model, device, elo_cats, boards_np, legal_np, move_indices
                )
                for pid, feat_vec in zip(p_ids, batch_feats):
                    rows_to_insert.append((pid, *[float(v) for v in feat_vec]))

            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            pbar.update(len(raw_chunk))

    elapsed = time.time() - start_time
    final_count = cursor.execute("SELECT COUNT(*) FROM maia2_features;").fetchone()[0]
    conn.close()
    print(f"\nMaia-2 öznitelik çıkarımı tamamlandı! Süre: {elapsed / 60:.1f} dk | Toplam kayıt: {final_count:,}")


if __name__ == "__main__":
    main()