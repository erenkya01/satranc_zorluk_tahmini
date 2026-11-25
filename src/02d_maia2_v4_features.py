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

CHUNK_READ_SIZE = 24_576
GPU_BATCH_SIZE = 1024

V4_COLUMNS = [
    "puzzle_id",
    # Ply 1 Rapid 1900 derin & tuzak sinyalleri
    "maia_top1_prob_1900",
    "maia_top2_prob_1900",
    "maia_top1_top2_gap_1900",
    "maia_move_rank_1900",
    "maia_prob_ratio_1900",
    # Ply 1 Blitz vs Rapid bilissel tuzak sinyalleri
    "maia_blitz_prob_1500",
    "maia_blitz_prob_1900",
    "maia_rapid_minus_blitz_1900",
    "maia_blitz_trap_gap_1900",
    "maia_blitz_rapid_top1_agree",
    # Ply 2 Rakip savunma dallanma karmasikligi
    "opp_p2_entropy_1900",
    "opp_p2_top1_prob_1900",
    "opp_p2_sol_prob_1900",
    # Ply 5 (3. Cozucu Hamlesi) derin hesaplama sinyalleri
    "has_ply5",
    "maia_p5_prob_1500",
    "maia_p5_prob_1900",
    "maia_p5_rank_1900",
]

_WORKER_ALL_MOVES = None
_WORKER_ELO_DICT = None
_DUMMY_BOARD = np.zeros((18, 8, 8), dtype=np.float32)
_DUMMY_LEGAL = np.zeros((1880,), dtype=np.float32)
_DUMMY_LEGAL[0] = 1.0


def _init_worker(all_moves_dict, elo_dict):
    global _WORKER_ALL_MOVES, _WORKER_ELO_DICT
    _WORKER_ALL_MOVES = all_moves_dict
    _WORKER_ELO_DICT = elo_dict


def _encode_board_and_move(board: chess.Board, move_uci: str):
    mirrored = inference.mirror_move(move_uci) if board.turn == chess.BLACK else move_uci
    move_idx = _WORKER_ALL_MOVES.get(mirrored, -1)
    if move_idx < 0:
        raise ValueError(f"Hamle sözlükte yok: {mirrored}")

    b_in, _, _, l_mask = inference.preprocessing(
        board.fen(), 1500, 1500, _WORKER_ELO_DICT, _WORKER_ALL_MOVES
    )
    b_arr = b_in.numpy() if isinstance(b_in, torch.Tensor) else np.asarray(b_in)
    l_arr = l_mask.numpy() if isinstance(l_mask, torch.Tensor) else np.asarray(l_mask)
    return b_arr.astype(np.float32), l_arr.astype(np.float32), int(move_idx)


def _preprocess_v4_strict(row: tuple):
    puzzle_id, fen, moves_str = row
    tokens = moves_str.split()
    if len(tokens) < 2:
        raise ValueError(f"Geçersiz hamle dizisi: {moves_str}")

    board = chess.Board(fen)
    # Rakibin hazırlık hatası
    board.push(chess.Move.from_uci(tokens[0]))

    # 1. Ply 1 (Çözücü 1. hamlesi)
    b1, l1, idx1 = _encode_board_and_move(board, tokens[1])
    board.push(chess.Move.from_uci(tokens[1]))

    # 2. Ply 2 (Rakip savunması - solution_ply >= 2 ise)
    if len(tokens) >= 3:
        has_p2 = 1.0
        b2, l2, idx2 = _encode_board_and_move(board, tokens[2])
        board.push(chess.Move.from_uci(tokens[2]))
    else:
        has_p2 = 0.0
        b2, l2, idx2 = _DUMMY_BOARD, _DUMMY_LEGAL, 0

    # 3. Ply 5 (Çözücü 3. hamlesi - solution_ply >= 5 ise tokens[5] mevcuttur)
    if len(tokens) >= 6:
        board.push(chess.Move.from_uci(tokens[3]))  # Çözücü 2. hamlesi (Ply 3)
        board.push(chess.Move.from_uci(tokens[4]))  # Rakip 2. savunması (Ply 4)
        has_p5 = 1.0
        b5, l5, idx5 = _encode_board_and_move(board, tokens[5])
    else:
        has_p5 = 0.0
        b5, l5, idx5 = _DUMMY_BOARD, _DUMMY_LEGAL, 0

    return puzzle_id, b1, l1, idx1, has_p2, b2, l2, idx2, has_p5, b5, l5, idx5


def _preprocess_v4_safe(row: tuple):
    try:
        return _preprocess_v4_strict(row)
    except Exception:
        return None


def verify_v4_alignment(all_moves_dict, elo_dict, conn):
    sample_rows = conn.execute("""
        SELECT r.puzzle_id, r.fen, r.moves
        FROM raw_puzzles r
        INNER JOIN structural_features sf ON r.puzzle_id = sf.puzzle_id
        WHERE sf.solution_ply >= 5
        LIMIT 20;
    """).fetchall()
    _init_worker(all_moves_dict, elo_dict)
    for r in sample_rows:
        _, _, l1, idx1, has_p2, _, l2, idx2, has_p5, _, l5, idx5 = _preprocess_v4_strict(r)
        if l1[idx1] <= 0.5 or (has_p2 > 0.5 and l2[idx2] <= 0.5) or (has_p5 > 0.5 and l5[idx5] <= 0.5):
            raise RuntimeError(f"v4 koordinat/ayna hizalama hatası! puzzle_id={r[0]}")
    print(f"v4 (Ply 1 + Ply 2 + Ply 5) hizalama testi geçti ({len(sample_rows)}/20 örnek doğrulandı).")


def _forward_probs(net, device, boards_t, legal_t, elo_cat: int, neg_inf):
    B = boards_t.shape[0]
    elo_t = torch.full((B,), elo_cat, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        out = net(boards_t, elo_t, elo_t)
        logits = out[0] if isinstance(out, (tuple, list)) else out
    masked = torch.where(legal_t > 0.5, logits.float(), neg_inf)
    return torch.softmax(masked, dim=-1)


@torch.inference_mode()
def evaluate_v4_batch(
    rapid_net,
    blitz_net,
    device,
    elo_1500_cat: int,
    elo_1900_cat: int,
    b1_np,
    l1_np,
    idx1_np,
    has_p2_np,
    b2_np,
    l2_np,
    idx2_np,
    has_p5_np,
    b5_np,
    l5_np,
    idx5_np,
):
    B = len(idx1_np)
    neg_inf = torch.tensor(-1e4, device=device, dtype=torch.float32)

    # --- 1. Ply 1 Rapid 1900 + Blitz 1500/1900 ---
    b1_t = torch.from_numpy(b1_np).to(device, non_blocking=True)
    l1_t = torch.from_numpy(l1_np).to(device, non_blocking=True)
    idx1_t = torch.from_numpy(idx1_np).to(device, non_blocking=True).unsqueeze(1)

    p1_rapid_1900 = _forward_probs(rapid_net, device, b1_t, l1_t, elo_1900_cat, neg_inf)
    p1_blitz_1500 = _forward_probs(blitz_net, device, b1_t, l1_t, elo_1500_cat, neg_inf)
    p1_blitz_1900 = _forward_probs(blitz_net, device, b1_t, l1_t, elo_1900_cat, neg_inf)

    sol_r1900 = p1_rapid_1900.gather(1, idx1_t).squeeze(1)
    top2_vals, top2_idxs = torch.topk(p1_rapid_1900, k=2, dim=-1)
    top1_r1900 = top2_vals[:, 0]
    top2_r1900 = top2_vals[:, 1]
    gap_r1900 = top1_r1900 - top2_r1900
    rank_r1900 = (p1_rapid_1900 > sol_r1900.unsqueeze(1)).sum(dim=-1).float() + 1.0
    ratio_r1900 = sol_r1900 / torch.clamp(top1_r1900, min=1e-6)

    sol_b1500 = p1_blitz_1500.gather(1, idx1_t).squeeze(1)
    sol_b1900 = p1_blitz_1900.gather(1, idx1_t).squeeze(1)
    rapid_minus_blitz = sol_r1900 - sol_b1900
    b1900_top1_prob, b1900_top1_idx = p1_blitz_1900.max(dim=-1)
    blitz_trap_gap = b1900_top1_prob - sol_b1900
    top1_agree = (top2_idxs[:, 0] == b1900_top1_idx).float()

    del b1_t, l1_t, p1_rapid_1900, p1_blitz_1500, p1_blitz_1900

    # --- 2. Ply 2 Rakip Savunma Dallanması (Sadece aktif satırlarda) ---
    has_p2_t = torch.from_numpy(has_p2_np).to(device, non_blocking=True)
    opp_ent_1900 = torch.zeros(B, device=device, dtype=torch.float32)
    opp_top1_1900 = torch.ones(B, device=device, dtype=torch.float32)
    opp_sol_1900 = torch.ones(B, device=device, dtype=torch.float32)

    active_p2 = np.nonzero(has_p2_np > 0.5)[0]
    if len(active_p2) > 0:
        b2_t = torch.from_numpy(b2_np[active_p2]).to(device, non_blocking=True)
        l2_t = torch.from_numpy(l2_np[active_p2]).to(device, non_blocking=True)
        idx2_t = torch.from_numpy(idx2_np[active_p2]).to(device, non_blocking=True).unsqueeze(1)

        p2_probs = _forward_probs(rapid_net, device, b2_t, l2_t, elo_1900_cat, neg_inf)
        safe_p2 = torch.clamp(p2_probs, min=1e-12)
        ent2 = -torch.sum(
            torch.where(l2_t > 0.5, p2_probs * torch.log(safe_p2), torch.zeros_like(p2_probs)),
            dim=-1,
        )
        t1_2, _ = p2_probs.max(dim=-1)
        sol_2 = p2_probs.gather(1, idx2_t).squeeze(1)

        act_p2_t = torch.from_numpy(active_p2).to(device)
        opp_ent_1900[act_p2_t] = ent2
        opp_top1_1900[act_p2_t] = t1_2
        opp_sol_1900[act_p2_t] = sol_2
        del b2_t, l2_t, p2_probs

    # --- 3. Ply 5 (3. Çözücü Hamlesi - Sadece solution_ply >= 5 olanlarda) ---
    has_p5_t = torch.from_numpy(has_p5_np).to(device, non_blocking=True)
    p5_1500 = torch.ones(B, device=device, dtype=torch.float32)
    p5_1900 = torch.ones(B, device=device, dtype=torch.float32)
    p5_rank_1900 = torch.ones(B, device=device, dtype=torch.float32)

    active_p5 = np.nonzero(has_p5_np > 0.5)[0]
    if len(active_p5) > 0:
        b5_t = torch.from_numpy(b5_np[active_p5]).to(device, non_blocking=True)
        l5_t = torch.from_numpy(l5_np[active_p5]).to(device, non_blocking=True)
        idx5_t = torch.from_numpy(idx5_np[active_p5]).to(device, non_blocking=True).unsqueeze(1)

        probs5_1500 = _forward_probs(rapid_net, device, b5_t, l5_t, elo_1500_cat, neg_inf)
        probs5_1900 = _forward_probs(rapid_net, device, b5_t, l5_t, elo_1900_cat, neg_inf)

        s5_1500 = probs5_1500.gather(1, idx5_t).squeeze(1)
        s5_1900 = probs5_1900.gather(1, idx5_t).squeeze(1)
        r5_1900 = (probs5_1900 > s5_1900.unsqueeze(1)).sum(dim=-1).float() + 1.0

        act_p5_t = torch.from_numpy(active_p5).to(device)
        p5_1500[act_p5_t] = s5_1500
        p5_1900[act_p5_t] = s5_1900
        p5_rank_1900[act_p5_t] = r5_1900
        del b5_t, l5_t, probs5_1500, probs5_1900

    feats = torch.stack(
        [
            top1_r1900,
            top2_r1900,
            gap_r1900,
            rank_r1900,
            ratio_r1900,
            sol_b1500,
            sol_b1900,
            rapid_minus_blitz,
            blitz_trap_gap,
            top1_agree,
            opp_ent_1900,
            opp_top1_1900,
            opp_sol_1900,
            has_p5_t,
            p5_1500,
            p5_1900,
            p5_rank_1900,
        ],
        dim=1,
    )
    return feats.cpu().numpy()


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Maia-2 v4 (Blitz-Rapid Tuzak + Ply 2 Savunma + Ply 5) GPU Çıkarımı | Cihaz: {device}")

    raw_moves = inference.get_all_possible_moves()
    all_moves_dict = (
        {m: i for i, m in enumerate(raw_moves)}
        if isinstance(raw_moves, list)
        else raw_moves
    )
    elo_dict = inference.create_elo_dict()
    elo_1500_cat = inference.map_to_category(1500, elo_dict)
    elo_1900_cat = inference.map_to_category(1900, elo_dict)

    MAIA2_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    print("Maia-2 Rapid ve Blitz ağları GPU'ya yükleniyor...")
    rapid_net = model.from_pretrained(type="rapid", device=str(device), save_root=str(MAIA2_SAVE_ROOT))
    blitz_net = model.from_pretrained(type="blitz", device=str(device), save_root=str(MAIA2_SAVE_ROOT))
    rapid_net.eval()
    blitz_net.eval()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    verify_v4_alignment(all_moves_dict, elo_dict, conn)

    col_defs = ",\n            ".join([f"{c} REAL NOT NULL" for c in V4_COLUMNS[1:]])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS maia2_v4_features (
            puzzle_id TEXT PRIMARY KEY,
            {col_defs}
        );
    """)
    conn.commit()

    already_done = cursor.execute("SELECT COUNT(*) FROM maia2_v4_features;").fetchone()[0]
    total_all = cursor.execute("SELECT COUNT(*) FROM raw_puzzles;").fetchone()[0]
    remaining = total_all - already_done
    print(f"Toplam bulmaca: {total_all:,} | Hazır: {already_done:,} | GPU'da işlenecek: {remaining:,}")

    if remaining <= 0:
        print("Tüm bulmacalar için v4 öznitelikleri tamamlanmış!")
        conn.close()
        return

    placeholders = ", ".join(["?"] * len(V4_COLUMNS))
    insert_sql = f"INSERT OR IGNORE INTO maia2_v4_features VALUES ({placeholders})"

    read_cursor = conn.cursor()
    read_cursor.execute("""
        SELECT r.puzzle_id, r.fen, r.moves
        FROM raw_puzzles r
        LEFT JOIN maia2_v4_features m4 ON r.puzzle_id = m4.puzzle_id
        WHERE m4.puzzle_id IS NULL
    """)

    num_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(all_moves_dict, elo_dict),
    ) as executor, tqdm(total=remaining, unit="bulmaca", desc="Maia-2 v4 GPU") as pbar:
        while True:
            raw_chunk = read_cursor.fetchmany(CHUNK_READ_SIZE)
            if not raw_chunk:
                break

            preprocessed = [
                item
                for item in executor.map(_preprocess_v4_safe, raw_chunk, chunksize=384)
                if item is not None
            ]

            rows_to_insert = []
            valid_ids = {x[0] for x in preprocessed}
            for r in raw_chunk:
                if r[0] not in valid_ids:
                    rows_to_insert.append(
                        (r[0], 0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
                    )

            for start_i in range(0, len(preprocessed), GPU_BATCH_SIZE):
                sub = preprocessed[start_i : start_i + GPU_BATCH_SIZE]
                p_ids = [x[0] for x in sub]

                b1_np = np.stack([x[1] for x in sub], axis=0)
                l1_np = np.stack([x[2] for x in sub], axis=0)
                idx1_np = np.array([x[3] for x in sub], dtype=np.int64)

                has_p2_np = np.array([x[4] for x in sub], dtype=np.float32)
                b2_np = np.stack([x[5] for x in sub], axis=0)
                l2_np = np.stack([x[6] for x in sub], axis=0)
                idx2_np = np.array([x[7] for x in sub], dtype=np.int64)

                has_p5_np = np.array([x[8] for x in sub], dtype=np.float32)
                b5_np = np.stack([x[9] for x in sub], axis=0)
                l5_np = np.stack([x[10] for x in sub], axis=0)
                idx5_np = np.array([x[11] for x in sub], dtype=np.int64)

                batch_feats = evaluate_v4_batch(
                    rapid_net,
                    blitz_net,
                    device,
                    elo_1500_cat,
                    elo_1900_cat,
                    b1_np,
                    l1_np,
                    idx1_np,
                    has_p2_np,
                    b2_np,
                    l2_np,
                    idx2_np,
                    has_p5_np,
                    b5_np,
                    l5_np,
                    idx5_np,
                )
                for pid, feat_vec in zip(p_ids, batch_feats):
                    rows_to_insert.append((pid, *[float(v) for v in feat_vec]))

            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            pbar.update(len(raw_chunk))

    elapsed = time.time() - start_time
    final_count = cursor.execute("SELECT COUNT(*) FROM maia2_v4_features;").fetchone()[0]
    conn.close()
    print(f"\nMaia-2 v4 öznitelik çıkarımı tamamlandı! Süre: {elapsed / 60:.1f} dk | Toplam kayıt: {final_count:,}")


if __name__ == "__main__":
    main()