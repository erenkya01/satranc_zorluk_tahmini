import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import chess
from tqdm import tqdm

DB_PATH = Path("data") / "puzzles.db"
BATCH_SIZE = 50_000
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10,
}

FEATURE_COLUMNS = [
    "puzzle_id",
    "solution_ply",
    "solver_moves_count",
    "ends_in_mate",
    "is_mate_in_1",
    "total_pieces",
    "solver_pawns",
    "solver_minors",
    "solver_rooks",
    "solver_queens",
    "opp_pawns",
    "opp_minors",
    "opp_rooks",
    "opp_queens",
    "solver_material",
    "opp_material",
    "material_diff_start",
    "material_diff_end",
    "material_gain",
    "game_phase",
    "is_endgame",
    "solver_turn_white",
    "in_check_at_start",
    "setup_is_capture",
    "setup_is_check",
    "first_move_is_check",
    "first_move_is_capture",
    "first_move_is_quiet",
    "captured_piece_val_1",
    "is_sacrifice",
    "first_move_piece",
    "first_move_dist",
    "first_move_is_fork",
    "first_move_attacks_queen",
    "first_move_attacks_king_zone",
    "solver_checks",
    "solver_captures",
    "solver_promotions",
    "total_quiet_solver_moves",
    "second_solver_move_is_quiet",
    "max_piece_value_moved",
    "distinct_solver_pieces_moved",
    "legal_moves_start",
    "avail_checks_start",
    "avail_captures_start",
    "quiet_despite_checks",
    "opp_legal_moves_ply2",
    "opp_is_forced_ply2",
    "solver_legal_moves_ply3",
    "solver_king_attackers",
    "opp_king_attackers",
    "opp_king_mobility",
    "solver_pins_count",
    "opp_pins_count",
    "total_attacked_opp_pieces",
    "undefended_opp_pieces",
    "undefended_solver_pieces",
]


def _material_and_counts(board: chess.Board, color: chess.Color):
    pawns = len(board.pieces(chess.PAWN, color))
    knights = len(board.pieces(chess.KNIGHT, color))
    bishops = len(board.pieces(chess.BISHOP, color))
    rooks = len(board.pieces(chess.ROOK, color))
    queens = len(board.pieces(chess.QUEEN, color))
    minors = knights + bishops
    material = pawns * 1 + minors * 3 + rooks * 5 + queens * 9
    phase = minors * 1 + rooks * 2 + queens * 4
    return pawns, minors, rooks, queens, material, phase


def _king_zone_pressure(board: chess.Board, king_color: chess.Color) -> int:
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    attacker_color = not king_color
    pressure = len(board.attackers(attacker_color, king_sq))
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        pressure += len(board.attackers(attacker_color, sq))
    return pressure


def extract_single_puzzle(row: tuple) -> tuple | None:
    puzzle_id, fen, moves_str = row
    try:
        move_tokens = moves_str.split()
        if len(move_tokens) < 2:
            return None

        board = chess.Board(fen)

        # 1. Rakibin hazırlık hatası
        setup_move = chess.Move.from_uci(move_tokens[0])
        setup_is_capture = int(board.is_capture(setup_move))
        board.push(setup_move)
        setup_is_check = int(board.is_check())

        # Gerçek bulmaca başlangıç pozisyonu
        solver_color = board.turn
        opp_color = not solver_color
        solver_turn_white = int(solver_color == chess.WHITE)

        s_p, s_m, s_r, s_q, s_mat_start, s_phase = _material_and_counts(board, solver_color)
        o_p, o_m, o_r, o_q, o_mat_start, o_phase = _material_and_counts(board, opp_color)
        total_pieces = s_p + s_m + s_r + s_q + o_p + o_m + o_r + o_q + 2
        mat_diff_start = s_mat_start - o_mat_start
        game_phase = min(24, s_phase + o_phase)
        is_endgame = int(game_phase <= 6)

        in_check_at_start = int(board.is_check())

        # Aday hamleler ve dallanma genişliği
        legal_moves = list(board.legal_moves)
        legal_moves_start = len(legal_moves)
        avail_checks_start = 0
        avail_captures_start = 0
        for lm in legal_moves:
            if board.is_capture(lm):
                avail_captures_start += 1
            if board.gives_check(lm):
                avail_checks_start += 1

        # Şah baskısı ve rakip şah kaçış alanı
        solver_king_attackers = _king_zone_pressure(board, solver_color)
        opp_king_attackers = _king_zone_pressure(board, opp_color)
        opp_king_sq = board.king(opp_color)
        opp_king_zone_bb = 0
        opp_king_mobility = 0
        if opp_king_sq is not None:
            opp_king_zone_bb = chess.BB_KING_ATTACKS[opp_king_sq] | chess.BB_SQUARES[opp_king_sq]
            opp_occ = board.occupied_co[opp_color]
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[opp_king_sq]):
                if not (opp_occ & chess.BB_SQUARES[sq]) and not board.is_attacked_by(solver_color, sq):
                    opp_king_mobility += 1

        # Açmaz (pin) ve savunmasız taş sayıları
        solver_pins_count = 0
        total_attacked_opp_pieces = 0
        undefended_opp_pieces = 0
        for sq in chess.SquareSet(board.occupied_co[opp_color]):
            if board.is_pinned(opp_color, sq):
                solver_pins_count += 1
            if board.is_attacked_by(solver_color, sq):
                total_attacked_opp_pieces += 1
                if not board.is_attacked_by(opp_color, sq):
                    undefended_opp_pieces += 1

        opp_pins_count = 0
        undefended_solver_pieces = 0
        for sq in chess.SquareSet(board.occupied_co[solver_color]):
            if board.is_pinned(solver_color, sq):
                opp_pins_count += 1
            if board.is_attacked_by(opp_color, sq) and not board.is_attacked_by(solver_color, sq):
                undefended_solver_pieces += 1

        solution_tokens = move_tokens[1:]
        solution_ply = len(solution_tokens)
        solver_moves_count = (solution_ply + 1) // 2

        first_move_is_check = 0
        first_move_is_capture = 0
        first_move_is_quiet = 0
        captured_piece_val_1 = 0
        first_move_piece = 0
        first_move_dist = 0
        first_move_is_fork = 0
        first_move_attacks_queen = 0
        first_move_attacks_king_zone = 0
        is_sacrifice = 0

        solver_checks = 0
        solver_captures = 0
        solver_promotions = 0
        total_quiet_solver_moves = 0
        second_solver_move_is_quiet = 0
        max_piece_value_moved = 0
        distinct_solver_pieces_moved = 0
        last_solver_to_sq = None

        opp_legal_moves_ply2 = 0
        opp_is_forced_ply2 = 0
        solver_legal_moves_ply3 = 0

        for idx, uci in enumerate(solution_tokens):
            mv = chess.Move.from_uci(uci)
            is_solver_turn = (idx % 2 == 0)

            is_cap = board.is_capture(mv)
            is_promo = mv.promotion is not None
            moving_piece = board.piece_at(mv.from_square)
            moving_type = moving_piece.piece_type if moving_piece else 0

            if is_solver_turn:
                p_val = PIECE_VALUES.get(moving_type, 0)
                if p_val > max_piece_value_moved:
                    max_piece_value_moved = p_val
                if mv.from_square != last_solver_to_sq:
                    distinct_solver_pieces_moved += 1
                last_solver_to_sq = mv.to_square

            if idx == 0:
                first_move_piece = moving_type
                f1, r1 = chess.square_file(mv.from_square), chess.square_rank(mv.from_square)
                f2, r2 = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
                first_move_dist = abs(f1 - f2) + abs(r1 - r2)
                first_move_is_capture = int(is_cap)
                if is_cap:
                    if board.is_en_passant(mv):
                        captured_piece_val_1 = 1
                    else:
                        cap_p = board.piece_at(mv.to_square)
                        captured_piece_val_1 = PIECE_VALUES.get(cap_p.piece_type, 0) if cap_p else 0

            board.push(mv)
            is_chk = board.is_check()
            is_quiet = not (is_cap or is_chk or is_promo)

            if idx == 0:
                first_move_is_check = int(is_chk)
                first_move_is_quiet = int(is_quiet)

                # Çatal, vezir tehdidi ve şah bölgesi saldırısı
                attacks_bb = int(board.attacks(mv.to_square))
                opp_occ_bb = int(board.occupied_co[opp_color])
                attacked_targets = chess.popcount(attacks_bb & opp_occ_bb)
                first_move_is_fork = int(attacked_targets >= 2)
                first_move_attacks_queen = int(bool(attacks_bb & int(board.pieces(chess.QUEEN, opp_color))))
                first_move_attacks_king_zone = int(
                    bool((chess.BB_SQUARES[mv.to_square] | attacks_bb) & opp_king_zone_bb)
                )

                # Rakibin cevaptaki zorunluluk (forcing) derecesi
                opp_legal_moves_ply2 = board.legal_moves.count()
                opp_is_forced_ply2 = int(opp_legal_moves_ply2 == 1)

            if idx == 1:
                _, _, _, _, s_mat_mid, _ = _material_and_counts(board, solver_color)
                _, _, _, _, o_mat_mid, _ = _material_and_counts(board, opp_color)
                if (s_mat_mid - o_mat_mid) < mat_diff_start:
                    is_sacrifice = 1
                if solution_ply >= 3:
                    solver_legal_moves_ply3 = board.legal_moves.count()

            if idx == 2:
                second_solver_move_is_quiet = int(is_quiet)

            if is_solver_turn:
                solver_checks += int(is_chk)
                solver_captures += int(is_cap)
                solver_promotions += int(is_promo)
                total_quiet_solver_moves += int(is_quiet)

        quiet_despite_checks = int(
            first_move_is_quiet == 1 and (avail_checks_start + avail_captures_start > 0)
        )

        ends_in_mate = int(board.is_checkmate())
        is_mate_in_1 = int(ends_in_mate == 1 and solution_ply == 1)

        _, _, _, _, s_mat_end, _ = _material_and_counts(board, solver_color)
        _, _, _, _, o_mat_end, _ = _material_and_counts(board, opp_color)
        mat_diff_end = s_mat_end - o_mat_end
        material_gain = mat_diff_end - mat_diff_start

        return (
            puzzle_id,
            solution_ply,
            solver_moves_count,
            ends_in_mate,
            is_mate_in_1,
            total_pieces,
            s_p,
            s_m,
            s_r,
            s_q,
            o_p,
            o_m,
            o_r,
            o_q,
            s_mat_start,
            o_mat_start,
            mat_diff_start,
            mat_diff_end,
            material_gain,
            game_phase,
            is_endgame,
            solver_turn_white,
            in_check_at_start,
            setup_is_capture,
            setup_is_check,
            first_move_is_check,
            first_move_is_capture,
            first_move_is_quiet,
            captured_piece_val_1,
            is_sacrifice,
            first_move_piece,
            first_move_dist,
            first_move_is_fork,
            first_move_attacks_queen,
            first_move_attacks_king_zone,
            solver_checks,
            solver_captures,
            solver_promotions,
            total_quiet_solver_moves,
            second_solver_move_is_quiet,
            max_piece_value_moved,
            distinct_solver_pieces_moved,
            legal_moves_start,
            avail_checks_start,
            avail_captures_start,
            quiet_despite_checks,
            opp_legal_moves_ply2,
            opp_is_forced_ply2,
            solver_legal_moves_ply3,
            solver_king_attackers,
            opp_king_attackers,
            opp_king_mobility,
            solver_pins_count,
            opp_pins_count,
            total_attacked_opp_pieces,
            undefended_opp_pieces,
            undefended_solver_pieces,
        )
    except Exception:
        return None


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı! Önce 01_build_db.py çalıştırın.")

    num_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"v2 Özellik Çıkarımı (56 öznitelik) başlatılıyor | Çekirdek: {num_workers}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    cursor.execute("DROP TABLE IF EXISTS structural_features;")
    cols_sql = ",\n            ".join(
        ["puzzle_id TEXT PRIMARY KEY"]
        + [f"{col} INTEGER NOT NULL" for col in FEATURE_COLUMNS[1:]]
    )
    cursor.execute(f"CREATE TABLE structural_features (\n            {cols_sql}\n        );")
    conn.commit()

    total_rows = cursor.execute("SELECT COUNT(*) FROM raw_puzzles;").fetchone()[0]
    placeholders = ", ".join(["?"] * len(FEATURE_COLUMNS))
    insert_sql = f"INSERT INTO structural_features VALUES ({placeholders})"

    read_cursor = conn.cursor()
    read_cursor.execute("SELECT puzzle_id, fen, moves FROM raw_puzzles;")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor, tqdm(
        total=total_rows, unit="bulmaca", desc="v2 Öznitelik Çıkarımı"
    ) as pbar:
        while True:
            batch = read_cursor.fetchmany(BATCH_SIZE)
            if not batch:
                break

            results = list(executor.map(extract_single_puzzle, batch, chunksize=1000))
            valid_rows = [r for r in results if r is not None]

            cursor.executemany(insert_sql, valid_rows)
            conn.commit()
            pbar.update(len(batch))

    elapsed = time.time() - start_time
    final_count = cursor.execute("SELECT COUNT(*) FROM structural_features;").fetchone()[0]
    conn.close()

    print(f"\nTamamlandı! Süre: {elapsed / 60:.1f} dakika | Kaydedilen satır: {final_count:,}")


if __name__ == "__main__":
    main()