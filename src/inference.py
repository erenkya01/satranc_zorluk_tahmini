import json
import re
import sqlite3
from pathlib import Path

import chess
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

DB_PATH = Path("data") / "puzzles.db"
MAIA2_SAVE_ROOT = Path("maia2_models")
ELO_LEVELS_P1 = [1100, 1300, 1500, 1700, 1900]
ELO_LEVELS_P3 = [1100, 1500, 1900]

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10,
}

STRUCTURAL_FEATURE_COLS = [
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

MAIA2_P1_COLS = [
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

MAIA2_P3_COLS = [
    "has_ply3",
    "maia_p3_prob_1100",
    "maia_p3_prob_1500",
    "maia_p3_prob_1900",
    "maia_p3_top1_match_1500",
    "maia_p3_rank_1500",
    "maia_p3_entropy_1500",
]

MAIA2_V4_COLS = [
    "maia_top1_prob_1900",
    "maia_top2_prob_1900",
    "maia_top1_top2_gap_1900",
    "maia_move_rank_1900",
    "maia_prob_ratio_1900",
    "maia_blitz_prob_1500",
    "maia_blitz_prob_1900",
    "maia_rapid_minus_blitz_1900",
    "maia_blitz_trap_gap_1900",
    "maia_blitz_rapid_top1_agree",
    "opp_p2_entropy_1900",
    "opp_p2_top1_prob_1900",
    "opp_p2_sol_prob_1900",
    "has_ply5",
    "maia_p5_prob_1500",
    "maia_p5_prob_1900",
    "maia_p5_rank_1900",
]

PIECE_NAMES_TR = {
    chess.PAWN: "Piyon",
    chess.KNIGHT: "At",
    chess.BISHOP: "Fil",
    chess.ROOK: "Kale",
    chess.QUEEN: "Vezir",
    chess.KING: "Şah",
}

_MAIA2_RUNTIME = None


def _get_maia2_runtime(need_blitz: bool = False):
    global _MAIA2_RUNTIME
    import torch
    from maia2 import inference as m2_inf
    from maia2 import model as m2_model

    if _MAIA2_RUNTIME is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_moves = m2_inf.get_all_possible_moves()
        all_moves_dict = (
            {m: i for i, m in enumerate(raw_moves)}
            if isinstance(raw_moves, list)
            else raw_moves
        )
        elo_dict = m2_inf.create_elo_dict()

        MAIA2_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        maia_net = m2_model.from_pretrained(
            type="rapid",
            device=str(device),
            save_root=str(MAIA2_SAVE_ROOT),
        )
        maia_net.eval()

        _MAIA2_RUNTIME = {
            "torch": torch,
            "m2_inf": m2_inf,
            "device": device,
            "all_moves_dict": all_moves_dict,
            "elo_dict": elo_dict,
            "model": maia_net,
            "blitz_model": None,
        }

    if need_blitz and _MAIA2_RUNTIME["blitz_model"] is None:
        blitz_net = m2_model.from_pretrained(
            type="blitz",
            device=str(_MAIA2_RUNTIME["device"]),
            save_root=str(MAIA2_SAVE_ROOT),
        )
        blitz_net.eval()
        _MAIA2_RUNTIME["blitz_model"] = blitz_net

    return _MAIA2_RUNTIME


def _eval_single_board_probs(net, rt, fen_str: str, elo_val: int):
    torch, m2_inf, device = rt["torch"], rt["m2_inf"], rt["device"]
    all_moves_dict, elo_dict = rt["all_moves_dict"], rt["elo_dict"]

    board_input, _, _, legal_mask = m2_inf.preprocessing(
        fen_str, elo_val, elo_val, elo_dict, all_moves_dict
    )
    boards_t = board_input.unsqueeze(0).to(device)
    legal_t = legal_mask.unsqueeze(0).float().to(device)
    neg_inf = torch.tensor(-1e4, device=device, dtype=torch.float32)

    elo_cat = m2_inf.map_to_category(elo_val, elo_dict)
    elo_t = torch.tensor([elo_cat], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = net(boards_t, elo_t, elo_t)
        logits = (out[0] if isinstance(out, (tuple, list)) else out).float()
        masked_logits = torch.where(legal_t > 0.5, logits, neg_inf)
        probs = torch.softmax(masked_logits, dim=-1)[0]
    return probs, legal_t[0]


def compute_maia2_p1_live(puzzle_fen: str, sol_move_uci: str) -> dict:
    rt = _get_maia2_runtime(need_blitz=False)
    torch, m2_inf, maia_net = rt["torch"], rt["m2_inf"], rt["model"]
    all_moves_dict = rt["all_moves_dict"]

    board = chess.Board(puzzle_fen)
    mirrored_sol = m2_inf.mirror_move(sol_move_uci) if board.turn == chess.BLACK else sol_move_uci
    move_idx = all_moves_dict.get(mirrored_sol, -1)
    if move_idx < 0:
        raise ValueError(f"Çözüm hamlesi Maia-2 sözlüğünde bulunamadı: {sol_move_uci}")

    probs_per_elo = []
    top1_prob_1500 = top1_match_1500 = top1_match_1900 = 0.0
    rank_1500 = 1.0
    entropy_1500 = entropy_1900 = 0.0

    for elo_val in ELO_LEVELS_P1:
        probs, legal_1d = _eval_single_board_probs(maia_net, rt, puzzle_fen, elo_val)
        sol_p = float(probs[move_idx].item())
        probs_per_elo.append(sol_p)

        if elo_val in (1500, 1900):
            t1_prob, t1_idx = probs.max(dim=-1)
            matches = float(int(t1_idx.item()) == move_idx)
            safe_p = torch.clamp(probs, min=1e-12)
            ent = float(
                -torch.sum(
                    torch.where(legal_1d > 0.5, probs * torch.log(safe_p), torch.zeros_like(probs))
                ).item()
            )
            if elo_val == 1500:
                top1_prob_1500 = float(t1_prob.item())
                top1_match_1500 = matches
                entropy_1500 = ent
                rank_1500 = float((probs > sol_p).sum().item() + 1.0)
            else:
                top1_match_1900 = matches
                entropy_1900 = ent

    p1100, p1300, p1500, p1700, p1900 = probs_per_elo
    return {
        "maia_prob_1100": p1100,
        "maia_prob_1300": p1300,
        "maia_prob_1500": p1500,
        "maia_prob_1700": p1700,
        "maia_prob_1900": p1900,
        "maia_prob_mean": sum(probs_per_elo) / 5.0,
        "maia_prob_slope": p1900 - p1100,
        "maia_top1_prob_1500": top1_prob_1500,
        "maia_top1_matches_1500": top1_match_1500,
        "maia_top1_matches_1900": top1_match_1900,
        "maia_move_rank_1500": rank_1500,
        "maia_prob_ratio_1500": p1500 / max(top1_prob_1500, 1e-6),
        "maia_entropy_1500": entropy_1500,
        "maia_entropy_1900": entropy_1900,
    }


def compute_maia2_p3_live(ply3_fen: str | None, sol2_uci: str | None) -> dict:
    default_p3 = {
        "has_ply3": 0.0,
        "maia_p3_prob_1100": 1.0,
        "maia_p3_prob_1500": 1.0,
        "maia_p3_prob_1900": 1.0,
        "maia_p3_top1_match_1500": 1.0,
        "maia_p3_rank_1500": 1.0,
        "maia_p3_entropy_1500": 0.0,
    }
    if not ply3_fen or not sol2_uci:
        return default_p3

    rt = _get_maia2_runtime(need_blitz=False)
    torch, m2_inf, maia_net = rt["torch"], rt["m2_inf"], rt["model"]
    all_moves_dict = rt["all_moves_dict"]

    board = chess.Board(ply3_fen)
    mirrored_sol = m2_inf.mirror_move(sol2_uci) if board.turn == chess.BLACK else sol2_uci
    move_idx = all_moves_dict.get(mirrored_sol, -1)
    if move_idx < 0:
        return default_p3

    probs_per_elo = []
    top1_match_1500 = 0.0
    rank_1500 = 1.0
    entropy_1500 = 0.0

    for elo_val in ELO_LEVELS_P3:
        probs, legal_1d = _eval_single_board_probs(maia_net, rt, ply3_fen, elo_val)
        sol_p = float(probs[move_idx].item())
        probs_per_elo.append(sol_p)

        if elo_val == 1500:
            _, t1_idx = probs.max(dim=-1)
            top1_match_1500 = float(int(t1_idx.item()) == move_idx)
            rank_1500 = float((probs > sol_p).sum().item() + 1.0)
            safe_p = torch.clamp(probs, min=1e-12)
            entropy_1500 = float(
                -torch.sum(
                    torch.where(legal_1d > 0.5, probs * torch.log(safe_p), torch.zeros_like(probs))
                ).item()
            )

    return {
        "has_ply3": 1.0,
        "maia_p3_prob_1100": probs_per_elo[0],
        "maia_p3_prob_1500": probs_per_elo[1],
        "maia_p3_prob_1900": probs_per_elo[2],
        "maia_p3_top1_match_1500": top1_match_1500,
        "maia_p3_rank_1500": rank_1500,
        "maia_p3_entropy_1500": entropy_1500,
    }


def compute_maia2_v4_live(
    puzzle_fen: str,
    sol1_uci: str,
    ply2_fen: str | None,
    opp1_uci: str | None,
    ply5_fen: str | None,
    sol3_uci: str | None,
) -> dict:
    rt = _get_maia2_runtime(need_blitz=True)
    torch, m2_inf = rt["torch"], rt["m2_inf"]
    rapid_net, blitz_net = rt["model"], rt["blitz_model"]
    all_moves_dict = rt["all_moves_dict"]

    b1 = chess.Board(puzzle_fen)
    m1 = m2_inf.mirror_move(sol1_uci) if b1.turn == chess.BLACK else sol1_uci
    idx1 = all_moves_dict[m1]

    p1_r1900, _ = _eval_single_board_probs(rapid_net, rt, puzzle_fen, 1900)
    p1_b1500, _ = _eval_single_board_probs(blitz_net, rt, puzzle_fen, 1500)
    p1_b1900, _ = _eval_single_board_probs(blitz_net, rt, puzzle_fen, 1900)

    sol_r1900 = float(p1_r1900[idx1].item())
    top2_vals, top2_idxs = torch.topk(p1_r1900, k=2)
    top1_r1900 = float(top2_vals[0].item())
    top2_r1900 = float(top2_vals[1].item())
    gap_r1900 = top1_r1900 - top2_r1900
    rank_r1900 = float((p1_r1900 > sol_r1900).sum().item() + 1.0)
    ratio_r1900 = sol_r1900 / max(top1_r1900, 1e-6)

    sol_b1500 = float(p1_b1500[idx1].item())
    sol_b1900 = float(p1_b1900[idx1].item())
    rapid_minus_blitz = sol_r1900 - sol_b1900
    b_t1_prob, b_t1_idx = p1_b1900.max(dim=-1)
    blitz_trap_gap = float(b_t1_prob.item()) - sol_b1900
    top1_agree = float(int(top2_idxs[0].item()) == int(b_t1_idx.item()))

    opp_ent_1900, opp_top1_1900, opp_sol_1900 = 0.0, 1.0, 1.0
    if ply2_fen and opp1_uci:
        b2 = chess.Board(ply2_fen)
        m2 = m2_inf.mirror_move(opp1_uci) if b2.turn == chess.BLACK else opp1_uci
        idx2 = all_moves_dict.get(m2, -1)
        if idx2 >= 0:
            p2_probs, l2_1d = _eval_single_board_probs(rapid_net, rt, ply2_fen, 1900)
            safe_p2 = torch.clamp(p2_probs, min=1e-12)
            opp_ent_1900 = float(
                -torch.sum(torch.where(l2_1d > 0.5, p2_probs * torch.log(safe_p2), torch.zeros_like(p2_probs))).item()
            )
            opp_top1_1900 = float(p2_probs.max().item())
            opp_sol_1900 = float(p2_probs[idx2].item())

    has_ply5, p5_1500, p5_1900, p5_rank_1900 = 0.0, 1.0, 1.0, 1.0
    if ply5_fen and sol3_uci:
        b5 = chess.Board(ply5_fen)
        m5 = m2_inf.mirror_move(sol3_uci) if b5.turn == chess.BLACK else sol3_uci
        idx5 = all_moves_dict.get(m5, -1)
        if idx5 >= 0:
            probs5_1500, _ = _eval_single_board_probs(rapid_net, rt, ply5_fen, 1500)
            probs5_1900, _ = _eval_single_board_probs(rapid_net, rt, ply5_fen, 1900)
            has_ply5 = 1.0
            p5_1500 = float(probs5_1500[idx5].item())
            p5_1900 = float(probs5_1900[idx5].item())
            p5_rank_1900 = float((probs5_1900 > p5_1900).sum().item() + 1.0)

    return {
        "maia_top1_prob_1900": top1_r1900,
        "maia_top2_prob_1900": top2_r1900,
        "maia_top1_top2_gap_1900": gap_r1900,
        "maia_move_rank_1900": rank_r1900,
        "maia_prob_ratio_1900": ratio_r1900,
        "maia_blitz_prob_1500": sol_b1500,
        "maia_blitz_prob_1900": sol_b1900,
        "maia_rapid_minus_blitz_1900": rapid_minus_blitz,
        "maia_blitz_trap_gap_1900": blitz_trap_gap,
        "maia_blitz_rapid_top1_agree": top1_agree,
        "opp_p2_entropy_1900": opp_ent_1900,
        "opp_p2_top1_prob_1900": opp_top1_1900,
        "opp_p2_sol_prob_1900": opp_sol_1900,
        "has_ply5": has_ply5,
        "maia_p5_prob_1500": p5_1500,
        "maia_p5_prob_1900": p5_1900,
        "maia_p5_rank_1900": p5_rank_1900,
    }


def add_v3_interactions_dict(feat_dict: dict) -> None:
    p1_1500 = float(feat_dict["maia_prob_1500"])
    p1_1900 = float(feat_dict["maia_prob_1900"])
    p3_1100 = float(feat_dict["maia_p3_prob_1100"])
    p3_1500 = float(feat_dict["maia_p3_prob_1500"])
    p3_1900 = float(feat_dict["maia_p3_prob_1900"])
    has_p3 = float(feat_dict["has_ply3"])
    r1_1500 = float(feat_dict["maia_move_rank_1500"])
    r3_1500 = float(feat_dict["maia_p3_rank_1500"])

    feat_dict["maia_chain_prob_1500"] = p1_1500 * p3_1500
    feat_dict["maia_chain_prob_1900"] = p1_1900 * p3_1900
    feat_dict["maia_min_prob_1500"] = min(p1_1500, p3_1500)
    feat_dict["maia_log_prob_1900"] = float(np.log(np.clip(p1_1900, 1e-6, 1.0)))
    feat_dict["maia_p3_log_prob_1900"] = float(np.log(np.clip(p3_1900, 1e-6, 1.0)))
    feat_dict["maia_p3_slope"] = p3_1900 - p3_1100
    feat_dict["maia_rank_sum_1500"] = r1_1500 + (r3_1500 - 1.0) * has_p3


def add_v4_interactions_dict(feat_dict: dict) -> None:
    p1_1900 = float(feat_dict["maia_prob_1900"])
    p3_1900 = float(feat_dict["maia_p3_prob_1900"])
    p5_1900 = float(feat_dict["maia_p5_prob_1900"])
    has_p3 = float(feat_dict["has_ply3"])
    has_p5 = float(feat_dict["has_ply5"])

    r1_1900 = float(feat_dict["maia_move_rank_1900"])
    r3_1900 = max(1.0, float(feat_dict["maia_p3_rank_1500"]) * 0.75)
    r5_1900 = float(feat_dict["maia_p5_rank_1900"])

    blitz_trap = float(feat_dict["maia_blitz_trap_gap_1900"])
    top1_match_1900 = float(feat_dict["maia_top1_matches_1900"])
    opp_ent = float(feat_dict["opp_p2_entropy_1900"])
    opp_sol = float(feat_dict["opp_p2_sol_prob_1900"])

    full_chain = p1_1900 * p3_1900 * p5_1900
    feat_dict["maia_full_chain_prob_1900"] = full_chain
    feat_dict["maia_full_chain_log_1900"] = float(np.log(np.clip(full_chain, 1e-9, 1.0)))
    feat_dict["maia_min_chain_prob_1900"] = min(p1_1900, p3_1900, p5_1900)
    feat_dict["maia_rank_sum_1900"] = r1_1900 + (r3_1900 - 1.0) * has_p3 + (r5_1900 - 1.0) * has_p5
    feat_dict["maia_trap_severity_1900"] = blitz_trap * (1.0 - top1_match_1900)
    feat_dict["maia_calc_burden_1900"] = (1.0 - opp_sol) * opp_ent


def fetch_table_row_dict(table_name: str, cols: list[str], puzzle_id: str | None) -> dict | None:
    if not puzzle_id or not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cols_str = ", ".join(cols)
        row = conn.execute(
            f"SELECT {cols_str} FROM {table_name} WHERE puzzle_id = ?",
            (puzzle_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {col: float(val) for col, val in zip(cols, row)}
    except Exception:
        return None


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


def parse_moves_sequence(fen: str, moves_raw: str) -> list[chess.Move]:
    cleaned = re.sub(r"\{[^}]*\}", " ", moves_raw)
    cleaned = re.sub(r"\d+\.(\.\.)?", " ", cleaned)
    tokens = [t.strip("!?+#") if len(t) > 4 else t.strip("!?") for t in cleaned.split()]
    tokens = [t for t in tokens if t and t not in ("1-0", "0-1", "1/2-1/2", "*")]

    board = chess.Board(fen)
    parsed_moves = []
    for tok in tokens:
        try:
            mv = chess.Move.from_uci(tok)
            if mv not in board.legal_moves:
                mv = board.parse_san(tok)
        except ValueError:
            mv = board.parse_san(tok)
        parsed_moves.append(mv)
        board.push(mv)
    return parsed_moves


class PuzzleDifficultyPredictor:
    def __init__(self, model_dir: str | Path | None = None):
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        elif Path("models_v4").exists():
            self.model_dir = Path("models_v4")
        elif Path("models_v3_1").exists():
            self.model_dir = Path("models_v3_1")
        elif Path("models_v3_full").exists():
            self.model_dir = Path("models_v3_full")
        elif Path("models_hybrid_maia2").exists():
            self.model_dir = Path("models_hybrid_maia2")
        elif Path("models_structural_only").exists():
            self.model_dir = Path("models_structural_only")
        else:
            self.model_dir = Path("models")

        self.is_v4 = (self.model_dir / "lgb_v4.joblib").exists()
        self.is_v3 = (self.model_dir / "lgb_v3.joblib").exists() or self.is_v4
        self.is_v2 = (self.model_dir / "lgb_hybrid.joblib").exists()
        self.is_hybrid = self.is_v3 or self.is_v2

        if self.is_v4:
            self.lgb_model = joblib.load(self.model_dir / "lgb_v4.joblib")
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(self.model_dir / "xgb_v4.json"))
            self.cat_model = CatBoostRegressor()
            self.cat_model.load_model(str(self.model_dir / "cat_v4.cbm"))
        elif self.is_v3:
            self.lgb_model = joblib.load(self.model_dir / "lgb_v3.joblib")
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(self.model_dir / "xgb_v3.json"))
            self.cat_model = CatBoostRegressor()
            self.cat_model.load_model(str(self.model_dir / "cat_v3.cbm"))
        elif self.is_v2:
            self.lgb_model = joblib.load(self.model_dir / "lgb_hybrid.joblib")
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(self.model_dir / "xgb_hybrid.json"))
            self.cat_model = CatBoostRegressor()
            self.cat_model.load_model(str(self.model_dir / "cat_hybrid.cbm"))

        if self.is_hybrid:
            with open(self.model_dir / "feature_cols.json", "r", encoding="utf-8") as f:
                f_info = json.load(f)
            self.feature_cols = f_info["feature_cols"]
            self.context_cols = f_info["context_cols"]

            with open(self.model_dir / "ensemble_weights.json", "r", encoding="utf-8") as f:
                w_info = json.load(f)
            self.weights = [w_info["lgb"], w_info["xgb"], w_info["cat"]]

            self.residual_models = joblib.load(self.model_dir / "residual_models.joblib")
            self.calibrator = joblib.load(self.model_dir / "calibrator.joblib")
            with open(self.model_dir / "residual_meta.json", "r", encoding="utf-8") as f:
                r_meta = json.load(f)
            self.alpha = float(r_meta["alpha"])
            self.calibrator_type = r_meta.get("calibrator_type", "Ridge")
            self.uncertainty_scale = 8.0
            unc_path = self.model_dir / "uncertainty_models.joblib"
            self.uncertainty_models = joblib.load(unc_path) if unc_path.exists() else None
        else:
            self.lgb_model = lgb.Booster(model_file=str(self.model_dir / "lgb_model.txt"))
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(self.model_dir / "xgb_model.json"))
            self.cat_model = CatBoostRegressor()
            self.cat_model.load_model(str(self.model_dir / "cat_model.cbm"))

            meta = joblib.load(self.model_dir / "ensemble_meta.joblib")
            self.weights = meta["weights"]
            self.feature_cols = meta.get("feature_cols", STRUCTURAL_FEATURE_COLS)
            self.context_cols = []

            stage2 = joblib.load(self.model_dir / "stage2_artifacts.joblib")
            self.residual_models = stage2["residual_models"]
            self.alpha = stage2["alpha"]
            self.calibrator_type = stage2["calibrator_type"]
            self.calibrator = stage2["calibrator"]
            self.uncertainty_scale = stage2["uncertainty_scale"]
            self.uncertainty_models = None

    def extract_features_and_states(self, fen: str, moves_raw: str, puzzle_id: str | None = None):
        moves = parse_moves_sequence(fen, moves_raw)
        if len(moves) < 2:
            raise ValueError(
                "En az 2 hamle girilmelidir (1. hamle rakibin hazırlık hatası, 2. hamle çözüm başlangıcı)."
            )

        board = chess.Board(fen)
        states = [
            {
                "step": 0,
                "label": "Ham PGN/FEN Başlangıcı",
                "fen": board.fen(),
                "last_move": None,
                "san": "-",
            }
        ]

        setup_move = moves[0]
        setup_san = board.san(setup_move)
        setup_is_capture = int(board.is_capture(setup_move))
        board.push(setup_move)
        setup_is_check = int(board.is_check())

        puzzle_start_fen = board.fen()
        first_solution_uci = moves[1].uci()
        ply2_fen = None
        first_opp_uci = moves[2].uci() if len(moves) >= 3 else None
        ply3_fen = None
        second_solution_uci = moves[3].uci() if len(moves) >= 4 else None
        ply5_fen = None
        third_solution_uci = moves[5].uci() if len(moves) >= 6 else None

        states.append(
            {
                "step": 1,
                "label": f"Rakip Hatası: {setup_san}",
                "fen": puzzle_start_fen,
                "last_move": setup_move,
                "san": setup_san,
            }
        )

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

        legal_moves = list(board.legal_moves)
        legal_moves_start = len(legal_moves)
        avail_checks_start = sum(1 for lm in legal_moves if board.gives_check(lm))
        avail_captures_start = sum(1 for lm in legal_moves if board.is_capture(lm))

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

        solution_moves = moves[1:]
        solution_ply = len(solution_moves)
        solver_moves_count = (solution_ply + 1) // 2

        first_move_is_check = first_move_is_capture = first_move_is_quiet = 0
        captured_piece_val_1 = first_move_piece = first_move_dist = 0
        first_move_is_fork = first_move_attacks_queen = first_move_attacks_king_zone = 0
        is_sacrifice = 0

        solver_checks = solver_captures = solver_promotions = 0
        total_quiet_solver_moves = second_solver_move_is_quiet = 0
        max_piece_value_moved = distinct_solver_pieces_moved = 0
        last_solver_to_sq = None

        opp_legal_moves_ply2 = opp_is_forced_ply2 = solver_legal_moves_ply3 = 0

        for idx, mv in enumerate(solution_moves):
            is_solver_turn = idx % 2 == 0
            san_str = board.san(mv)
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

            actor = "Çözücü" if is_solver_turn else "Rakip"
            states.append(
                {
                    "step": idx + 2,
                    "label": f"{idx + 1}. {actor}: {san_str}",
                    "fen": board.fen(),
                    "last_move": mv,
                    "san": san_str,
                }
            )

            if idx == 0:
                first_move_is_check = int(is_chk)
                first_move_is_quiet = int(is_quiet)
                attacks_bb = int(board.attacks(mv.to_square))
                opp_occ_bb = int(board.occupied_co[opp_color])
                first_move_is_fork = int(chess.popcount(attacks_bb & opp_occ_bb) >= 2)
                first_move_attacks_queen = int(bool(attacks_bb & int(board.pieces(chess.QUEEN, opp_color))))
                first_move_attacks_king_zone = int(
                    bool((chess.BB_SQUARES[mv.to_square] | attacks_bb) & opp_king_zone_bb)
                )
                opp_legal_moves_ply2 = board.legal_moves.count()
                opp_is_forced_ply2 = int(opp_legal_moves_ply2 == 1)
                if solution_ply >= 2:
                    ply2_fen = board.fen()

            if idx == 1:
                _, _, _, _, s_mat_mid, _ = _material_and_counts(board, solver_color)
                _, _, _, _, o_mat_mid, _ = _material_and_counts(board, opp_color)
                if (s_mat_mid - o_mat_mid) < mat_diff_start:
                    is_sacrifice = 1
                if solution_ply >= 3:
                    solver_legal_moves_ply3 = board.legal_moves.count()
                    ply3_fen = board.fen()

            if idx == 2:
                second_solver_move_is_quiet = int(is_quiet)

            if idx == 3 and solution_ply >= 5:
                ply5_fen = board.fen()

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

        feat_dict = {
            "solution_ply": solution_ply,
            "solver_moves_count": solver_moves_count,
            "ends_in_mate": ends_in_mate,
            "is_mate_in_1": is_mate_in_1,
            "total_pieces": total_pieces,
            "solver_pawns": s_p,
            "solver_minors": s_m,
            "solver_rooks": s_r,
            "solver_queens": s_q,
            "opp_pawns": o_p,
            "opp_minors": o_m,
            "opp_rooks": o_r,
            "opp_queens": o_q,
            "solver_material": s_mat_start,
            "opp_material": o_mat_start,
            "material_diff_start": mat_diff_start,
            "material_diff_end": mat_diff_end,
            "material_gain": material_gain,
            "game_phase": game_phase,
            "is_endgame": is_endgame,
            "solver_turn_white": solver_turn_white,
            "in_check_at_start": in_check_at_start,
            "setup_is_capture": setup_is_capture,
            "setup_is_check": setup_is_check,
            "first_move_is_check": first_move_is_check,
            "first_move_is_capture": first_move_is_capture,
            "first_move_is_quiet": first_move_is_quiet,
            "captured_piece_val_1": captured_piece_val_1,
            "is_sacrifice": is_sacrifice,
            "first_move_piece": first_move_piece,
            "first_move_dist": first_move_dist,
            "first_move_is_fork": first_move_is_fork,
            "first_move_attacks_queen": first_move_attacks_queen,
            "first_move_attacks_king_zone": first_move_attacks_king_zone,
            "solver_checks": solver_checks,
            "solver_captures": solver_captures,
            "solver_promotions": solver_promotions,
            "total_quiet_solver_moves": total_quiet_solver_moves,
            "second_solver_move_is_quiet": second_solver_move_is_quiet,
            "max_piece_value_moved": max_piece_value_moved,
            "distinct_solver_pieces_moved": distinct_solver_pieces_moved,
            "legal_moves_start": legal_moves_start,
            "avail_checks_start": avail_checks_start,
            "avail_captures_start": avail_captures_start,
            "quiet_despite_checks": quiet_despite_checks,
            "opp_legal_moves_ply2": opp_legal_moves_ply2,
            "opp_is_forced_ply2": opp_is_forced_ply2,
            "solver_legal_moves_ply3": solver_legal_moves_ply3,
            "solver_king_attackers": solver_king_attackers,
            "opp_king_attackers": opp_king_attackers,
            "opp_king_mobility": opp_king_mobility,
            "solver_pins_count": solver_pins_count,
            "opp_pins_count": opp_pins_count,
            "total_attacked_opp_pieces": total_attacked_opp_pieces,
            "undefended_opp_pieces": undefended_opp_pieces,
            "undefended_solver_pieces": undefended_solver_pieces,
        }

        if self.is_hybrid:
            m1_feats = fetch_table_row_dict("maia2_features", MAIA2_P1_COLS, puzzle_id)
            if m1_feats is None:
                m1_feats = compute_maia2_p1_live(puzzle_start_fen, first_solution_uci)
            feat_dict.update(m1_feats)

        if self.is_v3:
            m3_feats = fetch_table_row_dict("maia2_ply3_features", MAIA2_P3_COLS, puzzle_id)
            if m3_feats is None:
                m3_feats = compute_maia2_p3_live(ply3_fen, second_solution_uci)
            feat_dict.update(m3_feats)
            add_v3_interactions_dict(feat_dict)

        if self.is_v4:
            m4_feats = fetch_table_row_dict("maia2_v4_features", MAIA2_V4_COLS, puzzle_id)
            if m4_feats is None:
                m4_feats = compute_maia2_v4_live(
                    puzzle_start_fen,
                    first_solution_uci,
                    ply2_fen,
                    first_opp_uci,
                    ply5_fen,
                    third_solution_uci,
                )
            feat_dict.update(m4_feats)
            add_v4_interactions_dict(feat_dict)

        X = np.array([[feat_dict[col] for col in self.feature_cols]], dtype=np.float32)
        return X, feat_dict, states, solver_color

    def predict(self, fen: str, moves_raw: str, puzzle_id: str | None = None) -> dict:
        X, feat_dict, states, solver_color = self.extract_features_and_states(
            fen, moves_raw, puzzle_id=puzzle_id
        )

        p_lgb = float(self.lgb_model.predict(X)[0])
        p_xgb = float(self.xgb_model.predict(X)[0])
        p_cat = float(self.cat_model.predict(X)[0])
        p_ens = float(
            self.weights[0] * p_lgb + self.weights[1] * p_xgb + self.weights[2] * p_cat
        )

        preds_stack = np.array([[p_lgb, p_xgb, p_cat]], dtype=np.float32)
        pred_std = float(np.std(preds_stack))
        pred_range = float(np.max(preds_stack) - np.min(preds_stack))

        if self.is_hybrid:
            base_meta = np.array(
                [
                    [
                        p_lgb,
                        p_xgb,
                        p_cat,
                        p_ens,
                        pred_std,
                        pred_range,
                        p_lgb - p_xgb,
                        p_lgb - p_cat,
                        p_xgb - p_cat,
                    ]
                ],
                dtype=np.float32,
            )
            ctx_arr = np.array([[feat_dict[c] for c in self.context_cols]], dtype=np.float32)
            X_meta = np.hstack([base_meta, ctx_arr])
        else:
            X_meta = np.hstack(
                [
                    X,
                    preds_stack,
                    np.array(
                        [[p_ens, pred_std, pred_range, p_lgb - p_cat, p_lgb - p_xgb]],
                        dtype=np.float32,
                    ),
                ]
            )

        res_pred = float(np.mean([m.predict(X_meta)[0] for m in self.residual_models]))
        p_after_res = p_ens + self.alpha * res_pred

        if isinstance(self.calibrator, list):
            if self.calibrator_type == "BaggedSplineRidge":
                p_final = float(
                    np.mean([cal.predict([[p_after_res]])[0] for cal in self.calibrator])
                )
            else:
                p_final = float(
                    np.mean([iso.predict([p_after_res])[0] for iso in self.calibrator])
                )
        elif self.calibrator_type == "İzotonik Kalibrasyon":
            p_final = float(self.calibrator.predict([p_after_res])[0])
        else:
            p_final = float(self.calibrator.predict([[p_after_res]])[0])

        if self.is_hybrid and getattr(self, "uncertainty_models", None) is not None:
            pred_log_var = float(np.mean([um.predict(X_meta)[0] for um in self.uncertainty_models]))
            expected_margin = float(
                np.clip(np.sqrt(max(np.exp(pred_log_var) - 2500.0, 6400.0)), 85.0, 460.0)
            )
        elif self.is_hybrid:
            uncertainty_raw = pred_std + 0.35 * abs(res_pred)
            expected_margin = float(np.clip(105.0 + uncertainty_raw * self.uncertainty_scale, 105.0, 440.0))
        else:
            uncertainty_raw = pred_std + 0.25 * abs(res_pred)
            expected_margin = max(80.0, uncertainty_raw * self.uncertainty_scale)

        return {
            "predicted_rating": round(p_final),
            "expected_margin": round(expected_margin),
            "ci_low": round(p_final - expected_margin),
            "ci_high": round(p_final + expected_margin),
            "solver_color": "Beyaz" if solver_color == chess.WHITE else "Siyah",
            "is_hybrid": self.is_hybrid,
            "is_v3": self.is_v3,
            "is_v4": self.is_v4,
            "models": {
                "LightGBM": round(p_lgb, 1),
                "XGBoost": round(p_xgb, 1),
                "CatBoost": round(p_cat, 1),
                "3'lü Topluluk (Ensemble)": round(p_ens, 1),
                "Artık Düzeltme (Residual)": round(self.alpha * res_pred, 1),
                "Nihai Tahmin (Kalibreli)": round(p_final, 1),
            },
            "features": feat_dict,
            "states": states,
        }