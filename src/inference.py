import re
from pathlib import Path

import chess
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

FEATURE_COLS = [
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
    "in_check_at_start",
    "setup_is_capture",
    "setup_is_check",
    "first_move_is_check",
    "first_move_is_capture",
    "first_move_is_quiet",
    "is_sacrifice",
    "first_move_piece",
    "first_move_dist",
    "solver_checks",
    "solver_captures",
    "solver_promotions",
    "legal_moves_start",
    "solver_king_attackers",
    "opp_king_attackers",
]

PIECE_NAMES_TR = {
    chess.PAWN: "Piyon",
    chess.KNIGHT: "At",
    chess.BISHOP: "Fil",
    chess.ROOK: "Kale",
    chess.QUEEN: "Vezir",
    chess.KING: "Şah",
}


def _material_and_counts(board: chess.Board, color: chess.Color):
    pawns = len(board.pieces(chess.PAWN, color))
    knights = len(board.pieces(chess.KNIGHT, color))
    bishops = len(board.pieces(chess.BISHOP, color))
    rooks = len(board.pieces(chess.ROOK, color))
    queens = len(board.pieces(chess.QUEEN, color))
    material = pawns * 1 + (knights + bishops) * 3 + rooks * 5 + queens * 9
    return pawns, (knights + bishops), rooks, queens, material


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
    """Hem UCI (e2e4 e7e5) hem de PGN/SAN (1. e4 e5 2. Nf3) formatını destekler."""
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
        elif Path("models_structural_only").exists():
            self.model_dir = Path("models_structural_only")
        else:
            self.model_dir = Path("models")

        self.lgb_model = lgb.Booster(model_file=str(self.model_dir / "lgb_model.txt"))
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(self.model_dir / "xgb_model.json"))
        self.cat_model = CatBoostRegressor()
        self.cat_model.load_model(str(self.model_dir / "cat_model.cbm"))

        meta = joblib.load(self.model_dir / "ensemble_meta.joblib")
        self.weights = meta["weights"]

        stage2 = joblib.load(self.model_dir / "stage2_artifacts.joblib")
        self.residual_models = stage2["residual_models"]
        self.alpha = stage2["alpha"]
        self.calibrator_type = stage2["calibrator_type"]
        self.calibrator = stage2["calibrator"]
        self.uncertainty_scale = stage2["uncertainty_scale"]

    def extract_features_and_states(self, fen: str, moves_raw: str):
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

        # 1. Rakibin hazırlık hatası
        setup_move = moves[0]
        setup_san = board.san(setup_move)
        setup_is_capture = int(board.is_capture(setup_move))
        board.push(setup_move)
        setup_is_check = int(board.is_check())

        states.append(
            {
                "step": 1,
                "label": f"Rakip Hatası: {setup_san}",
                "fen": board.fen(),
                "last_move": setup_move,
                "san": setup_san,
            }
        )

        # Gerçek bulmaca başlangıç pozisyonu
        solver_color = board.turn
        opp_color = not solver_color

        s_p, s_m, s_r, s_q, s_mat_start = _material_and_counts(board, solver_color)
        o_p, o_m, o_r, o_q, o_mat_start = _material_and_counts(board, opp_color)
        total_pieces = s_p + s_m + s_r + s_q + o_p + o_m + o_r + o_q + 2
        mat_diff_start = s_mat_start - o_mat_start

        in_check_at_start = int(board.is_check())
        legal_moves_start = board.legal_moves.count()
        solver_king_attackers = _king_zone_pressure(board, solver_color)
        opp_king_attackers = _king_zone_pressure(board, opp_color)

        solution_moves = moves[1:]
        solution_ply = len(solution_moves)
        solver_moves_count = (solution_ply + 1) // 2

        first_move_is_check = 0
        first_move_is_capture = 0
        first_move_is_quiet = 0
        first_move_piece = 0
        first_move_dist = 0
        is_sacrifice = 0
        solver_checks = 0
        solver_captures = 0
        solver_promotions = 0

        for idx, mv in enumerate(solution_moves):
            is_solver_turn = idx % 2 == 0
            san_str = board.san(mv)
            is_cap = board.is_capture(mv)
            is_promo = mv.promotion is not None

            if idx == 0:
                piece = board.piece_at(mv.from_square)
                first_move_piece = piece.piece_type if piece else 0
                f1, r1 = chess.square_file(mv.from_square), chess.square_rank(mv.from_square)
                f2, r2 = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
                first_move_dist = abs(f1 - f2) + abs(r1 - r2)
                first_move_is_capture = int(is_cap)

            board.push(mv)
            is_chk = board.is_check()

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
                first_move_is_quiet = int(not (is_cap or is_chk or is_promo))

            if idx == 1:
                _, _, _, _, s_mat_mid = _material_and_counts(board, solver_color)
                _, _, _, _, o_mat_mid = _material_and_counts(board, opp_color)
                if (s_mat_mid - o_mat_mid) < mat_diff_start:
                    is_sacrifice = 1

            if is_solver_turn:
                solver_checks += int(is_chk)
                solver_captures += int(is_cap)
                solver_promotions += int(is_promo)

        ends_in_mate = int(board.is_checkmate())
        is_mate_in_1 = int(ends_in_mate == 1 and solution_ply == 1)

        _, _, _, _, s_mat_end = _material_and_counts(board, solver_color)
        _, _, _, _, o_mat_end = _material_and_counts(board, opp_color)
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
            "in_check_at_start": in_check_at_start,
            "setup_is_capture": setup_is_capture,
            "setup_is_check": setup_is_check,
            "first_move_is_check": first_move_is_check,
            "first_move_is_capture": first_move_is_capture,
            "first_move_is_quiet": first_move_is_quiet,
            "is_sacrifice": is_sacrifice,
            "first_move_piece": first_move_piece,
            "first_move_dist": first_move_dist,
            "solver_checks": solver_checks,
            "solver_captures": solver_captures,
            "solver_promotions": solver_promotions,
            "legal_moves_start": legal_moves_start,
            "solver_king_attackers": solver_king_attackers,
            "opp_king_attackers": opp_king_attackers,
        }

        X = np.array([[feat_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
        return X, feat_dict, states, solver_color

    def predict(self, fen: str, moves_raw: str) -> dict:
        X, feat_dict, states, solver_color = self.extract_features_and_states(fen, moves_raw)

        p_lgb = float(self.lgb_model.predict(X)[0])
        p_xgb = float(self.xgb_model.predict(X)[0])
        p_cat = float(self.cat_model.predict(X)[0])
        p_ens = float(
            self.weights[0] * p_lgb + self.weights[1] * p_xgb + self.weights[2] * p_cat
        )

        preds_stack = np.array([[p_lgb, p_xgb, p_cat]], dtype=np.float32)
        pred_std = float(np.std(preds_stack))
        pred_range = float(np.max(preds_stack) - np.min(preds_stack))

        X_meta = np.hstack(
            [
                X,
                preds_stack,
                np.array([[p_ens, pred_std, pred_range, p_lgb - p_cat, p_lgb - p_xgb]], dtype=np.float32),
            ]
        )

        res_pred = float(np.mean([m.predict(X_meta)[0] for m in self.residual_models]))
        p_after_res = p_ens + self.alpha * res_pred

        if self.calibrator_type == "İzotonik Kalibrasyon":
            p_final = float(self.calibrator.predict([p_after_res])[0])
        else:
            p_final = float(self.calibrator.predict([[p_after_res]])[0])

        uncertainty_raw = pred_std + 0.25 * abs(res_pred)
        expected_margin = max(80.0, uncertainty_raw * self.uncertainty_scale)

        return {
            "predicted_rating": round(p_final),
            "expected_margin": round(expected_margin),
            "ci_low": round(p_final - expected_margin),
            "ci_high": round(p_final + expected_margin),
            "solver_color": "Beyaz" if solver_color == chess.WHITE else "Siyah",
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