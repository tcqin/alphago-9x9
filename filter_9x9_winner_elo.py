import sgfmill
import sgfmill.sgf
import os
import re
import shutil

data_dir = "data/9x9"
filtered_dir = "data/9x9_filtered"
os.makedirs(filtered_dir, exist_ok=True)

MIN_ELO = 1500


def extract_elo(player_string):
    """Extract ELO from strings like 'delilah (1372)'"""
    match = re.search(r"\((\d+)\)", player_string)
    if match:
        return int(match.group(1))
    return None


kept = 0
skipped_no_winner = 0
skipped_low_rank = 0
skipped_no_rank = 0
total = 0

files = os.listdir(data_dir)
for i, fname in enumerate(files):
    if not fname.endswith(".sgf"):
        continue
    total += 1
    if i % 10000 == 0:
        print(f"Processing {i}/{len(files)}, kept {kept} so far")
    path = os.path.join(data_dir, fname)
    try:
        with open(path, "rb") as f:
            game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
        # Filter: must have a winner
        if game.get_winner() is None:
            skipped_no_winner += 1
            continue

        # Filter: must have rank info for both players
        root = game.get_root()
        try:
            b_name = root.get("PB")
            w_name = root.get("PW")
        except:
            skipped_no_rank += 1
            continue

        b_elo = extract_elo(b_name)
        w_elo = extract_elo(w_name)

        if b_elo is None or w_elo is None:
            skipped_no_rank += 1
            continue

        # Filter: both players must meet minimum ELO
        if b_elo < MIN_ELO or w_elo < MIN_ELO:
            skipped_low_rank += 1
            continue

        # Keep clean games
        kept += 1
        shutil.copy(path, filtered_dir)
    except:
        continue

# for fname in files:
#     path = os.path.join(data_dir, fname)
#     with open(path, "rb") as f:
#         game = sgfmill.sgf.Sgf_game.from_bytes(f.read())

#     root = game.get_root()
#     board_size = game.get_size()
#     winner = game.get_winner()
#     moves = list(game.get_main_sequence())

#     print(f"File: {fname}")
#     print(f"  Board size: {board_size}")
#     print(f"  Winner: {winner}")
#     print(f"  Num moves: {len(moves)}")

#     for prop in ["GM", "KM", "RU", "BR", "WR", "BN", "WN", "PB", "PW", "RE"]:
#         try:
#             val = root.get(prop)
#             print(f"  {prop}: {val}")
#         except KeyError:
#             pass
