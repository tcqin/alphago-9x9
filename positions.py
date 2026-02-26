import os
import sgfmill
import sgfmill.sgf

data_dir = "data/9x9_filtered"
files = [f for f in os.listdir(data_dir) if f.endswith(".sgf")]

total_moves = 0
for i, fname in enumerate(files):
    path = os.path.join(data_dir, fname)
    try:
        with open(path, "rb") as f:
            game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
        moves = list(game.get_main_sequence())
        total_moves += len(moves)
    except:
        continue
    if i % 10000 == 0:
        print(f"{i}/{len(files)}, total moves so far: {total_moves}")

print(f"Total games: {len(files)}")
print(f"Total positions: {total_moves}")
print(f"Average game length: {total_moves / len(files):.1f} moves")
