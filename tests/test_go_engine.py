import sys

sys.path.append("..")
from go_engine import GoGame, BLACK, WHITE, EMPTY


def test_basic_play():
    g = GoGame()
    g.play(0, 0)
    assert g.board[0, 0] == BLACK
    assert g.current_player == WHITE
    print("test_basic_play passed")


def test_capture():
    # Surround a white stone and capture it
    g = GoGame()
    g.play(0, 1)  # black
    g.play(0, 0)  # white plays corner
    g.play(1, 0)  # black
    # white stone at (0,0) should now be captured
    assert g.board[0, 0] == EMPTY, "White stone should be captured"
    print("test_capture passed")


def test_ko():
    # Set up a ko situation
    g = GoGame(size=9)
    # Simple ko setup
    g.play(0, 1)  # B
    g.play(0, 2)  # W
    g.play(1, 0)  # B
    g.play(1, 3)  # W
    g.play(2, 1)  # B
    g.play(2, 2)  # W
    g.play(1, 2)  # B throws in
    g.play(1, 1)  # W captures - creates ko
    assert g.ko_point is not None, "Ko point should be set"
    assert not g.is_legal(1, 2, BLACK), "Ko recapture should be illegal"
    print("test_ko passed")


def test_pass_and_game_over():
    g = GoGame()
    g.play(None, None)
    g.play(None, None)
    assert g.is_game_over()
    print("test_pass_and_game_over passed")


def test_suicide_illegal():
    g = GoGame()
    # Fill all liberties of corner except one, then check suicide is blocked
    g.board[0, 1] = WHITE
    g.board[1, 0] = WHITE
    assert not g.is_legal(0, 0, BLACK), "Suicide should be illegal"
    print("test_suicide_illegal passed")


def test_random_game():
    import random

    g = GoGame(size=9)
    move_count = 0
    while not g.game_over and move_count < 500:
        moves = g.legal_moves()
        move = random.choice(moves)
        if move is None:
            g.play(None, None)
        else:
            g.play(*move)
        move_count += 1
    b, w = g.score()
    print(f"Game over after {move_count} moves. Black: {b}, White: {w}")
    print(g)
    print("test_random_game passed")


def test_many_random_games():
    import random

    for i in range(1000):
        g = GoGame(size=9)
        move_count = 0
        while not g.game_over and move_count < 500:
            moves = g.legal_moves()
            move = random.choice(moves)
            if move is None:
                g.play(None, None)
            else:
                g.play(*move)
            move_count += 1
    print("test_many_random_games passed — 1000 games completed without errors")


if __name__ == "__main__":
    test_basic_play()
    test_capture()
    test_ko()
    test_pass_and_game_over()
    test_suicide_illegal()
    test_random_game()
    test_many_random_games()
    print("\nAll tests passed!")
