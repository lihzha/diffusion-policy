import numpy as np


def main(
    num,
    max_x=10,
    max_y=10,
    min_dist=2,
):
    xy = np.empty((num, 4), dtype=int)
    cnt = 0
    while True:
        x1 = np.random.randint(0, max_x - 1)
        y1 = np.random.randint(0, max_y - 1)
        x2 = np.random.randint(0, max_x - 1)
        y2 = np.random.randint(0, max_y - 1)
        if np.linalg.norm([x1 - x2, y1 - y2]) < min_dist:
            continue
        xy[cnt] = [x1, y1, x2, y2]
        cnt += 1
        if cnt == num:
            break

    # save config and data in a file
    with open("eval.txt", "w") as f:
        f.write(f"Max x, y: {max_x} {max_y}\n")
        f.write(f"Min dist: {min_dist}\n")
        f.write(f"Num: {num}\n")
        for i in range(num):
            f.write(f"{xy[i][0]} {xy[i][1]} {xy[i][2]} {xy[i][3]}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int)
    parser.add_argument("--max_x", type=int, default=10)
    parser.add_argument("--max_y", type=int, default=10)
    parser.add_argument("--min_dist", type=int, default=2)
    args = parser.parse_args()

    main(args.num, args.max_x, args.max_y, args.min_dist)
