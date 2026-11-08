import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def args(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--no-plot", action="store_true", help="print results only (used by CI)")
    return p


def show(fig, name):
    os.makedirs("out", exist_ok=True)
    path = os.path.join("out", name)
    fig.savefig(path, dpi=100)
    print("saved", path)
