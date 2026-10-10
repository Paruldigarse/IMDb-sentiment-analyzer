"""
run_pipeline.py
───────────────
One-command runner: preprocess → train → evaluate → launch API.

Usage:
  python run_pipeline.py              # full pipeline + start server
  python run_pipeline.py --skip-train # skip preprocessing/training (model exists)
  python run_pipeline.py --no-server  # run pipeline only, do not start Flask
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="IMDB Sentiment Pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip preprocessing and training (reuse saved model)")
    parser.add_argument("--no-server",  action="store_true",
                        help="Run pipeline only; do not start the Flask server")
    args = parser.parse_args()

    if not args.skip_train:
        print("=" * 60)
        print(" STEP 1 — Preprocessing")
        print("=" * 60)
        import preprocess
        preprocess.run()

        print("\n" + "=" * 60)
        print(" STEP 2 — Training")
        print("=" * 60)
        import train
        train.run()

    print("\n" + "=" * 60)
    print(" STEP 3 — Evaluation")
    print("=" * 60)
    import evaluate
    evaluate.run()

    if not args.no_server:
        print("\n" + "=" * 60)
        print(" STEP 4 — Starting Flask API  (http://localhost:5000)")
        print("         Frontend → http://localhost:5000/app")
        print("         Report  → http://localhost:5000/report")
        print("=" * 60 + "\n")
        import subprocess, os
        subprocess.run(
            [sys.executable, os.path.join("backend", "app.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
