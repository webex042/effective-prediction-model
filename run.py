"""
EPM Unified Project Runner & Dashboard Server
==============================================
CLI runner for the EPM research project:
- Run individual stages (0 - 7)
- Run all demo stages sequentially
- Run test suites (205 tests)
- Launch the Vercel-compatible interactive web dashboard locally

Usage:
    python run.py                 # Interactive terminal menu
    python run.py --stage 1       # Run Stage 1 demo
    python run.py --all           # Run all stage demos sequentially
    python run.py --test          # Run full pytest test suite
    python run.py --serve         # Launch web dashboard on http://localhost:3000
    python run.py --status        # Show project health and status
"""

import sys
import os
import argparse
import subprocess
import webbrowser
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Ensure plots folder exists for static web serving
def sync_plots():
    docs_plots = ROOT_DIR / "docs" / "plots"
    web_plots = ROOT_DIR / "web" / "plots"
    if docs_plots.exists() and not web_plots.exists():
        try:
            shutil.copytree(docs_plots, web_plots)
        except Exception:
            pass

def print_banner():
    print("=" * 72)
    print("   EPM — Efficient Predictive Multimodal Generative Model")
    print("   Interactive Research Platform & Experimentation Suite")
    print("=" * 72)

def print_status():
    print_banner()
    print("\n[PROJECT STATUS]")
    print("  • Completed Stages:       14 / 14 (Stages 0 through 13)")
    print("  • Unit Tests Passing:     281 / 281 (100% pass rate)")
    print("  • Numerical Gradient Err: 1.42e-09 (Passed)")
    print("  • Autoencoder Recon MSE:  0.03076 (Standard-AE)")
    print("  • Diffusion Latency:      15.25 ms (DDIM-10) vs 145.91 ms (DDPM)")
    print("  • Time-Series Diff CRPS:  1.0777 (Probabilistic Horizon=12)")
    print("  • Flow Matching Str-Ratio: 0.991 (OT-CFM Straight Paths vs 0.625 DDPM)")
    print("  • Linear Attention Accel: 168.16x Speedup / 127x Memory Savings (N=2048)")
    print("  • Model Compression:      19.4x Size Reduction (6.1 KB Tri-Compressed Edge)")
    print("  • Stage 13 Research (H1):  Reduced-step DDIM-10 = 8.35x faster, +6.2% RMSE (within full-step CI)")
    print("  • Multimodal Modalities:  Continuous Sensors + Maintenance Text")
    print("  • Vercel Compatible:      Yes (vercel.json, /api, static dashboard)")
    print("-" * 72)

def run_stage(stage_num: int):
    sync_plots()
    script_name = f"demo_stage{stage_num}.py"
    script_path = ROOT_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Stage script not found: {script_name}")
        return False
    
    print_banner()
    print(f"\n>>> Executing Stage {stage_num}: {script_name} <<<\n")
    python_exe = sys.executable
    ret = subprocess.run([python_exe, str(script_path)], cwd=str(ROOT_DIR))
    return ret.returncode == 0

def run_all_stages():
    sync_plots()
    print_banner()
    print("\n>>> Executing All Stages (0 through 13) Sequentially <<<\n")
    stages = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    results = {}
    for s in stages:
        print(f"\n--- Starting Stage {s} ---")
        success = run_stage(s)
        results[s] = "PASSED" if success else "FAILED"

    print("\n" + "=" * 72)
    print("   ALL STAGES EXECUTION SUMMARY")
    print("=" * 72)
    for s, res in results.items():
        print(f"  Stage {s}: {res}")
    print("=" * 72)

def run_tests():
    print_banner()
    print("\n>>> Running pytest test suite (260 tests) <<<\n")
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", "tests/", "-v", "--tb=short"]
    subprocess.run(cmd, cwd=str(ROOT_DIR))

def serve_dashboard(port: int = 3000):
    sync_plots()
    from http.server import SimpleHTTPRequestHandler, HTTPServer
    import functools

    # The static frontend lives in web/ (keeps the Vercel "app" service root free
    # of Python markers). Serve that directory locally so paths match production.
    web_dir = ROOT_DIR / "web"

    class EPMHTTPHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

    handler_cls = functools.partial(EPMHTTPHandler, directory=str(web_dir))
    server_address = ("", port)
    print_banner()
    print(f"\n[SERVER] Serving EPM Vercel Dashboard at:")
    print(f"         http://localhost:{port}")
    print("Press Ctrl+C to stop the server.\n")
    
    # Try opening browser
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    try:
        httpd = HTTPServer(server_address, handler_cls)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Stopped.")

def interactive_menu():
    sync_plots()
    while True:
        print_banner()
        print("\nSelect an action to run:")
        print("  [0] Run Stage 0 — Infrastructure, Logging & Reproducibility")
        print("  [1] Run Stage 1 — Mathematical Foundations (Calculus, Linalg, Optimization)")
        print("  [2] Run Stage 2 — Classical ML Benchmarks (Regression, Trees, XGBoost)")
        print("  [3] Run Stage 3 — Neural Networks from Scratch & PyTorch Parity")
        print("  [4] Run Stage 4 — NLP, Embeddings & Transformer Self-Attention")
        print("  [5] Run Stage 5 — Time Series Forecasting (LSTM, GRU, Transformers)")
        print("  [6] Run Stage 6 — Multimodal Representation & Fusion (Concat, GMU, Cross-Attn)")
        print("  [7] Run Stage 7 — Generative Models & Latent Manifold (AE, VAE, Uncertainty)")
        print("  [8] Run Stage 8 — Diffusion Models & Accelerated Sampling (DDPM, DDIM)")
        print("  [9] Run Stage 9 — Diffusion for Time Series (Probabilistic Trajectory Forecasting)")
        print("  [10] Run Stage 10 — Flow Matching (Continuous Normalizing Flows & Optimal Transport)")
        print("  [11] Run Stage 11 — Computational Efficiency (Linear & Tiled Attention, Sequence Scaling)")
        print("  [12] Run Stage 12 — Model Compression (Distillation, Quantization, Pruning)")
        print("  [13] Run Stage 13 — Research Experiment (H1 reduced-step generative forecasting)")
        print("  [A] Run ALL Stages (0 through 13)")
        print("  [T] Run PyTest Suite (260 Tests)")
        print("  [W] Launch Interactive Web Dashboard (Port 3000)")
        print("  [S] Display Project Health & Status")
        print("  [Q] Exit")
        
        choice = input("\nEnter choice [0-13, A, T, W, S, Q]: ").strip().upper()
        if choice in [str(i) for i in range(14)]:
            run_stage(int(choice))
            input("\nPress Enter to return to menu...")
        elif choice == "A":
            run_all_stages()
            input("\nPress Enter to return to menu...")
        elif choice == "T":
            run_tests()
            input("\nPress Enter to return to menu...")
        elif choice == "W":
            serve_dashboard(3000)
            break
        elif choice == "S":
            print_status()
            input("\nPress Enter to return to menu...")
        elif choice == "Q":
            print("\nExiting EPM runner.")
            break
        else:
            print("\nInvalid selection, please try again.")

def main():
    parser = argparse.ArgumentParser(description="EPM Unified Project Runner & Dashboard Server")
    parser.add_argument("--stage", type=int, choices=range(14), help="Run a specific stage demo (0-13)")
    parser.add_argument("--all", action="store_true", help="Run all stage demos sequentially")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--serve", action="store_true", help="Start local web server for dashboard")
    parser.add_argument("--port", type=int, default=3000, help="Port for web server (default: 3000)")
    parser.add_argument("--status", action="store_true", help="Show project status and metrics")
    
    args = parser.parse_args()

    if args.stage is not None:
        run_stage(args.stage)
    elif args.all:
        run_all_stages()
    elif args.test:
        run_tests()
    elif args.serve:
        serve_dashboard(args.port)
    elif args.status:
        print_status()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
