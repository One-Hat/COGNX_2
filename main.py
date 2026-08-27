import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="MNEMA: Neuromorphic Continual Learning Engine")
    parser.add_argument("--demo", action="store_true", help="Launch live webcam event-camera demonstrator")
    parser.add_argument("--benchmark", action="store_true", help="Run Split-MNIST Class-IL benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate benchmark evaluation plots")
    
    args = parser.parse_args()

    if args.demo:
        from demo.live_camera import run_live_demonstrator
        run_live_demonstrator()
    elif args.benchmark:
        from experiments.run_split_mnist_benchmark import run_benchmark
        run_benchmark()
    elif args.plot:
        from experiments.plot_benchmark import generate_benchmark_figures
        generate_benchmark_figures()
    else:
        print("Usage:")
        print("  uv run python main.py --demo       # Launch live webcam edge demonstrator")
        print("  uv run python main.py --benchmark  # Run Split-MNIST benchmark")
        print("  uv run python main.py --plot       # Generate evaluation figures")
        print("  uv run streamlit run demo/dashboard.py # Launch telemetry dashboard")

if __name__ == "__main__":
    main()