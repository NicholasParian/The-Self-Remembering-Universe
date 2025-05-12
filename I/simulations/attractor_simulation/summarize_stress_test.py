import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summarize_stress_test(results_dir="results/stress_test", output_csv="results/stress_summary.csv"):
    data = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".npz"):
            continue

        path = os.path.join(results_dir, fname)
        result = np.load(path)
        collapsed = result.get("collapsed", False)
        sigma = float(result.get("sigma", -1))
        lambda_E = float(result.get("lambda_E", -1))

        data.append({"sigma": sigma, "lambda_E": lambda_E, "collapsed": bool(collapsed)})

    df = pd.DataFrame(data)
    df = df.sort_values(by=["sigma", "lambda_E"])
    df.to_csv(output_csv, index=False)
    print(f"📄 Saved summary to {output_csv}")
    print(df)

    # Optional: Plot heatmap
    pivot = df.pivot(index="sigma", columns="lambda_E", values="collapsed")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot, cmap="RdYlGn_r", origin="lower", aspect="auto")
    plt.colorbar(label="Collapse (1=True, 0=False)")
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.title("Collapse Map: sigma vs. lambda_E")
    plt.xlabel("λ_E")
    plt.ylabel("σ")
    plt.tight_layout()
    plt.savefig("results/collapse_map.png")
    plt.show()

if __name__ == "__main__":
    summarize_stress_test()
