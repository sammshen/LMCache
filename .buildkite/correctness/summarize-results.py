import pandas as pd
import matplotlib.pyplot as plt
import json

def load_results(filename):
    rows = []
    with open(filename, "r") as f:
        for line in f:
            row = json.loads(line)
            for subject, data in row.items():
                rows.append({
                    "Subject": subject,
                    "Accuracy": round(data["accuracy"] * 100, 2),
                    "Questions": data["num_questions"]
                })
    df = pd.DataFrame(rows)
    return df

def render_table(df, output_file):
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(df)))  # height scales with number of rows
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # adjust as needed
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model.split("/")[-1].replace(":", "_").replace("/", "_")

    df1 = load_results(f"vllm-{model_name}.jsonl")
    df2 = load_results(f"lmcache-{model_name}.jsonl")

    df1["Run"] = "vLLM"
    df2["Run"] = "LMCache"
    df_combined = pd.concat([df1, df2])
    df_combined = df_combined.sort_values(by=["Subject", "Run"])

    output_filename = f"results-{model_name}.png"
    render_table(df_combined, output_filename)