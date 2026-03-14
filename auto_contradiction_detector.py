import pandas as pd
from transformers import pipeline

print("Loading dataset...")

data = pd.read_csv("data/papers.csv")

print("Loading contradiction detection model...")

classifier = pipeline(
    "text-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1"
)

results = []

print("Checking contradictions across papers...\n")

for i in range(len(data)):
    for j in range(i + 1, len(data)):

        text1 = data.iloc[i]["conclusion"][:500]
        text2 = data.iloc[j]["conclusion"][:500]

        result = classifier(f"{text1} </s> {text2}")

        label = result[0]["label"]
        score = result[0]["score"]

        if label == "CONTRADICTION" and score > 0.80:

            results.append({
                "Paper 1": data.iloc[i]["title"],
                "Paper 2": data.iloc[j]["title"],
                "Confidence": score
            })

            print("Contradiction Found!")
            print("Paper 1:", data.iloc[i]["title"])
            print("Paper 2:", data.iloc[j]["title"])
            print("Score:", score)
            print("-------------")

print("\nTotal contradictions found:", len(results))


# Save results
df = pd.DataFrame(results)
df.to_csv("data/contradictions_found.csv", index=False)

print("Results saved to data/contradictions_found.csv")