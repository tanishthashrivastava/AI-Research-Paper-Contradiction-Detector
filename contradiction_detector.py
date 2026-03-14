from transformers import pipeline

print("Loading contradiction detection model...")

# Use smaller model (faster + less RAM)
classifier = pipeline(
    "text-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1"
)

text1 = input("Enter Paper 1 conclusion: ")
text2 = input("Enter Paper 2 conclusion: ")

result = classifier(f"{text1} </s> {text2}")

label = result[0]["label"]
score = result[0]["score"]

print("\nResult:", result)

if label == "CONTRADICTION":
    print("\n⚠️ These papers contradict each other.")
elif label == "ENTAILMENT":
    print("\nThese papers support each other.")
else:
    print("\nRelationship unclear.")

print("Confidence Score:", score)