import arxiv
import pandas as pd
import time

print("Fetching research papers from arXiv...")

search = arxiv.Search(
    query="machine learning",
    max_results=20,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()

titles = []
conclusions = []

for result in client.results(search):

    titles.append(result.title)
    conclusions.append(result.summary)

    time.sleep(2)   # important delay to avoid API limit

data = pd.DataFrame({
    "title": titles,
    "conclusion": conclusions
})

data.to_csv("data/papers.csv", index=False)

print("Dataset created successfully!")
print("Total papers:", len(data))