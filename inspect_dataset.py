from datasets import load_dataset

ds = load_dataset("bhavya777/GNHK-dataset")

print(ds)

sample = ds["train"][0]

print(sample.keys())
print(sample)