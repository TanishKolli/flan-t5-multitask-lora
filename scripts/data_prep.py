import json
import random
from datasets import load_dataset
from tqdm import tqdm

def format_race_with_difficulty(num_middle=3000, num_high=7000, medium_ratio=0.4, seed=42):
    random.seed(seed)
    formatted = []

    # Load RACE-M and RACE-H
    race_m = load_dataset("race", "middle", split="train").shuffle(seed=seed)
    race_h = load_dataset("race", "high", split="train").shuffle(seed=seed)

    # Helper to process either RACE-M or RACE-H
    def process_race(split, max_articles, base_diff, allow_medium=False):
        articles = {}
        for sample in split:
            aid = sample['example_id'].split('_')[0]
            if aid not in articles:
                articles[aid] = {
                    "passage": sample['article'],
                    "questions": []
                }

            try:
                q = sample['question']
                opts = sample['options']
                ans_char = sample['answer']
                ans_idx = ord(ans_char) - ord('A')
                ans_text = opts[ans_idx]
                difficulty = base_diff
                if allow_medium and random.random() < medium_ratio:
                    difficulty = "Medium"
                articles[aid]["questions"].append({
                    "question_text": q,
                    "options": opts,
                    "correct_answer_char": ans_char,
                    "correct_answer_text": ans_text,
                    "difficulty": difficulty
                })
            except Exception:
                continue

        valid_articles = [a for a in articles.values() if a["questions"]]
        return valid_articles[:max_articles]

    # Process both RACE splits
    middle_articles = process_race(race_m, num_middle, "Easy", False)
    high_articles = process_race(race_h, num_high, "Hard", True)
    all_articles = middle_articles + high_articles
    random.shuffle(all_articles)

    # Format for training
    for article in tqdm(all_articles, desc="Formatting RACE"):
        passage = article["passage"]
        qs = article["questions"]

        if not qs:
            continue

        counts = {"Easy": 0, "Medium": 0, "Hard": 0}
        mcq_lines = []
        for i, q in enumerate(qs):
            counts[q["difficulty"]] += 1
            options = " ".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(q["options"])])
            mcq_lines.append(f"{i+1}. Question: {q['question_text']}\n   Options:\n   {options}\n   Correct: {q['correct_answer_char']}\n   Difficulty: {q['difficulty']}")

        formatted.append({
            "input_text": f"Generate MCQs (Easy: {counts['Easy']}, Medium: {counts['Medium']}, Hard: {counts['Hard']}): {passage}",
            "target_text": "MCQs:\n" + "\n".join(mcq_lines)
        })

    print(f"Total formatted MCQ samples: {len(formatted)}")
    return formatted

def format_cnn_dailymail(num_samples=10000, seed=42):
    print("Loading CNN/DailyMail...")
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train").shuffle(seed=seed)
    formatted = []

    for i, sample in tqdm(enumerate(cnn_dm), desc="Formatting CNN/DailyMail", total=num_samples):
        if i >= num_samples:
            break
        formatted.append({
            "input_text": f"Summarize the following article: {sample['article']}",
            "target_text": f"Summary: {sample['highlights']}"
        })

    print(f"Total formatted summarization samples: {len(formatted)}")
    return formatted

def main():
    race_samples = format_race_with_difficulty()
    cnn_dm_samples = format_cnn_dailymail()

    all_samples = race_samples + cnn_dm_samples
    random.shuffle(all_samples)

    output_file = "combined_mcq_summarization_finetune.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(all_samples, desc="Saving combined dataset"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved combined dataset to {output_file}")
    print("--- Sample MCQ ---")
    for s in all_samples:
        if "Generate MCQs" in s["input_text"]:
            print("Input:", s["input_text"][:300], "...\nTarget:", s["target_text"][:300], "...\n")
            break
    print("--- Sample Summary ---")
    for s in all_samples:
        if "Summarize the following article" in s["input_text"]:
            print("Input:", s["input_text"][:300], "...\nTarget:", s["target_text"][:300], "...\n")
            break

if __name__ == "__main__":
    main()
