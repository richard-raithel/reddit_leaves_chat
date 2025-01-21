import json
import os
from datetime import datetime, timezone

def preprocess_data(submissions_file, comments_file):
    if not os.path.exists(submissions_file):
        raise FileNotFoundError(f"File not found: {submissions_file}")
    if not os.path.exists(comments_file):
        raise FileNotFoundError(f"File not found: {comments_file}")

    with open(submissions_file, 'r', encoding='utf-8') as sub_file:
        submissions = [json.loads(line) for line in sub_file]

    with open(comments_file, 'r', encoding='utf-8') as com_file:
        comments = [json.loads(line) for line in com_file]

    comments_by_submission = {}
    for comment in comments:
        link_id = comment["link_id"].split("_")[1]

        # Convert created_utc from string to float or int
        float_timestamp = float(comment["created_utc"])
        created_time_str = datetime.fromtimestamp(float_timestamp, tz=timezone.utc).isoformat()

        if link_id not in comments_by_submission:
            comments_by_submission[link_id] = []
        comments_by_submission[link_id].append({
            "comment_id": comment["id"],
            "author": comment["author"],
            "body": comment["body"],
            "created_utc": created_time_str,  # Use the ISO 8601 string
            "score": comment["score"]
        })

    combined_data = []
    for submission in submissions:
        # Convert submission time
        float_timestamp = float(submission["created_utc"])
        created_time_str = datetime.fromtimestamp(float_timestamp, tz=timezone.utc).isoformat()

        combined_data.append({
            "submission_id": submission["id"],
            "subreddit": submission["subreddit"],
            "title": submission["title"],
            "body": submission["selftext"],
            "author": submission["author"],
            "created_utc": created_time_str,
            "score": submission["score"],
            "comments": comments_by_submission.get(submission["id"], [])
        })

    return combined_data

# Example usage
data = preprocess_data("../data/leaves_submissions.json", "../data/leaves_comments.json")

# Save the combined data
with open("../data/leaves_combined.json", "w", encoding='utf-8') as outfile:
    json.dump(data, outfile)

print("Data preprocessing complete!")