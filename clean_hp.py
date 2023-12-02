import csv

'''This script cleans the Harry Potter dataset to be in the same format as our other data.'''

def main():
    full_text = []

    with open("datasets/0orig_entertainment_harrypotter.csv", "r") as file:
        for line in csv.reader(file):
            line_id, _, text, meta, subreddit, cat, time, author, ups, downs, authorkarma, linkkarma, authorisgold = line
            full_text.append([line_id, text, meta, subreddit, cat, time, author,
                        ups, downs, authorkarma, linkkarma, authorisgold])
    
    with open("datasets/entertainment_harrypotter.csv", "w", newline='') as output_file:
        out = csv.writer(output_file, delimiter=',')
        out.writerows(full_text)

    print("done")
    return 0

if __name__ == "__main__":
    main()