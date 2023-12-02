import preprocess
import csv

''' Get 32 lines from the middle-end of LOL dataset to avoid Pokemon skew
    Last 32 lines of the LOL dataset appear to be coming from the same post comparing
    different LOL characters to Pokemon types.
'''

def main():
    get_lines_from_end(0, 32, "datasets/gaming_leagueoflegends.csv", "datasets/32_lol.csv")

def get_lines_from_end(line_start, num_lines, input_file, output_file):
    count = line_start
    i = 0 # how many times to loop
    full_text = []

    # Read entire file first and discard empty lines
    with open(input_file, "r") as file:
       for line in csv.reader(file):
            line_id = line[0]            
            text = line[1]
            cat = line[3]

            if text != "" and text != " deleted " and text != " removed ":
                full_text.append([line_id, text, cat])
            else:
                continue


    ignore = 150
    for line in reversed(full_text):
        if (ignore > 0):
            ignore -= 1
            continue
        text = preprocess.preprocess(line[1])
        cat = line[2]
        with open(output_file, "a", newline='') as outfile:
            out = csv.writer(outfile, delimiter=',')
            out.writerow([count, cat, text])

        i += 1
        count += 1
        if i >= num_lines:
            return count

    return count

if __name__ == "__main__":
    main()