import preprocess
import csv

'''This script processes our datasets.'''

def main():
    todo = ["entertainment_harrypotter.csv", "entertainment_starwars.csv", "gaming_leagueoflegends.csv",
            "gaming_pokemon.csv", "television_gameofthrones.csv", "television_himym.csv",
            "television_mylittlepony.csv", "television_startrek.csv"]
    todo = ['datasets/' + s for s in todo]
    count = 0

    '''Concatenate first 1000 lines of each dataset'''
    print("1000")
    for file_name in todo:
        count = get_lines(count, 1000, file_name, "datasets/final/1000.csv")

    '''Concatenate first 12500 lines of each dataset'''
    print("12500")
    count = 0 # Reset line start
    for file_name in todo:
        count = get_lines(count, 12500, file_name, "datasets/final/fullset.csv")

    '''Concatenate last 32 lines of each dataset'''
    print("-32")
    count = 0
    for file_name in todo:
        count = get_lines_from_end(count, 32, file_name, "datasets/final/eval.csv")

    print("Done")
    return 0


def get_lines(line_start, num_lines, input_file, output_file):
    count = line_start
    i = 0

    with open(input_file, "r") as file:
       for line in csv.reader(file):
           text = line[1]
           cat = line[3]
           if text != "" and text != " deleted " and text != " removed ":
                # Preprocess the data
                text = preprocess.preprocess(text)

                # Write to file
                with open(output_file, "a", newline='') as outfile:
                    out = csv.writer(outfile, delimiter=',')
                    out.writerow([count, cat, text])
                
                i += 1
                count += 1
                if i >= num_lines:
                    return count

    return count

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

    for line in reversed(full_text):
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