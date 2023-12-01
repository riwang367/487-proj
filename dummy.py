import preprocess
import csv

'''This script outputs the processed first 1000 lines of a large file.'''

def main():
    num_lines = 1000
    count = get_hp(0, num_lines)
    todo = ["entertainment_starwars.csv", "gaming_leagueoflegends.csv",
            "gaming_pokemon.csv", "television_gameofthrones.csv", "television_himym.csv",
            "television_mylittlepony.csv", "television_startrek.csv"]
    todo = ['datasets/' + s for s in todo]
    for file_name in todo:
        count = get_lines(count, num_lines, file_name)
    return 0

# special snowflake hp doesn't work in our normal code
def get_hp(count, num_lines):
    i = 0
    with open("datasets/entertainment_harrypotter.csv", "r") as file:
        for line in csv.reader(file):
            _, _, text, _, cat, _, _, _, _, _, _, _, _ = line
            if text != "":
                # preprocess
                text = preprocess.preprocess(text)
                print(f"{count},{cat},{text}")
                i += 1
                count += 1
                if i >= num_lines:
                    return count
    return count


def get_lines(count, num_lines, file_name):
    i = 0
    with open(file_name, "r") as file:
       for line in csv.reader(file):
           text = line[1]
           cat = line[3]
           if text != "":
                # preprocess
                text = preprocess.preprocess(text)
                print(f"{count},{cat},{text}")
                i += 1
                count += 1
                if i >= num_lines:
                    return count
    return count


if __name__ == "__main__":
    main()