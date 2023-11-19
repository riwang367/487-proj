import re

def preprocess(test_str):
    # eyes = "[8:=;]"
    # nose = "['`\-]?"

    x = re.sub("https?:\/\/\S+\b|www\.(\w+\.)+\S*/", "<URL>", test_str)
    x = re.sub("\/", " / ", x) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    x = re.sub("u\/[A-Za-z0-9_-]+", "<USER>", x) #u/(username) format on Reddit
    x = re.sub("((8|:|=|;)('|`|\/\|-)*\)+)|(\(+)('|`|\/\|-)*(8|:|=|;))", "<SMILE>", x)
    x = re.sub("(8|:|=|;)('|`|\/\|-)*p+", "<LOLFACE>", x)
    x = re.sub("((8|:|=|;)('|`|\/\|-)*\(+)|(\)+)('|`|\/\|-)*(8|:|=|;))", "<SADFACE>", x)
    x = re.sub("(8|:|=|;)('|`|\/\|-)*(\/|l*)+", "<NEUTRALLFACE>", x)
    x = re.sub("<3","<HEART>", x)
    x = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*/", "<NUMBER>", x)
    x = re.sub("([!?.]){2,}/", repeat_sub(x), x) # Mark punctuation repeats
    x = re.sub("\b(\S*?)(.)\2{2,}\b", elongate_sub(x), x)
    x = re.sub("([^a-z0-9()<>'`\-]){2,}", downcase_sub(x), x)

    return x


def repeat_sub(str):
    # Mark punctuation repeats
    repeat = re.findall("([!?.]){2,}", str)
    new_str = ""
    for re in repeat:
        new_str += re[0]
    x = re.sub("([!?.]){2,}", new_str + " <REPEAT>", str) 
    return x


def elongate_sub(str):
    # Mark elongated words
    repeat = re.split("\b(\S*?)(.)\2{2,}\b", str)
    x = re.sub("\b(\S*?)(.)\2{2,}\b", repeat[0] + " <ELONG>", str)
    return x


def downcase_sub(str):
    return str.lower() + " <ALLCAPS>"