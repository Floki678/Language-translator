import os
def load_data(english=True):
    data_dir="data/"
    if(english):
        filename="europarl-v7.{0}-en.en".format("fr")
    else:
        filename="europarl-v7.{0}-en.{0}".format("fr")
    start="aaaa "
    end=" zzzz"
    path = os.path.join(data_dir, filename)
    with open(path, encoding="utf-8") as file:
        texts = [start + line.strip() + end for line in file]
    return texts
#print(texts[1])
