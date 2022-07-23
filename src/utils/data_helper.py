import json
import re
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def preprocessing(sentence: str, lang: str) -> str:
    if lang in ["gang", "jeon"]:
        try:
            sentence = sentence.split("\t")[1].strip()
        except IndexError:
            sentence = sentence
    elif lang == "jeju":
        sentence = re.sub('^JJ\w+', "", sentence).strip()

    sentence = sentence.replace("(())", "")
    sentence = re.sub('^\d:', "", sentence)
    sentence = re.sub('(@\w+)', "", sentence)
    sentence = sentence.replace("  ", "")
    return sentence.strip()


def preprocessing2(sentence: str):
    sentence = sentence.replace("(())", "")
    sentence = re.sub(r'\&[^)]*\&', '', sentence)
    sentence = re.sub(r'\{[^)]*\}', '', sentence)
    sentence = re.sub(r'\([^)]*\)', '', sentence)
    sentence = re.sub(r'\-[^)]*\-', '', sentence)
    sentence = sentence.replace("  ", " ")
    return sentence


def split_train_eval_test(file_path: str):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    X = df["표준어"]
    y = df["전라도"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000, shuffle=True, random_state=1004)
    x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=10000, shuffle=True, random_state=1004)
    lang = "jeon"
    save_path = f"../Data/{lang}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(f"{save_path}/ko.train", "w", encoding="utf-8-sig") as f:
        for i in x_train:
            f.write(str(i) + "\n")

    with open(f"{save_path}/{lang}.train", "w", encoding="utf-8-sig") as f:
        for i in y_train:
            f.write(str(i) + "\n")

    with open(f"{save_path}/ko.eval", "w", encoding="utf-8-sig") as f:
        for i in x_eval:
            f.write(str(i) + "\n")

    with open(f"{save_path}/{lang}.eval", "w", encoding="utf-8-sig") as f:
        for i in y_eval:
            f.write(str(i) + "\n")

    with open(f"{save_path}/ko.test", "w", encoding="utf-8-sig") as f:
        for i in x_test:
            f.write(str(i) + "\n")

    with open(f"{save_path}/{lang}.test", "w", encoding="utf-8-sig") as f:
        for i in y_test:
            f.write(i + "\n")


if __name__ == "__main__":
    split_train_eval_test("../../Data/modify_data/jeon_data.csv")

    exit()
    dialects = {"전라도": "jeon", "제주도": "jeju", "강원도": "gang", "경상도": "gy", "충청도": "chung"}
    for dialect in dialects.keys():
        path = "D:/Dataset/한국어 방언 발화 데이터({})/Training/[라벨]{}_학습데이터_1/".format(dialect, dialect)
        lst = [i for i in os.listdir(path) if i.endswith(".json")]
        result = []
        for i in tqdm(lst):
            try:
                with open(path + i, "r", encoding="utf-8-sig") as json_data:
                    sentence = json.load(json_data)
            except Exception as e:
                print(path + i)

            for sen in sentence["utterance"]:
                standard_sentence = preprocessing2(sen['standard_form'])
                dialect_sentence = preprocessing2(sen['dialect_form'])
                result.append([standard_sentence, dialect_sentence])

        df = pd.DataFrame(result, columns=["표준어", dialect])
        # if not os.path.isdir(f"./{dialects[dialect]}"):
        #     os.makedirs(f"./{dialects[dialect]}")
        #
        # with open(f"./{dialects[dialect]}/{dialects[dialect]}_data.txt", "w", encoding="utf-8-sig") as f:
        #     for sentence in df[dialect]:
        #         f.write(sentence + "\n")
        #
        # with open(f"./{dialects[dialect]}/ko_data.txt", "w", encoding="utf-8-sig") as f:
        #     for sentence in df["표준어"]:
        #         f.write(sentence + "\n")
        #
        df.to_csv("./{}_data.csv".format(dialects[dialect]), index=False, encoding="utf-8-sig")