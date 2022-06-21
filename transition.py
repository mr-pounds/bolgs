"""
Name         : md 格式优化
Description  : 将 obsidian 的标签格式转化为 hexo 支持的格式
Version      : 1.0.1
Author       : zzz
Date         : 2022-06-21 15:13:42
LastEditors  : zzz
LastEditTime : 2022-06-21 15:21:42
"""
import os

target_dir = "source/_posts/"


def extract_hash_text(text):
    text = text.split("#")[1:]
    if text == []:
        return None
    return [i.strip(" ") for i in text]


def extract_tags(text):
    text = text.split("/")
    if text[1:] == []:
        return text[0]
    return text[1]


for file_name in os.listdir(target_dir):
    file_path = os.path.join(target_dir, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            result = []
            tags = []
            catagories = []
            for i, line in enumerate(lines):
                if line.startswith("知识类型"):
                    c = extract_hash_text(line)
                    if c is not None:
                        catagories.extend([extract_tags(i) for i in c])
                    continue
                if line.startswith("知识标签") or line.startswith("相关企业"):
                    c = extract_hash_text(line)
                    if c is not None:
                        tags.extend([extract_tags(i) for i in c])
                    continue
                result.append(line)

        if catagories != [] or tags != []:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(result[:3])
                f.write("\ncategories:\n")
                f.writelines([f"- {i}\n" for i in catagories])
                f.writelines(
                    [
                        "tags:\n",
                    ]
                )
                f.writelines([f"- {i}\n" for i in tags])
                f.writelines(result[3:])

# tt = "知识类型： #"
# print(extract_hash_text(tt))
