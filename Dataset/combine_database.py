OUTPUT_DIR = "all"
FOLD = 1
for f in range(5):
    print(("=" * 10) + " FOLD " + str(f+1) + " " + ("=" * 10))
    COMBINED_DATASET = []
    DATASET = []
    CUR_DATA:str = ""
    FILELIST = [
        f"Dataset/backtran/org-bt_all-fold{str(f+1)}.txt",
        f"Dataset/gpt-gen/gpt4omini-fold{str(f+1)}.txt",
        f"Dataset/original/org-fold{str(f+1)}.txt",
        f"Dataset/parrot/org-pp-fold{str(f+1)}.txt"
    ]

    for PATH in FILELIST:
        with open(PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()
            CUR_DATA = str(PATH).split("/")[1]
            print(f"{CUR_DATA}: {len(lines)}")
            COMBINED_DATASET.extend(lines)
    print(f"Combined dataset length: {len(COMBINED_DATASET)}")

    for data in COMBINED_DATASET:
        if data not in DATASET:
            DATASET.append(data)
    print(f"No duplicated dataset length: {len(DATASET)}")

    with open(f"Dataset/{OUTPUT_DIR}/all-fold{str(f+1)}.txt", "a", encoding="utf-8") as output_file:
        output_file.writelines(DATASET)
