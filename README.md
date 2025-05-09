# sp2024-eabsace
Source code of senior project 2024 in the topic of Enhancing Aspect-Based Sentiment Analysis on Course Evaluation, Faculty of Information and Communication Technology, Mahidol University.

This experiment used 2 main models which is Bart-Large-CNN and Llama3.2-[n]b-instruct. While the source code split in to 4 main directory included both MAMS (Main task)
 and SAMS (Transfer Learning).

## Dataset:
- For MAMS: `./Dataset/original`, `./Dataset/backtran`, `./Dataset/gpt-gen`, `./Dataset/parrot` and `./Dataset/all`
- For SAMS: `./Dataset/coursera`
- For Instruction-Tuning: `./Dataset/instructed`

## Bart-Large-CNN:
- `cd Bart-Large-CNN-xxMS`
- In order to run Bart training, first you need to config `train.py` and `test_SAMS.py` to match your train and test dataset.
- The model configuration is in `train.py`.
- `python train.py`

## Llama3-Inst:
- `cd Llama3-Inst-xxMS`
- Our experiment chose to use `atsc_train.sh` or `Aspect Term Sentiment Classification` to do Aspect-Based Sentiment Analysis.
- The model configuration is in `Scripts/atsc_train.sh`
- `./atsc_train.sh`


## Reference
- [InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis (arXiv:2302.08624)](https://arxiv.org/abs/2302.08624)
- InstructABSA by kevinscaria: [kevinscaria/InstructABSA](https://github.com/kevinscaria/InstructABSA)

