# Celery-LaTex-OCR

Yet another LaTex OCR Project written in PyTorch, based on ConvNeXt and Transformer.

This project is the backend of [CeleryMath](https://github.com/MODCT/CeleryMath).

Give us a star if this project helps you :hugs:

## Develop

Any further developments and contributions are welcome :smile:

### Training Instruction

Follow the following instructions to train by yourself:

1. Create virtual environment.

```bash
poetry install
poetry shell
```

2. Create dataset.

You can download generated dataset from [here (2.05G)](https://drive.google.com/file/d/1yF9xSRevWvCPPgebkkPFg7GAy0ctyTyN/view?usp=sharing)
or generate by yourself with the following code, the generation may be slow:

```bash
python -m src.utils.latex2png -i dataset/data/full_math.txt -w dataset/data/full_set -b 1
```

3. Edit config file.

Edit the `src/config/config_convnext.json` and replace the dataset path to yours.

4. Run training

```bash
python -m src.train
```

### Dataset Instruction

If you have your own latex formula dataset, you can add them to `dataset/data/full_math.txt` and regenerate `tokenizers` and images.

1. tokenizers from hugging face was used, if you want to change formula file and output file location, edit `src/dataset.py`

```bash
python -m src.dataset
```

2. generate dataset, TexLive or MikeTex or similar program must be installed.

```bash
python -m src.utils.latex2png -i dataset/data/full_math.txt -w dataset/data/full_set -b 1
```

## Issues

Open an issue if you have any questions or a PR if you can fix it.

## TODO

- [ ] API
- [x] Desktop Deploy, see [CeleryMath](https://github.com/MODCT/CeleryMath)
- [x] ONNX
- [ ] Use pytorch-lightning to manage training and evaluation

## Acknowledgement

This project was inspired by the following project, and some methods or codes were also
borrowed from them, THANKS A LOT!!! :handshake:

1. LaTex-OCR, LICENSE: MIT, [https://github.com/lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)
2. LaTeX_OCR_PRO, LICENSE: GPL-3.0, [https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO](https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO)

## LICENSE

GPL-3.0, details [here](LICENSE)
