import unittest
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from ..deploy.src.utils.config import Config
from ..deploy.src.models.model import get_model

class DeployCMDTest(unittest.TestCase):
    img_paths = None
    tex_strs = None
    cur_bs = 0
    batch_size = 50
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_img_path = "test/test_img"
        self.tex_file = "dataset/data/full_math.txt"

        self.conf = Config("deploy/conf/conf.json")
        self.model = get_model(self.conf)
    
    def setUp(self) -> None:
        self.img_paths = glob.glob(self.test_img_path+"/*.png", recursive=True)
        with open(self.tex_file, "r") as f:
            texs = f.readlines()
        idxs = [int(p.split("/")[-1].split(".")[0]) for p in self.img_paths]
        self.tex_strs = [texs[idx] for idx in idxs]
        return super().setUp()

    def test_cmd(self):
        true_strs = []
        pred_strs = []
        while self.cur_bs < np.ceil(len(self.img_paths) / self.batch_size):
            imgs = []
            # one batch
            for i in tqdm(range(self.cur_bs, self.cur_bs+self.batch_size)):
                if i > len(self.img_paths):
                    break
                p = self.img_paths[i]
            # for i, p in enumerate(tqdm(self.img_paths[self.cur_bs:self.cur_bs+self.batch_size])):
                tex = self.model.tokenizer.encode(self.tex_strs[i])
                true_strs.append(
                    self.model.detokenize([1, *tex.ids, 2])
                )
                img = Image.open(p).convert("L")
                imgs.append(img)
            out_strs = self.model(imgs, out_list=True)
            pred_strs.extend(out_strs)
            bleu4 = 0.0
            for j, k in zip(true_strs, pred_strs):
                bleu4 += sentence_bleu(j, k)
            bleu4 = bleu4 / len(true_strs)
            print(f"[{i}], BLEU4: {bleu4:.4f}")
            self.cur_bs += self.batch_size

        bleu4 = 0.0
        for j, k in zip(true_strs, pred_strs):
            bleu4 += sentence_bleu(j, k)
        bleu4 = bleu4 / len(true_strs)

        print(f"BLEU4: {bleu4:.4f}")


if __name__ == "__main__":
    unittest.main()
