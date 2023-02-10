# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import os
from pathlib import Path
import random

import re
import shutil
import sys
import io
import glob
import shlex
import subprocess
import traceback
from multiprocessing import Pool
from typing import List, Union
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm


class Latex(object):
    BASE = r'''
\documentclass[multi={mymath},border=1pt]{standalone}
\usepackage{amssymb}
\usepackage{unicode-math}
\usepackage{amsmath}
\usepackage[version=4]{mhchem}
\setmathfont{%s}
\newenvironment{mymath}{$\displaystyle}{$}
\begin{document}
%s
\end{document}
'''
    BASE_EQ = r"""
\begin{mymath}
%s
\end{mymath}"""
    def __init__(self, math: List, workdir: Union[Path, str], img_names: Union[List[str], None] = None):
        self.math: List[str] = math
        self.fonts = [
            "Latin Modern Math", "Asana Math", "XITS Math", "Noto Sans Math",
            "DejaVu Math TeX Gyre", "Cambria Math", "STIX Two Math"
        ]
        self.workdir = Path(workdir)
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True)
        self.img_names = img_names or [str(i) for i in range(len(self.math))]
        assert len(self.math) == len(self.img_names), "equation list must have exact same length as their new names"
        # used for calculate error formula index
        self.prefix_line = self.BASE.split("\n").index("%s")
        self.failed_idxs = []

    def write(self, idxs: List[int], batch: int):
        try:
            font = random.choice(self.fonts)
            names = [str(self.img_names[idx]) for idx in idxs]
            eqs = "\n".join([self.BASE_EQ % (eq.strip()) for eq in [self.math[idx] for idx in idxs]])
            document = self.BASE % (font, eqs, )
            texfile = str((self.workdir / f"{'_'.join(names)}.tex").absolute())
            with open(texfile, "w") as f:
                f.write(document)
                # print(texfile)
            # print(document)
            self.convert_file(texfile, str(self.workdir.absolute()), idxs=idxs, batch=batch)
        except Exception as e:

            print(e)
        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile: str, workdir: str, idxs: List[int], batch:int):
        """Convert one or more tex file to pdf, then to png"""
        infile = infile.replace('\\', '/')
        workdir = workdir.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = (
                "xelatex "
                "-interaction nonstopmode "
                "-file-line-error "
                f"-output-directory {workdir} "
                f"{infile} "
            )
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            sout, serr = p.communicate()
            if p.returncode != 0:
                self.failed_idxs.extend(idxs)
                raise Exception(f"convert {idxs}.tex to pdf failed")
            # extract error line from sout
            # error_index, status = self.extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
            # # extract success rendered equation
            # if error_index != []:
            #     # offset index start from 0, same as self.math
            #     error_index = [int(e)-self.prefix_line-1 for e in error_index]
            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            result, _ = self.extract(
                # text=sout, expression="Output written on %s \((\d+)? page" % pdffile
                text=sout, expression=r"\[\d+\]"
            )
            if len(result) != len(idxs):
                raise Exception(
                    'xelatex rendering error, generated %d formula\'s page, but the total number of formulas is %d.' % (
                    len(result), len(idxs)))
            # pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))
            pngfile = os.path.join(workdir, infile.replace('.tex', ''))
            dpi = random.randint(160, 400)
            cmd = (
                # "convert "
                # "-alpha off "
                # f"-density {self.dpi} "
                # "-colorspace gray "
                # "-quality 90 "
                "pdftoppm "
                "-png "
                "-gray "
                "-forcenum "
                # "-singlefile "
                f"-r {dpi} "
                f"{pdffile} "
                f"{pngfile} "
            )
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            if p.returncode != 0:
                self.failed_idxs.extend(idxs)
                raise Exception(
                    'PDF2PNG error', serr, cmd,
                    os.path.exists(pdffile),
                    os.path.exists(infile)
                )
            for idx in idxs:
                # batch no more than 1000
                if len(idxs) >= 100:
                    src = pngfile + f"-{idx%batch+1:03d}.png"
                if len(idxs) >= 10:
                    src = pngfile + f"-{idx%batch+1:02d}.png"
                else:
                    src = pngfile + f"-{idx%batch+1}.png"
                dst = os.path.dirname(pngfile) + f"/{idx}.png"
                shutil.move(src, dst)
        except Exception as e:
            # self.save_failed_idxs(f"tmp/failed/failed_idxs_{self.failed_idxs[0]}-{self.failed_idxs[-1]}.txt")
            print(e)
        finally:
            # Cleanup temporaries
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)

    def save_failed_idxs(self, p):
        with open(p, "w") as f:
            self.failed_idxs = [str(i) for i in self.failed_idxs]
            f.write("\n".join(self.failed_idxs))
            print(f"saved failed idxs to {p}")

    def extract(self, text: str, expression: str=None):
        """extract text from text by regular expression
        Args:
            text (str): input text
            expression (str, optional): regular expression. Defaults to None.
        """
        # print(f"text: {text}\n expr: {expression}")
        try:
            pattern = re.compile(expression)
            results = re.findall(pattern, text.replace("\n", ""))
            # print(results)
            return results, len(results) != 0
        except Exception:
            traceback.print_exc()


def main(in_file: str = "", work_dir: str = "", batch=1):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    with open(in_file, "r") as fin:
        eqs = fin.readlines()
    eq_names = [i for i,_ in enumerate(eqs)]
    gen_latex = Latex(eqs, workdir=work_dir, img_names=eq_names)
    # gen_latex.write(list(range(10)))

    pool = Pool(20)
    for idx in tqdm(range(0, len(eq_names), batch)):
        if (Path(work_dir) / f"{eq_names[idx]}.png").exists():
            # print(f"Skip {idx} ...")
            continue
        idxs = list(range(idx, idx+batch))
        pool.apply_async(gen_latex.write, (idxs, batch))
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-file", dest="input_file", type=str, default=None, help="equations to render in a file")
    parser.add_argument("-w", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=None)

    args = parser.parse_args(
        [
            "--input-file", "dataset/data/full_math.txt",
            "-w", "dataset/data/full_set",
            "-b", "1",
        ],
    )
    main(in_file=args.input_file, work_dir=args.work_dir, batch=args.batch_size)
