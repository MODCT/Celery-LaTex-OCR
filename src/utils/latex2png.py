# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import os
from pathlib import Path
import random

import re
import sys
import io
import glob
import tempfile
import shlex
import json
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
\usepackage{unicode-math}
\usepackage{amsmath}
\newenvironment{mymath}{$\displaystyle}{$}
\begin{document}
\begin{mymath}
%s
\end{mymath}
\end{document}
'''
    def __init__(
            self, math: List, dpi: int = 250,
            tempdir: Union[Path, str] = tempfile.gettempdir(),
            img_names: Union[List[str], None] = None
        ):
        self.math: List = math
        self.dpi = dpi
        self.tempdir = Path(tempdir)
        if not self.tempdir.exists():
            self.tempdir.mkdir(parents=True)
        self.img_names = img_names or [str(i) for i in range(len(self.math))]
        assert len(self.math) == len(self.img_names), "equation list must have exact same length as their new names"
        # used for calculate error formula index
        self.prefix_line = self.BASE.split("\n").index("%s")

    def write(self, idx: int):
        try:
            workdir = str(self.tempdir.absolute())
            document = self.BASE % (self.math[idx].replace("\n", ""), )
            texfile = str((self.tempdir / f"{self.img_names[idx]}.tex").absolute())
            with open(texfile, "w") as f:
                f.write(document)
            # print(texfile)
            # print(document)
            self.convert_file(texfile, workdir, one=True,)
        finally:
            pass
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False, one=False):
        """Convert one or more tex file to pdf, then to png"""
        infile = infile.replace('\\', '/')
        workdir = workdir.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = (
                "xelatex "
                "-interaction nonstopmode "
                # "-file-line-error "
                f"-output-directory {workdir} "
                f"{infile} "
            )
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            sout, serr = p.communicate()
            # extract error line from sout
            error_index, _ = extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
            # extract success rendered equation
            if error_index != []:
                # offset index start from 0, same as self.math
                error_index = [int(_)-self.prefix_line-1 for _ in error_index]
            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            result, _ = extract(
                text=sout, expression="Output written on %s \((\d+)? page" % pdffile)
            if not one:
                if int(result[0]) != len(self.math):
                    raise Exception('xelatex rendering error, generated %d formula\'s page, but the total number of formulas is %d.' % (
                        int(result[0]), len(self.math)))
            # pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))
            pngfile = os.path.join(workdir, infile.replace('.tex', ''))
            dpi = random.randint(110, 300)
            cmd = (
                # "convert "
                # "-alpha off "
                # f"-density {self.dpi} "
                # "-colorspace gray "
                # "-quality 90 "
                "pdftoppm "
                "-png "
                "-gray "
                "-singlefile "
                f"-r {dpi} "
                f"{pdffile} "
                f"{pngfile} "
            )# -bg Transparent -z 9
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(
                    pdffile), os.path.exists(infile))
            if return_bytes:
                if not one:
                    if len(self.math) > 1:
                        png = [open(pngfile.replace('.png', '')+'-%i.png' %
                                    i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace(
                        '.png', '')+'.png', 'rb').read()]
            else:
                # return path
                if not one:
                    if len(self.math) > 1:
                        png = [(pngfile.replace('.png', '')+'-%i.png' % i)
                               for i in range(len(self.math))]
                else:
                    png = [(pngfile.replace('.png', '')+'.png')]
            return png, error_index
        except Exception as e:
            print(e)
        finally:
            # Cleanup temporaries
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile+'*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}
def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, **kwargs):
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return (images, error_index) if return_error_index else images


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    # print(f"text: {text}\n expr: {expression}")
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text.replace("\n", ""))
        # print(results)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


def main(
        in_file: str = "", format: str = "", work_dir: str = "",
        eq_str: str = "", eq_file: str = "", brk_file: str = ""
    ):

    eq_names = None
    __supported_file_format__ = ("json", "txt")
    if not eq_str:
        # mode 3
        if brk_file:
            with open(brk_file, "r") as f:
                brk_images = json.load(f)
            with open(eq_file, "r") as f:
                eqs_lib = f.readlines()
            eqs, eq_names = generate_broken_images(eqs_lib, brk_images)
        # mode 2
        else:
            assert format in __supported_file_format__, f"file format {format} not supported"
            if format == "json":
                with open(in_file, "r") as fin:
                    eqs = json.load(fin)
            else:
                with open(in_file, "r") as fin:
                    eqs = fin.readlines()
            eq_names = [i for i,_ in enumerate(eqs)]

    # mode 1
    else:
        eqs = [eq_str, ]
    # eqs = [r"\begin{equation*}" + fr"{e}" + r"\end{equation*}" for e in eqs]
    # print(eqs, eq_names)
    if eq_names is not None:
        assert len(eq_names) == len(eqs)
    gen_latex = Latex(eqs, tempdir=work_dir, img_names=eq_names)

    pool = Pool(20)
    for idx in tqdm(range(len(eqs))):
        if (Path(work_dir) / f"{eq_names[idx]}.png").exists():
            # print(f"Skip {idx} ...")
            continue
        pool.apply_async(gen_latex.write, (idx, ))
    pool.close()
    pool.join()


def generate_broken_images(math_tex: List[str], broken_images: List[str]):
    eq_names = []
    eqs = []
    for img in broken_images:
        idx = int(Path(img).name.split(".")[0])
        tex = math_tex[idx]
        eqs.append(tex)
        eq_names.append(idx)
    # print(eqs, eq_names)
    return eqs, eq_names

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-file", dest="input_file", type=str, default=None, help="equations to render in a file")
    parser.add_argument("-e", "--equation", dest="equation", type=str, default=None, help="equation in str")
    parser.add_argument("-b", "--broken-file", dest="broken_file", type=str, default=None, help="broken images in a file")
    parser.add_argument("--equation-file", dest="equation_file", type=str, default=None, help="all equations in file")
    parser.add_argument("-f", "--file-format", dest="file_format", default="json")
    parser.add_argument("-w", "--work-dir", dest="work_dir", default=None)

    args = parser.parse_args(
        [
            # "-b", "broken_imgs.json",
            "-f", "txt",
            "--input-file", "dataset/data/full_math.txt",
            "-w", "/home/rainy/latexocrData/full_set_new1",
        ],
    )
    # mode 1, only --equation
    if args.equation:
        main(work_dir=args.work_dir, eq_str=args.equation)
    # mode 2, -i and -f
    elif args.input_file:
        main(in_file=args.input_file, format=args.file_format, work_dir=args.work_dir)
    # mode 3, -b
    elif args.broken_file and args.equation_file:
        main(brk_file=args.broken_file, eq_file=args.equation_file, work_dir=args.work_dir)
    else:
        raise NotImplementedError()
