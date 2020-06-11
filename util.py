import os
import ray
import time
import re
import PyPDF2 as ppdf

path_bn = os.path.basename


def path_bns(paths):
    return list(map(path_bn, paths))


def abs_file_paths(directory):
    ps = list()
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            ps.append(os.path.abspath(os.path.join(dirpath, f)))
    return ps


@ray.remote
def pdf_to_text(pdf: str) -> (list, str):
    """ Given a pdf file in absolute path, returns its text indexed
    by the page number and name.
    """
    start = time.time()
    with open(pdf, "rb") as f:
        name = os.path.basename(pdf)
        p = ppdf.PdfFileReader(f)
        print(f"{name}: num. pages: {p.numPages}")

        text = list()
        for i in range(p.numPages):
            page = p.getPage(i)
            t = page.extractText()
            text.append(t)
            if i % 30 == 0:
                print(f"{name}: parsed {i + 1}/{p.numPages}")
        print(f"{name}: done, took {time.time() - start}s")
        return text, name


def parse_pdfs(pdfs) -> dict:
    start = time.time()
    texts = ray.get([pdf_to_text.remote(p) for p in pdfs])
    print(f"parsed {len(texts)} pdfs, took {time.time() - start}s")
    return {n: t for t, n in texts}


def patch_nltk():
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")


class EtsiParser:
    @staticmethod
    def get_abbrs_from_file(pdf: str) -> dict:
        p = pdf
        abbr_file = p.replace(".pdf", "-abbr.txt")
        if not os.path.exists(abbr_file):
            with open(abbr_file, 'w'): pass

        with open(abbr_file, "r") as f:
            if os.stat(abbr_file).st_size == 0:
                raise Exception(f"fill in abbreviation table in {path_bn(abbr_file)} to proceed:"
                                f"1. copy (a) content table or (b) abbreviation table from pdf;"
                                f"2. add # to each line if it starts with an acronym.")

            abbrs = dict()
            for l in f.readlines():
                l = l.rstrip()
                if l.startswith("#"):
                    l = l.lstrip("#").split()
                    abbrs[l[0]] = " ".join(l[1:])
                else:
                    if " (" in l:
                        r = re.compile("\((.*?)\)").search(l)
                        abbr = r.group(1)
                        abbrs[abbr] = l.replace(r.group(0), "")
                    else:
                        abbrs[l] = l
                        abbrs[l.lower()] = l.lower()
            return abbrs

    @staticmethod
    def get_title_from_pages(pages: list):
        return pages[0].split("\n")[-5].rstrip()
