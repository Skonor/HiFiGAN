from pathlib import Path

import gzip
import os, shutil, wget

URL_LINK = {
    'kenlm': {'link': 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz', 'name': '3-gram.pruned.1e-7'}
}

def main():
    dir = Path(__file__).absolute().resolve().parent.parent
    lm_dir = dir / 'data' / 'LMs'
    lm_dir.mkdir(exist_ok=True, parents=True)
    lm_gzip_path = lm_dir / (URL_LINK['kenlm']['name'] + '.arpa.gz')
    if not lm_gzip_path.exists():
        wget.download(URL_LINK['kenlm']['link'], out=str(lm_dir))
    lm_path = lm_dir / (URL_LINK['kenlm']['name'] + '.arpa')
    if not lm_path.exists():
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)

if __name__ == "__main__":
    main()
