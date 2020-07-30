from indicnlp.tokenize import indic_tokenize
import requests
import tarfile
import os
data_dir = 'iitb'


def download_and_extract(url, target_path):
  if not os.path.exists(target_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
      raise ConnectionError('Download failed')
    with open(target_path, 'wb') as f:
      f.write(response.raw.read())

  tar = tarfile.open(target_path, "r:gz")
  tar.extractall(data_dir)
  tar.close()


def rewrite_tokenized(src_path, out_filename):
  src_file = open(src_path, 'r', encoding='utf-8')
  src_lines = src_file.read().split('\n')
  tokenized_lines = list(map(lambda line: ' '.join(indic_tokenize.trivial_tokenize(line)), src_lines))
  if not os.path.exists(os.path.join(data_dir, 'tokenized')):
    os.mkdir(os.path.join(data_dir, 'tokenized'))
  out_file = open(os.path.join(data_dir, 'tokenized', out_filename), 'w', encoding='utf-8')
  out_file.write('\n'.join(tokenized_lines))


def extract_and_tokenize():
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  train_url = 'http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/parallel.tgz'
  train_path = os.path.join(data_dir, 'parallel.tgz')
  test_url = 'http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/dev_test.tgz'
  test_path = os.path.join(data_dir, 'dev_test.tgz')
  download_and_extract(train_url, train_path)
  download_and_extract(test_url, test_path)
  rewrite_tokenized(os.path.join(data_dir, 'parallel', 'IITB.en-hi.en'), 'train.en')
  rewrite_tokenized(os.path.join(data_dir, 'parallel', 'IITB.en-hi.hi'), 'train.hi')
  rewrite_tokenized(os.path.join(data_dir, 'dev_test', 'test.en'), 'test.en')
  rewrite_tokenized(os.path.join(data_dir, 'dev_test', 'test.hi'), 'test.hi')
  rewrite_tokenized(os.path.join(data_dir, 'dev_test', 'dev.en'), 'dev.en')
  rewrite_tokenized(os.path.join(data_dir, 'dev_test', 'dev.hi'), 'dev.hi')


if __name__ == "__main__":
  extract_and_tokenize()
"""
TEXT=iitb/tokenized
fairseq-preprocess --source-lang hi --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir data-bin/iitb \
    --workers 12
"""

"""
fairseq-train \
    data-bin/iitb \
    --max-tokens 4096 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam \
    --lr 1e-4 --min-lr 1e-9 \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 2 \
    --max-epoch 10
"""

"""
fairseq-generate data-bin/iitb \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
  """
