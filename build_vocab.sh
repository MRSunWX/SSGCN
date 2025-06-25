#!/bin/bash
# build vocab for different datasets

python ./build_vocab.py --data_dir dataset/Restaurants_corenlp --vocab_dir dataset/Restaurants_corenlp
python ./build_vocab.py --data_dir dataset/Laptops_corenlp --vocab_dir dataset/Laptops_corenlp
python ./build_vocab.py --data_dir dataset/Tweets_corenlp --vocab_dir dataset/Tweets_corenlp

python ./build_vocab.py --data_dir dataset/Restaurants_allennlp --vocab_dir dataset/Restaurants_allennlp
python ./build_vocab.py --data_dir dataset/Laptops_allennlp --vocab_dir dataset/Laptops_allennlp
python ./build_vocab.py --data_dir dataset/Tweets_allennlp --vocab_dir dataset/Tweets_allennlp

python ./build_vocab.py --data_dir dataset/Restaurants_stanza --vocab_dir dataset/Restaurants_stanza
python ./build_vocab.py --data_dir dataset/Laptops_stanza --vocab_dir dataset/Laptops_stanza
python ./build_vocab.py --data_dir dataset/Tweets_stanza --vocab_dir dataset/Tweets_stanza

python ./build_vocab.py --data_dir dataset/Restaurants_biaffine --vocab_dir dataset/Restaurants_biaffine
python ./build_vocab.py --data_dir dataset/Laptops_biaffine --vocab_dir dataset/Laptops_biaffine
python ./build_vocab.py --data_dir dataset/Tweets_biaffine --vocab_dir dataset/Tweets_biaffine