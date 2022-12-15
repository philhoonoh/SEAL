#!/usr/bin/env python3

dataArray=("dev" "test")

for data in ${dataArray[@]}; do
  dataset="${data}.json"

  echo "TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr_qas --topics input.json \
    --output_format dpr --output output.json \
    --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
    --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ \
    --jobs 75 --progress --device cuda:1 --batch_size 20 \
    --beam 15
  "
  CUDA_VISIBLE_DEVICES=5 python passage_retrieval.py \
  --model_name_or_path facebook/contriever-msmarco \
  --passages /data/philhoon-relevance/FiD/open_domain_data/wikipedia_psgs/psgs_w100.tsv \
  --passages_embeddings "/data/philhoon-relevance/contriever/wikipedia_embeddings/contriever_msmacro/wikipedia_embeddings/*" \
  --data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
  --output_dir /data/philhoon-relevance/contriever/NQ/contriever-msmarco/ \
  --per_gpu_batch_size 256
done

#TOKENIZERS_PARALLELISM=false python -m seal.search \
#    --topics_format dpr --topics input.json \
#    --output_format dpr --output output.json \
#    --checkpoint checkpoint.pt \
#    --fm_index fm_index \
#    --jobs 75 --progress --device cuda:0 --batch_size 20 \
#    --beam 15