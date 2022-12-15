#!/usr/bin/env python3

dataArray=("dev" "test")

for data in ${dataArray[@]}; do
  dataset="${data}.jsonl"
  output="${data}.json"

  echo "TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr_out --topics /data/philhoon-relevance/contriever/NQ/contriever-msmarco/"$dataset" \
    --output_format dpr --output /data/philhoon-relevance/SEAL/NQ/"$output" \
    --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
    --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ \
    --jobs 75 --progress --device cuda:0 --batch_size 32 \
    --beam 15
  "
  TOKENIZERS_PARALLELISM=false python -m seal.search \
  --topics_format dpr_out --topics /data/philhoon-relevance/contriever/NQ/contriever-msmarco/"$dataset" \
  --output_format dpr --output /data/philhoon-relevance/SEAL/NQ/"$output" \
  --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
  --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ \
  --jobs 75 --progress --device cuda:0 --batch_size 32 \
  --beam 15
done

#TOKENIZERS_PARALLELISM=false python -m seal.search \
#    --topics_format dpr_out --topics /data/philhoon-relevance/contriever/NQ/contriever-msmarco/dev.jsonl \
#    --output_format dpr --output output.json \
#    --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
#    --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ \
#    --jobs 75 --progress --device cuda:0 --batch_size 20 \
#    --beam 15