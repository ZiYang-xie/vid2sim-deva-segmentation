seq_path=$1

mask_dir="$seq_path/masks"

#* Step 1: Generate heuristic dynamic masks
python mask_prompt.py --chunk_size 4 \
                      --img_path $seq_path/images \
                      --amp --temporal_setting semionline \
                      --size -1 \
                      --output $seq_path \
                      --prompt man.woman.human.pedestrian.cyclist # Input prompt here, separated by periods