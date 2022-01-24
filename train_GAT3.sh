id="GAT_enc_glu_beam5_3l_1_rl_1"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi

#python train.py --id $id \
#    --random_seed 1234 \
#    --caption_model gat \
#    --use_box 1 \
#    --use_glu_enc 1 \
#    --use_glu_dec 0 \
#    --N_model 3 \
#    --num_heads 8 \
#    --d_model 512 \
#    --label_smoothing 0.2 \
#    --input_json data/cocotalk.json \
#    --input_label_h5 data/cocotalk_label.h5 \
#    --input_fc_dir  data/cocobuu_fc \
#    --input_att_dir  data/cocobuu_att  \
#    --input_box_dir  data/cocobuu_box \
#    --seq_per_img 5 \
#    --batch_size 18 \
#    --beam_size 5 \
#    --learning_rate 5e-4 \
#    --num_layers 1 \
#    --input_encoding_size 512 \
#    --att_feat_size 2048 \
#    --rnn_size 512 \
#    --learning_rate_decay_start 0 \
#    --scheduled_sampling_start 0 \
#    --checkpoint_path log/log_$id  \
#    $start_from \
#    --save_checkpoint_every 6000 \
#    --language_eval 1 \
#    --val_images_use -1 \
#    --max_epochs 30 \
#    --scheduled_sampling_increase_every 5 \
#    --scheduled_sampling_max_prob 0.5 \
#    --learning_rate_decay_every 3 \

python train.py --id $id \
    --random_seed 12344 \
    --caption_model gat \
    --use_box 1 \
    --use_glu_enc 1 \
    --use_glu_dec 0 \
    --N_model 3 \
    --num_heads 8 \
    --d_model 512 \
    --input_json data/cocotalk.json \
    --input_label_h5 data/cocotalk_label.h5 \
    --input_fc_dir  data/cocobuu_fc \
    --input_att_dir  data/cocobuu_att  \
    --input_box_dir  data/cocobuu_box \
    --seq_per_img 5 \
    --batch_size 16 \
    --beam_size 5 \
    --num_layers 1 \
    --input_encoding_size 512 \
    --att_feat_size 2048 \
    --rnn_size 512 \
    --language_eval 1 \
    --val_images_use -1 \
    --save_checkpoint_every 3000 \
    # --start_from log/log_$id \
    --start_from "$start_from" \
    # --checkpoint_path log/log_$id \ 
    --checkpoint_path "$start_from"\
    --learning_rate 5e-5 \
    --max_epochs 60 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau
