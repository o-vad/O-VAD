export OPENAI_API_KEY="***REMOVED-CREDENTIAL***"

python /home/yizhou/Mprojects/VAD/TubeletGraph/TubeletGraph/vlm/prompt_vlm_compatible.py -c configs/default.yaml -p /home/yizhou/Mprojects/VAD/TubeletGraph/_pred_out/custom-0000-Ours \
    --detect_anomalies --sample_interval 10