# 환경변수 로드
source config.env
echo "MODEL_FLAGS: $MODEL_FLAGS"
echo "DIFFUSION_FLAGS: $DIFFUSION_FLAGS"

export LOGDIR
EMA_FILE=$(find "$LOGDIR" -type f -name "model*.pt" | tail -n 1)
echo "EMA_FILE: $EMA_FILE"

python scripts/image_sample.py --model_path $EMA_FILE $MODEL_FLAGS $SAMPLE_FLAGS

# python scripts/image_sample.py --model_path $EMA_FILE --use_ddim True $MODEL_FLAGS $DIFFUSION_FLAGS
# python scripts/image_sample.py --model_path $EMA_FILE $MODEL_FLAGS $DIFFUSION_FLAGS