# 환경변수 로드
source config.env

# 디버깅용 출력 (필요하면 사용)
echo "MODEL_FLAGS: $MODEL_FLAGS"
echo "DIFFUSION_FLAGS: $DIFFUSION_FLAGS"
echo "TRAIN_FLAGS: $TRAIN_FLAGS"
echo "DATA_DIR: $DATA_DIR"
export LOGDIR

# 명령 실행
python scripts/image_train.py $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS