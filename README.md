# Domain LLM API

Minimal repo for this flow:

`domain data -> LoRA training -> adapter artifact -> S3 -> GitHub Actions -> GPU EC2 -> FastAPI endpoint`

This repo does not store your private production data or model weights.

For quick lab success, it includes a starter sample dataset at `data/train.jsonl`.

## Keep In Repo

- `train/train.py`: fine-tune a LoRA adapter on your domain data
- `app/`: FastAPI + vLLM inference
- `scripts/bootstrap.sh`: one-time EC2 setup
- `.github/workflows/ci.yml`: lint, test, docker build
- `.github/workflows/deploy.yml`: deploy to EC2 and pull adapter from S3

## Step 1: Prepare domain data

Keep your dataset outside the repo. Use a path like `data/train.jsonl` locally or mount it from secure storage.

Expected format:

`{"text": "<training sample>"}` per line in JSONL.

Starter file available now: `data/train.jsonl` (personal profile examples for your lab).

## Step 2: Train the domain adapter

Run on a Linux GPU machine:

```bash
pip install -r train/requirements.txt
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct DATA_PATH=data/train.jsonl OUTPUT_DIR=artifacts/adapter python train/train.py
```

`HF_TOKEN` is not required for public models. Only set it in your shell if you later use a gated/private model.

The output in `artifacts/adapter/` is your model artifact:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files

This is a LoRA adapter, not the full base model.

## Step 3: Upload artifact to S3

In AWS Console:

1. Open `S3`.
2. Create a private bucket.
3. Upload the adapter folder, or run:

```bash
aws s3 sync artifacts/adapter/ s3://my-llm-artifacts/adapters/domain/
```

## Step 4: Create GPU EC2

In AWS Console:

1. Open `EC2` -> `Launch instance`.
2. Pick `Deep Learning AMI GPU Ubuntu 22.04`.
3. Pick a GPU type like `g4dn.xlarge` or `g5.xlarge`.
4. Create a key pair.
5. Enable public IP.
6. Security group:
   - allow `22` from your IP
   - allow `8000` from your IP temporarily
7. Use at least `100 GB` storage.

## Step 5: Bootstrap the server

```bash
scp -i domain-llm.pem scripts/bootstrap.sh ubuntu@<ec2-public-dns>:~/
ssh -i domain-llm.pem ubuntu@<ec2-public-dns>
chmod +x ~/bootstrap.sh && ~/bootstrap.sh
exit
ssh -i domain-llm.pem ubuntu@<ec2-public-dns>
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Step 6: Add GitHub secrets

In GitHub:

`Repo -> Settings -> Secrets and variables -> Actions`

Add:

- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`
- `GH_DEPLOY_TOKEN` if repo is private
- `S3_ADAPTER_URI` like `s3://my-llm-artifacts/adapters/domain/`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

## Step 7: CI

On push or PR, `ci.yml` runs:

1. `ruff check`
2. `pytest`
3. `docker build`

This validates code and container build. CI does not run GPU inference.

## Step 8: Deploy

On push to `main` or manual run, `deploy.yml`:

1. connects to EC2 over SSH
2. pulls latest repo
3. downloads the adapter from S3 to `/adapters/domain`
4. runs `docker compose build`
5. runs `docker compose up -d`

## Step 9: Create ALB

In AWS Console:

1. Open `EC2` -> `Load Balancers` -> `Create Application Load Balancer`.
2. Choose `Internet-facing`.
3. Create a target group:
   - type `Instances`
   - protocol `HTTP`
   - port `8000`
   - health check path `/health`
4. Register your EC2 instance.
5. Create an ALB security group with `80` open.
6. Update EC2 security group:
   - remove public `8000`
   - allow `8000` only from the ALB security group

## Step 10: Call the endpoint

```bash
curl http://<alb-dns>/health
curl -X POST http://<alb-dns>/v1/infer -H "Content-Type: application/json" -d "{\"prompt\":\"<your input>\"}"
```

Docs:

```bash
http://<alb-dns>/docs
```

## Local checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
ruff check app tests
pytest -q
```

## Important env vars

- `MODEL_NAME`
- `LORA_PATH`
- `MAX_TOKENS`
- `TEMPERATURE`
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`

## Next real-world upgrades

- `ACM` + HTTPS on ALB
- `CloudWatch` logs
- `Secrets Manager`
- `ECR` image registry
- `EFS` for model cache
