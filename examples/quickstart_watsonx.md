# Install provider extra
pip install -e .[watsonx]

# Select provider + set creds
export LLM_PROVIDER=watsonx
export WATSONX_API_KEY=YOUR_KEY
export WATSONX_URL=https://us-south.ml.cloud.ibm.com          # or your region URL
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID
export MODEL_ID=ibm/granite-3-3-8b-instruct                    # (optional) pick your model

# Pick any framework (native is fine)
export AGENT_FRAMEWORK=native

# Run server
uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000

# Verify both layers are ready
curl -s localhost:8000/readyz | jq
