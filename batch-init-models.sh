#!/bin/bash
# InvokeAI Model Initialization Script
#
# Waits for InvokeAI to be ready, then ensures the three required
# FLUX.2 Klein models are installed before signaling readiness.

set -e

INVOKEAI_URL="${INVOKEAI_URL:-http://invokeai:9090}"
MAX_WAIT=300  # seconds to wait for InvokeAI to be ready
POLL_INTERVAL=5

echo "=== InvokeAI Model Init ==="
echo "  InvokeAI URL: ${INVOKEAI_URL}"

# Wait for InvokeAI API to be reachable
echo "  Waiting for InvokeAI to start..."
elapsed=0
until curl -sf "${INVOKEAI_URL}/api/v1/app/version" > /dev/null 2>&1; do
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "  [ERROR] InvokeAI did not start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
    echo "  ... waiting (${elapsed}s)"
done
echo "  InvokeAI is ready"

# Check which models are already installed
installed_models=$(curl -sf "${INVOKEAI_URL}/api/v2/models/" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', data) if isinstance(data, dict) else data
for m in models:
    print(m.get('name', ''))
" 2>/dev/null || echo "")

install_model() {
    local name="$1"
    local source="$2"
    local body="$3"

    if echo "$installed_models" | grep -qF "$name"; then
        echo "  [OK] '$name' already installed"
        return 0
    fi

    echo "  Installing '$name'..."
    encoded_source=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$source', safe=''))")
    
    response=$(curl -sf -X POST \
        "${INVOKEAI_URL}/api/v2/models/install?source=${encoded_source}&inplace=true" \
        -H "Content-Type: application/json" \
        -d "$body")
    
    if [ $? -ne 0 ]; then
        echo "  [ERROR] Failed to start install for '$name'"
        return 1
    fi

    echo "  Install started for '$name'"
}

echo ""
echo "  Checking required models..."

# Model 1: FLUX.2 Klein 4B (GGUF Q4)
install_model \
    "FLUX.2 Klein 4B (GGUF Q4)" \
    "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf" \
    '{"name":"FLUX.2 Klein 4B (GGUF Q4)","base":"flux2","type":"main","description":"FLUX.2 Klein 4B GGUF Q4_K_M quantized - runs on 6-8GB VRAM. Installs with VAE and Qwen3 4B encoder. ~2.6GB","format":"gguf_quantized"}'

# Model 2: FLUX.2 VAE
install_model \
    "FLUX.2 VAE" \
    "black-forest-labs/FLUX.2-klein-4B::vae" \
    '{"name":"FLUX.2 VAE","base":"flux2","type":"vae","description":"FLUX.2 VAE (16-channel, same architecture as FLUX.1 VAE). ~335MB","format":null}'

# Model 3: FLUX.2 Klein Qwen3 4B Encoder
install_model \
    "FLUX.2 Klein Qwen3 4B Encoder" \
    "black-forest-labs/FLUX.2-klein-4B::text_encoder+tokenizer" \
    '{"name":"FLUX.2 Klein Qwen3 4B Encoder","base":"any","type":"qwen3_encoder","description":"Qwen3 4B text encoder for FLUX.2 Klein 4B (also compatible with Z-Image). ~8GB","format":null}'

# Wait for all installs to complete
echo ""
echo "  Waiting for model installs to complete..."
max_install_wait=600
install_elapsed=0

while true; do
    if [ $install_elapsed -ge $max_install_wait ]; then
        echo "  [ERROR] Model installs did not complete within ${max_install_wait}s"
        exit 1
    fi

    # Check install jobs status
    pending=$(curl -sf "${INVOKEAI_URL}/api/v2/models/install" | python3 -c "
import sys, json
data = json.load(sys.stdin)
jobs = data if isinstance(data, list) else data.get('jobs', data.get('items', []))
active = [j for j in jobs if j.get('status') in ('waiting', 'downloading', 'running')]
errored = [j for j in jobs if j.get('status') == 'error']
if errored:
    for e in errored:
        print(f\"ERROR: {e.get('source', 'unknown')}: {e.get('error', 'unknown error')}\", file=sys.stderr)
    sys.exit(2)
print(len(active))
" 2>/dev/null)

    if [ "$pending" = "0" ]; then
        echo "  All models installed successfully"
        break
    fi

    sleep $POLL_INTERVAL
    install_elapsed=$((install_elapsed + POLL_INTERVAL))
    echo "  ... waiting for ${pending} install(s) (${install_elapsed}s)"
done

echo ""
echo "=== Model init complete ==="
