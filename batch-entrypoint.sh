#!/bin/bash
# Batch poster generation entrypoint
#
# 1. Ensures required models are installed in InvokeAI
# 2. Runs the batch poster generation script
# 3. Exits (compose will then stop InvokeAI via depends_on)

set -e

echo "=== Batch Poster Generation ==="

# Step 1: Init models
./batch-init-models.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Model initialization failed"
    exit 1
fi

# Step 2: Run the batch poster generator
echo ""
echo "=== Starting poster generation ==="
python batch_poster_generate.py \
    --media-api "${MEDIA_API_URL:-http://localhost:8000}" \
    --invokeai "${INVOKEAI_URL:-http://invokeai:9090}" \
    --api-key "${API_KEY}" \
    --limit "${BATCH_LIMIT:-100}" \
    ${VERBOSE:+--verbose}

exit_code=$?

echo ""
echo "=== Batch complete (exit code: ${exit_code}) ==="
exit $exit_code
