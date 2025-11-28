#!/bin/bash
set -xe
# get the net device
# UCX_NET_DEVICES=$(ip a | grep '^2:' | awk '{print $2}' | cut -d: -f1)
# # check if UCX_NET_DEVICES is empty
# if [ -z "$UCX_NET_DEVICES" ]; then
#   echo "UCX_NET_DEVICES is empty. Please check your network configuration."
#   exit 1
# fi

# check if LMCACHE_CONTROLLER_ADDR_PULL is set
# if [ -z "$LMCACHE_CONTROLLER_ADDR_PULL" ]; then
#   echo "LMCACHE_CONTROLLER_ADDR_PULL is not set. Please set it to the lmcache controller address."
#   exit 1
# fi

# # check if LMCACHE_CONTROLLER_ADDR_REPLY is set
# if [ -z "$LMCACHE_CONTROLLER_ADDR_REPLY" ]; then
#   echo "LMCACHE_CONTROLLER_ADDR_REPLY is not set. Please set it to the lmcache controller address."
#   exit 1
# fi

DATACENTER_LABEL="$(cat /podinfo/datacenter_label)"

# check if DATACENTER_LABEL is set
if [ -z "$DATACENTER_LABEL" ]; then
  echo "DATACENTER_LABEL is not set. Please set it to the datacenter label."
  exit 1
fi

THIS_POD_NAME=$POD_IP
if [ -z "$THIS_POD_NAME" ]; then
  THIS_POD_NAME=$(hostname)
fi

NODE_INTERNAL_IP="$(cat /nodeinfo/internal_ip)"
if [ -z "$NODE_INTERNAL_IP" ]; then
  echo "NODE_INTERNAL_IP is not set. Please check your pod configuration."
  exit 1
else
  echo "$NODE_INTERNAL_IP llm-d-infer.qifand.com" >> /etc/hosts
  echo "Resolved llm-d-infer.qifand.com to $NODE_INTERNAL_IP"
  cat /etc/hosts
fi


# # prepare lmcache.yaml
# cat <<EOF > lmcache.yaml
# chunk_size: $LMCACHE_CHUNK_SIZE
# local_cpu: True
# max_local_cpu_size: $LMCACHE_MAX_LOCAL_CPU_SIZE
# enable_async_loading: $LMCACHE_ENABLE_ASYNC_LOADING

# enable_p2p: $LMCACHE_ENABLE_P2P
# p2p_host: "$THIS_POD_NAME"
# p2p_init_ports: 28200
# p2p_lookup_ports: 28201
# transfer_channel: "nixl"

# enable_controller: $LMCACHE_ENABLE_CONTROLLER
# lmcache_instance_id: "Datacenter-$DATACENTER_LABEL-Pod-$THIS_POD_NAME"
# controller_pull_url: "$LMCACHE_CONTROLLER_ADDR_PULL"
# controller_reply_url: "$LMCACHE_CONTROLLER_ADDR_REPLY"
# lmcache_worker_ports: 28500

# extra_config:
#   lookup_backoff_time: 0.001
# EOF


  # --quantization $QUANTIZATION \
  # # --kv-cache-dtype $KV_CACHE_DTYPE \
# UCX_NET_DEVICES=eth0 \
#   NCCL_OOB_NET_IFNAME=eth0 \
#   LMCACHE_LOG_LEVEL=DEBUG \
  # UCX_TLS=tcp \

echo always > /sys/kernel/mm/transparent_hugepage/enabled

ulimit -l unlimited
      # --enforce-eager \

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  PYTHONHASHSEED=$PYTHONHASHSEED \
      vllm serve $HF_MODEL \
      --max-model-len $MAX_MODEL_LEN \
      --max-num-seqs $MAX_NUM_SEQS \
      --host 0.0.0.0 \
      --port 8200 \
      --block-size $VLLM_BLOCK_SIZE \
      --prefix-caching-hash-algo sha256_cbor \
      --kv-transfer-config '{"kv_connector":"'"$KV_CONNECTOR"'", "kv_role":"kv_both"}' \
      --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://gaie-${GAIE_RELEASE_NAME_POSTFIX}-epp.${NAMESPACE}.svc.cluster.local:5557\",\"topic\":\"kv@${POD_IP}@$HF_MODEL\"}" \
      --disable-uvicorn-access-log \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION 
fi

if [ -n "$KV_CACHE_MEMORY" ]; then
  PYTHONHASHSEED=$PYTHONHASHSEED \
      vllm serve $HF_MODEL \
      --max-model-len $MAX_MODEL_LEN \
      --max-num-seqs $MAX_NUM_SEQS \
      --host 0.0.0.0 \
      --port 8200 \
      --block-size $VLLM_BLOCK_SIZE \
      --prefix-caching-hash-algo sha256_cbor \
      --kv-transfer-config '{"kv_connector":"'"$KV_CONNECTOR"'", "kv_role":"kv_both"}' \
      --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://gaie-${GAIE_RELEASE_NAME_POSTFIX}-epp.${NAMESPACE}.svc.cluster.local:5557\",\"topic\":\"kv@${POD_IP}@$HF_MODEL\"}" \
      --disable-uvicorn-access-log \
      --kv-cache-memory $KV_CACHE_MEMORY 
fi

HF_HOME=./ LMCACHE_MAX_LOCAL_CPU_SIZE=30 vllm serve Qwen/Qwen3-0.6B\
      --host 0.0.0.0 \
      --port 8200 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1",
        "kv_role":"kv_both"
        }' \
      --max-model-len 256 \
      --max-num-seqs 256 \
      --disable-uvicorn-access-log \
      --gpu-memory-utilization 0.85 

HF_HOME=./ \
VLLM_USE_FLASHINFER_SAMPLER=1 \
VLLM_ATTENTION_BACKEND=FLASHINFER \
  vllm serve Qwen/Qwen3-0.6B\
      --host 0.0.0.0 \
      --port 8200 \
      --max-model-len 256 \
      --max-num-seqs 1024 \
      --disable-uvicorn-access-log \
      --gpu-memory-utilization 0.82 

# echo "Either GPU_MEMORY_UTILIZATION or KV_CACHE_MEMORY must be set."
# exit -1
