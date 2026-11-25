#!/usr/bin/env bash
set -euo pipefail
projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$projectRoot"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
config="${CONFIG:-configs/smoke.yaml}"
trainArgs=()
while (( $# )); do
    case "$1" in
        --config)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                echo "--config requires a YAML path." >&2
                exit 2
            fi
            config="$2"
            shift 2
            ;;
        --config=*)
            config="${1#--config=}"
            if [[ -z "$config" ]]; then
                echo "--config requires a YAML path." >&2
                exit 2
            fi
            shift
            ;;
        *)
            trainArgs+=("$1")
            shift
            ;;
    esac
done
python prepareData.py --config "$config"
python fitStatistics.py --config "$config"
python train.py --config "$config" "${trainArgs[@]}"