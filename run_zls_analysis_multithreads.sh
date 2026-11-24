#!/bin/bash

# ============================================================================
# Run Koopman_Classic_analysis_zls.jl on all 32 cores, inside a tmux session
# ----------------------------------------------------------------------------

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_DIR="/home/sofia/Documents/julia_projects/KoopmanAnalysis"
SCRIPT_PATH="examples/Koopman_fitting_methods.jl"
REGIME_SYSTEM="FHN"
EXPERIMENT_TYPE="HiddenDim-LargeExp-multitraj-delays"

# Regimes to process SEQUENTIALLY (each gets all 32 cores).
# Uncomment / reorder as needed.
REGIMES=(
    #"c4b-SN-FLC"
    # "c2s-SN-SH"
    # "c3s-SN-supH"
    # "c2b-SN-SH"
    # "c14b-subH-SH"
    # "c16b-subH-FLC"
    "stable-limit-cycle"
)

# Thread / core configuration
JULIA_THREADS=32          # one Julia thread per core
CORES="0-31"              # pin to all 32 cores

# ---------------------------------------------------------------------------
# CRITICAL: BLAS / MKL / OpenMP must be single-threaded.
# find_zls_gradient_descent parallelises via Threads.@spawn; if BLAS also
# tried to use 32 threads we would have 32 * 32 = 1024 effective threads
# fighting for 32 cores (severe oversubscription, often slower than serial).
# The script also calls BLAS.set_num_threads(1) explicitly, but exporting
# these env vars ensures the value is correct from the moment Julia loads.
# ---------------------------------------------------------------------------
export JULIA_NUM_THREADS=${JULIA_THREADS}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SAVE_DIR_PATTERN="${PROJECT_DIR}/logs/${EXPERIMENT_TYPE}"
JULIA_BIN=$(which julia 2>/dev/null || echo "julia")

SESSION_NAME="koopman_fitting_analysis"

# ============================================================================
# BUILD THE INNER SHELL COMMAND THAT RUNS INSIDE tmux
# ============================================================================
# For each regime, we:
#   1. cd into PROJECT_DIR
#   2. Make sure SAVE_DIR exists
#   3. Run Julia with the regime as $1, piping both stdout and stderr
#      through tee into the log file.
#   4. Append a completion marker so you can grep the log for done-status.
#
# `stdbuf -oL -eL` makes stdout/stderr line-buffered so tee streams in
# real-time (otherwise you'd see no output until the buffer flushes).
# ============================================================================

INNER_CMD=""
for i in "${!REGIMES[@]}"; do
    REGIME="${REGIMES[$i]}"

    SAVE_DIR="${SAVE_DIR_PATTERN}/${REGIME_SYSTEM}-${REGIME}"
    LOG_FILE="${SAVE_DIR}/zls_analysis_${REGIME}_full.txt"

    # Build the per-regime block. Use printf to avoid quoting headaches.
    BLOCK=$(cat <<EOF
echo "============================================================"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting ZLS analysis for regime: ${REGIME}"
echo "    Julia threads : ${JULIA_THREADS}"
echo "    BLAS threads  : 1"
echo "    CPU cores     : ${CORES}"
echo "    Log file      : ${LOG_FILE}"
echo "============================================================"
mkdir -p '${SAVE_DIR}'
cd '${PROJECT_DIR}' && \\
stdbuf -oL -eL taskset -c ${CORES} '${JULIA_BIN}' -t ${JULIA_THREADS} --project='${PROJECT_DIR}' '${SCRIPT_PATH}' '${REGIME}' 2>&1 | tee '${LOG_FILE}'
echo "--- ZLS ANALYSIS COMPLETE FOR ${REGIME} ---" | tee -a '${LOG_FILE}'
echo ""
EOF
)
    # Chain blocks with a newline separator
    if [ -z "$INNER_CMD" ]; then
        INNER_CMD="$BLOCK"
    else
        INNER_CMD="${INNER_CMD}
${BLOCK}"
    fi
done

# Add a final summary marker at the end
INNER_CMD="${INNER_CMD}
echo '========================================================='
echo '[\$(date \'+%Y-%m-%d %H:%M:%S\')] ALL ZLS ANALYSES COMPLETE'
echo '========================================================='
"

# ============================================================================
# LAUNCH TMUX SESSION
# ============================================================================
mkdir -p "${SAVE_DIR_PATTERN}"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

echo "========================================================================="
echo "Launching tmux session: $SESSION_NAME"
echo "  Project       : $PROJECT_DIR"
echo "  Script        : $SCRIPT_PATH"
echo "  Julia bin     : $JULIA_BIN"
echo "  Julia threads : $JULIA_THREADS  (BLAS threads = 1)"
echo "  CPU cores     : $CORES"
echo "  Regimes       : ${REGIMES[*]}"
echo "  Save dir      : $SAVE_DIR_PATTERN"
echo "========================================================================="
echo ""

# Write the inner command to a temp script so quoting is bullet-proof.
TMP_RUNNER=$(mktemp /tmp/koopman_zls_run.XXXXXX.sh)
printf '%s\n' "$INNER_CMD" > "$TMP_RUNNER"
chmod +x "$TMP_RUNNER"

tmux new-session -d -s "$SESSION_NAME" "bash '${TMP_RUNNER}'"

sleep 1

echo "✅ Session '$SESSION_NAME' launched."
echo ""
echo "Monitor:"
echo "  tmux attach -t ${SESSION_NAME}        # attach (Ctrl-b d to detach)"
echo "  tail -f ${SAVE_DIR_PATTERN}/*/*.txt   # live tail of all logs"
echo "  htop                                   # CPU usage (should hit ~32 cores)"
echo "  ps -eLf | grep julia | grep -v grep   # check Julia threads"
echo ""
echo "Kill session:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "Check progress of any regime's log:"
echo "  grep -E 'ZLS ANALYSIS COMPLETE|grad-desc' ${SAVE_DIR_PATTERN}/*/*.txt"
echo "========================================================================="