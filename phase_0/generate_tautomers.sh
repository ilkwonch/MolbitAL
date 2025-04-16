#!/bin/bash

QUERY_DIR=""
OUTPUT_DIR="smiles_all"
MAX_TAUTOMERS=16
PKA_NORM=""
MAX_SEARCH_TIME=30.0
DIR="/mnt/c/Users/user/Desktop/molbital/phase_0"

usage() {
    echo "Usage: $0 -i <query_directory> [-o <output_directory>] [-m <max_tautomers>] [-p] [-t <max_search_time>]"
    echo "Options:"
    echo "  -i  Query directory containing isomer SMI files (required)"
    echo "  -o  Output directory for tautomer SMI files (default: smiles_tautomers)"
    echo "  -m  Maximum number of tautomers/protomers to enumerate (default: 16)"
    echo "  -p  Enable pKa normalization to favor reasonable tautomers/protomers"
    echo "  -t  Maximum search time per molecule in seconds (default: 30.0)"
    echo "Description:"
    echo "  This script processes all '*_isomers.smi' files in the query directory,"
    echo "  generating tautomers/protomers using tautomer_generator.py and saving them"
    echo "  to the specified output directory."
    exit 1
}

while getopts "i:o:m:pt:" opt; do
    case $opt in
        i) QUERY_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) MAX_TAUTOMERS="$OPTARG" ;;
        p) PKA_NORM="--pka_norm" ;;
        t) MAX_SEARCH_TIME="$OPTARG" ;;
        ?) usage ;;
    esac
done

if [ -z "$QUERY_DIR" ]; then
    echo "Error: Query directory (-i) is required."
    usage
fi

if [ ! -d "$QUERY_DIR" ]; then
    echo "Error: Directory '$QUERY_DIR' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Could not create output directory '$OUTPUT_DIR'."
    exit 1
fi

SMI_FILES=$(find "$QUERY_DIR" -type f -name "*_isomers.smi")

if [ -z "$SMI_FILES" ]; then
    echo "Error: No *_isomers.smi files found in '$QUERY_DIR'."
    exit 1
fi

COUNT=0

for INPUT_SMI in $SMI_FILES; do
    BASENAME=$(basename "$INPUT_SMI" _isomers.smi)
    OUTPUT_SMI="$OUTPUT_DIR/${BASENAME}_tautomers.smi"

    echo "Processing: $INPUT_SMI -> $OUTPUT_SMI"

    python "$DIR/tautomer_enumerator.py" \
        --input_smi "$INPUT_SMI" \
        --output_smi "$OUTPUT_SMI" \
        --max_tautomers "$MAX_TAUTOMERS" \
        $PKA_NORM \
        --max_search_time "$MAX_SEARCH_TIME" \
        --quiet

    if [ $? -eq 0 ]; then
        echo "Successfully generated tautomers for $INPUT_SMI"
        ((COUNT++))
    else
        echo "Error: Failed to process $INPUT_SMI"
    fi
done

echo "Processed $COUNT SMI files."
echo "Output tautomers saved in '$OUTPUT_DIR'."
