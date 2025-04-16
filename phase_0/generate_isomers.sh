#!/bin/bash

QUERY_DIR=""
OUTPUT_DIR="smiles_isomers"
MAX_CENTERS=16
ENUM_NITROGEN=""
ENUM_BRIDGEHEAD=""
DIR="/mnt/c/Users/user/Desktop/molbital/phase_0"

usage() {
    echo "Usage: $0 -i <query_directory> [-o <output_directory>] [-m <max_centers>] [-n] [-b]"
    echo "  -i  Input directory containing SMI files (required)"
    echo "  -o  Output directory for stereoisomer SMI files (default: smiles_isomers)"
    echo "  -m  Maximum number of stereocenters to enumerate (default: 16)"
    echo "  -n  Enable nitrogen stereocenter enumeration"
    echo "  -b  Enable bridgehead stereocenter enumeration"
    exit 1
}

while getopts "i:o:m:nb" opt; do
    case $opt in
        i) QUERY_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) MAX_CENTERS="$OPTARG" ;;
        n) ENUM_NITROGEN="--enum_nitrogen" ;;
        b) ENUM_BRIDGEHEAD="--enum_bridgehead" ;;
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

SMI_FILES=$(find "$QUERY_DIR" -type f -name "*.smi")

if [ -z "$SMI_FILES" ]; then
    echo "Error: No .smi files found in '$QUERY_DIR'."
    exit 1
fi

COUNT=0

for INPUT_SMI in $SMI_FILES; do
    BASENAME=$(basename "$INPUT_SMI" .smi)
    OUTPUT_SMI="$OUTPUT_DIR/${BASENAME}_isomers.smi"

    echo "Processing: $INPUT_SMI -> $OUTPUT_SMI"

    python "$DIR/isomer_enumerator.py" \
        --input_smi "$INPUT_SMI" \
        --output_smi "$OUTPUT_SMI" \
        --max_centers "$MAX_CENTERS" \
        $ENUM_NITROGEN \
        $ENUM_BRIDGEHEAD

    if [ $? -eq 0 ]; then
        echo "Successfully generated isomers for $INPUT_SMI"
        ((COUNT++))
    else
        echo "Error: Failed to process $INPUT_SMI"
    fi
done

echo "Processed $COUNT SMI files."
echo "Output stereoisomers saved in '$OUTPUT_DIR'."
