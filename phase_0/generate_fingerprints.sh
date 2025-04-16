#!/bin/bash


# Default values
QUERY_DIR=""
OUTPUT_DIR="smiles_fingerprints"
RADIUS=2
N_BITS=1024
PROCESSES=4
DIR="/mnt/c/Users/user/Desktop/molbital/phase_0"
# Function to display usage
usage() {
    echo "Usage: $0 -i <query_directory> [-o <output_directory>] [-r <radius>] [-n <n_bits>] [-p <processes>]"
    echo "Options:"
    echo "  -i  Query directory containing SMI files (required)"
    echo "  -o  Output directory for fingerprint files (default: smiles_fingerprints)"
    echo "  -r  Radius for Morgan fingerprints (default: 2)"
    echo "  -n  Number of bits for fingerprint vectors (default: 1024)"
    echo "  -p  Number of parallel processes (default: 4)"
    echo "Description:"
    echo "  This script processes all '*.smi' files in the query directory,"
    echo "  generating sparse Morgan fingerprints (with chirality) using fingerprint_generator.py"
    echo "  and saving them as condensed text files in the specified output directory."
    exit 1
}

while getopts "i:o:r:n:p:" opt; do
    case $opt in
        i) QUERY_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        r) RADIUS="$OPTARG" ;;
        n) N_BITS="$OPTARG" ;;
        p) PROCESSES="$OPTARG" ;;
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
    echo "Error: No *.smi files found in '$QUERY_DIR'."
    exit 1
fi

COUNT=0

for INPUT_SMI in $SMI_FILES; do
    BASENAME=$(basename "$INPUT_SMI" .smi)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_fingerprints.txt"

    echo "Processing: $INPUT_SMI -> $OUTPUT_FILE"

    python "$DIR/fingerprinter.py" \
        --input_smi "$INPUT_SMI" \
        --output_file "$OUTPUT_FILE" \
        --radius "$RADIUS" \
        --n_bits "$N_BITS" \
        --processes "$PROCESSES" \
        --quiet

    if [ $? -eq 0 ]; then
        echo "Successfully generated fingerprints for $INPUT_SMI"
        ((COUNT++))
    else
        echo "Error: Failed to process $INPUT_SMI"
    fi
done

echo "Processed $COUNT TXT files."
echo "Output fingerprints saved in '$OUTPUT_DIR'."
