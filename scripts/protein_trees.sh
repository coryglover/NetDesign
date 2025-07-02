#!/bin/bash
# This script is used to run protein tree simulations on local computer

# Loop through all graph files in /Users/glover.co/Documents/laszlo/NetDesign/data/protein_complex/proteins/human/edgefiles
for graph_file in /Users/glover.co/Documents/laszlo/NetDesign/data/protein_complex/proteins/human/edgefiles/*.edge; do
    # Extract the base name of the graph file (without path and extension)
    name=${graph_file##*/}
    base=${name%.edge}
    graph_file_name=$graph_file
    X_file_name="/Users/glover.co/Documents/laszlo/NetDesign/data/protein_complex/proteins/human/Xfiles/X_$base.txt"
    echo "$X_file_name"
    echo "$graph_file_name"
    # Run the protein tree simulation script with the graph file as an argument
    python /Users/glover.co/Documents/laszlo/NetDesign/scripts/run_mcmc.py --graph_file "$graph_file_name" --X_file "$X_file_name" --output /Users/glover.co/Documents/laszlo/NetDesign/data/protein_complex/proteins/human/trees/
done