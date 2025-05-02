#!/bin/bash

# Script to run both LSTM-RoBERTa and HAN-RoBERTa main scripts sequentially.
# Pass any arguments you want to provide to the underlying scripts directly
# when calling this script.
# Example: ./run_all_models.sh --mode train --final_epochs 5 --sample_size 20000

# Store all arguments passed to this script
ARGS="$@"

echo "---------------------------------------------"
echo "Running LSTM-RoBERTa Script..."
echo "Arguments: python src/scripts/lstm_roberta_main.py $ARGS"
echo "---------------------------------------------"

# Execute the LSTM script, passing all arguments
python src/scripts/lstm_roberta_main.py $ARGS

# Check if the first script was successful
# $? holds the exit status of the last command
if [ $? -ne 0 ]; then
  echo "---------------------------------------------"
  echo "ERROR: LSTM-RoBERTa script failed. Stopping."
  echo "---------------------------------------------"
  exit 1 # Exit the script with an error code
fi

echo ""
echo "---------------------------------------------"
echo "LSTM-RoBERTa script finished successfully."
echo "---------------------------------------------"
echo ""
echo "---------------------------------------------"
echo "Running HAN-RoBERTa Script..."
echo "Arguments: python src/scripts/han_roberta_main.py $ARGS"
echo "---------------------------------------------"



# Execute the HAN script, passing all arguments
python src/scripts/han_roberta_main.py $ARGS

# Check if the second script was successful
if [ $? -ne 0 ]; then
  echo "---------------------------------------------"
  echo "ERROR: HAN-RoBERTa script failed."
  echo "---------------------------------------------"
  exit 1 # Exit the script with an error code
fi

echo "---------------------------------------------"
echo "Running GNN-RoBERTa Script..."
echo "Arguments: python src/scripts/gnn_roberta_main.py $ARGS"
echo "---------------------------------------------"

# Execute the GNN script, passing all arguments
python src/scripts/gnn_roberta_main.py $ARGS

# Check if the second script was successful
if [ $? -ne 0 ]; then
  echo "---------------------------------------------"
  echo "ERROR: GNN-RoBERTa script failed."
  echo "---------------------------------------------"
  exit 1 # Exit the script with an error code
fi

echo ""
echo "---------------------------------------------"
echo "GNN-RoBERTa script finished successfully."
echo "All scripts completed."
echo "---------------------------------------------"

exit 0 # Exit successfully
