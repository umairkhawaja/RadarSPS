import yaml
import subprocess
import tempfile
import argparse
import os
import copy

def find_best_weights(weights_dir):
    # List all files in the weights directory
    files = os.listdir(weights_dir)
    # Exclude 'last' and any hidden files
    weight_files = [f for f in files if f != 'last' and not f.startswith('.')]
    if not weight_files:
        raise FileNotFoundError(f"No weight file found in {weights_dir} excluding 'last'")
    # If there's more than one, you may need to adjust the selection criteria
    best_weight = os.path.join(weights_dir, weight_files[0])
    return best_weight

def main():
    parser = argparse.ArgumentParser(description='Automate testing over multiple experiments.')
    parser.add_argument('-e', '--experiments', nargs='+', required=True, help='List of experiment names.')
    parser.add_argument('-p', '--predict_script', default='python scripts/predict.py', help='Predict script command.')
    parser.add_argument('-l', '--logs_dir', default='tb_logs', help='Directory where experiment logs are stored.')
    args = parser.parse_args()

    experiments = args.experiments
    predict_command = args.predict_script
    logs_dir = args.logs_dir

    for experiment in experiments:
        print(f"\nProcessing experiment: {experiment}")

        # Define paths
        exp_dir = os.path.join(logs_dir, experiment, 'version_1')
        weights_dir = os.path.join(exp_dir, 'checkpoints')
        config_path = os.path.join(exp_dir, 'hparams.yaml')

        # Check if paths exist
        if not os.path.exists(exp_dir):
            print(f"Experiment directory {exp_dir} does not exist. Skipping.")
            continue
        if not os.path.exists(weights_dir):
            print(f"Weights directory {weights_dir} does not exist. Skipping.")
            continue
        if not os.path.exists(config_path):
            print(f"Config file {config_path} does not exist. Skipping.")
            continue

        # Find the best weights file
        try:
            weights_path = find_best_weights(weights_dir)
            print(f"Using weights file: {weights_path}")
        except FileNotFoundError as e:
            print(e)
            continue

        # Load the base config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set batch size to 1
        if 'TRAIN' in config and 'BATCH_SIZE' in config['TRAIN']:
            config['TRAIN']['BATCH_SIZE'] = 1
        else:
            print("Warning: 'BATCH_SIZE' not found in config['TRAIN']")

        # Get the list of test scenes from DATA.MAPS.TEST
        test_scenes_maps = config.get('DATA', {}).get('MAPS', {}).get('TEST', [])
        # Get the list of test scenes from DATA.SPLIT.TEST
        test_scenes_split = config.get('DATA', {}).get('SPLIT', {}).get('TEST', [])
        # Combine the lists and remove duplicates
        test_scenes = list(set(test_scenes_maps + test_scenes_split))

        if not test_scenes:
            print("No test scenes found in the config. Skipping this experiment.")
            continue

        for scene in test_scenes:
            # Create a copy of the config
            config_copy = copy.deepcopy(config)
            # Update the test scenes to only the current scene
            if 'DATA' in config_copy:
                if 'SPLIT' in config_copy['DATA'] and 'TEST' in config_copy['DATA']['SPLIT']:
                    config_copy['DATA']['SPLIT']['TEST'] = [scene]

            # Save the modified config to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_config_file:
                yaml.dump(config_copy, temp_config_file)
                temp_config_file_path = temp_config_file.name

            # Build the command
            command = f"{predict_command} -w {weights_path} -c {temp_config_file_path}"

            print(f"\nRunning prediction for scene: {scene}")
            print(f"Command: {command}")
            # Run the predict script
            subprocess.run(command, shell=True)

            # Remove the temporary config file
            os.remove(temp_config_file_path)

    print("\nAll experiments processed.")

if __name__ == "__main__":
    main()