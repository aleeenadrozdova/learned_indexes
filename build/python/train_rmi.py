import numpy as np
import json
import argparse
from sklearn.linear_model import LinearRegression

class RMITrainer:
    def __init__(self, branch_factor=100):
        """
        Initialize RMI trainer.
        """
        self.branch_factor = branch_factor
        self.stage1_model = None
        self.stage2_models = []
        self.errors = []
        
    def train(self, keys, positions):
        """
        Train RMI model.
        """
        n = len(keys)
        keys = np.array(keys).reshape(-1, 1)
        positions = np.array(positions)
        
        # Train first level model (root)
        self.stage1_model = LinearRegression()
        self.stage1_model.fit(keys, positions / n * self.branch_factor)
        
        # Distribute data between second level models
        stage1_predictions = np.clip(
            self.stage1_model.predict(keys), 
            0, 
            self.branch_factor - 1
        ).astype(int)
        
        # Initialize second level models
        self.stage2_models = [LinearRegression() for _ in range(self.branch_factor)]
        self.errors = [{"min_error": 0, "max_error": 0} for _ in range(self.branch_factor)]
        
        # Train each second level model
        for i in range(self.branch_factor):
            # Get keys belonging to this model
            mask = (stage1_predictions == i)
            if np.sum(mask) == 0:
                # No keys for this model, use default model
                self.stage2_models[i].coef_ = np.array([0])
                self.stage2_models[i].intercept_ = 0
                continue
                
            # Train model on its subset of keys
            model_keys = keys[mask]
            model_positions = positions[mask]
            
            if len(model_keys) == 1:
                # Only one key, use constant model
                self.stage2_models[i].coef_ = np.array([0])
                self.stage2_models[i].intercept_ = model_positions[0]
            else:
                # Multiple keys, use linear regression
                self.stage2_models[i].fit(model_keys, model_positions)
            
            # Calculate prediction errors
            preds = self.stage2_models[i].predict(model_keys)
            errors = model_positions - preds
            self.errors[i]["min_error"] = int(np.floor(np.min(errors)))
            self.errors[i]["max_error"] = int(np.ceil(np.max(errors)))
    
    def save_model(self, filename):
        """
        Save trained RMI model to file for use in C++.
        """
        model_params = {
            "branch_factor": self.branch_factor,
            "stage1": {
                "slope": float(self.stage1_model.coef_[0]),
                "intercept": float(self.stage1_model.intercept_)
            },
            "stage2": []
        }
        
        for i, model in enumerate(self.stage2_models):
            model_params["stage2"].append({
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "min_error": self.errors[i]["min_error"],
                "max_error": self.errors[i]["max_error"]
            })
        
        with open(filename, 'w') as f:
            json.dump(model_params, f, indent=2)

def load_data(data_file):
    """Load data from file"""
    keys = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                keys.append(int(line.strip()))
            except ValueError:
                # In case the value is not an integer, skip
                pass
    return np.array(keys)

def main():
    parser = argparse.ArgumentParser(description='Train RMI model')
    parser.add_argument('data_file', help='Path to the data file')
    parser.add_argument('model_file', help='Path to save the model')
    parser.add_argument('--num_models', type=int, default=100, 
                        help='Number of second-stage models')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_file}")
    keys = load_data(args.data_file)
    keys.sort()  # Sort keys if they are not already sorted
    positions = np.arange(len(keys))
    
    print(f"Training RMI model with {len(keys)} keys and {args.num_models} second-stage models")
    trainer = RMITrainer(branch_factor=args.num_models)
    trainer.train(keys, positions)
    
    print(f"Saving model to {args.model_file}")
    trainer.save_model(args.model_file)
    print("Training completed successfully")

if __name__ == "__main__":
    main()
