import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class ModelValidator:
    """
    ModelValidator executes validation pipelines for macroeconomic models.
    Computes key performance indicators, handles dynamic weighting, and outputs validation reports.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.model = None
        self.metrics_history = {}
        self.load_model()
        
    def load_model(self):
        """Loads a model from a pickle file if specified in the configuration."""
        model_config = self.config.get("model", {})
        model_path = model_config.get("model_path")
        if model_path:
            # Resolve relative path if necessary
            if not os.path.isabs(model_path):
                config_dir = os.path.dirname(self.config_path)
                model_path = os.path.join(config_dir, model_path)
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

    def predict(self, df):
        """Generates predictions using either a loaded model or configuration weights."""
        model_config = self.config.get("model", {})
        features = model_config.get("features")
        
        # If a trained model is loaded, use it
        if self.model is not None:
            if features is None and hasattr(self.model, "feature_names_in_"):
                features = list(self.model.feature_names_in_)
            
            if features is not None:
                X = df[features]
            else:
                X = df
                
            try:
                # If the model has a predict method, use it
                if hasattr(self.model, "predict"):
                    return self.model.predict(X)
            except Exception:
                # Fall back to manual evaluation if the model object does not support predict directly
                if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
                    coef = self.model.coef_
                    intercept = self.model.intercept_
                    if features is not None:
                        X_vals = df[features].values
                    else:
                        X_vals = df.select_dtypes(include=[np.number]).values
                    return np.dot(X_vals, coef) + intercept
                raise
                
        # Fallback to weights and bias from config
        weights = model_config.get("weights")
        bias = model_config.get("bias", 0.0)
        
        if weights is not None:
            weights = np.array(weights)
            if features is not None:
                X = df[features].values
            else:
                # Exclude non-numeric columns and target columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                exclude = ["target", "label", "y", "prediction"]
                features = [c for c in numeric_cols if c not in exclude]
                X = df[features].values
                
            if X.shape[1] != len(weights):
                if X.shape[1] > len(weights):
                    X = X[:, :len(weights)]
                else:
                    raise ValueError(f"Feature dimension {X.shape[1]} is less than weights length {len(weights)}")
            return np.dot(X, weights) + bias
            
        raise ValueError("No valid model file loaded and no weights/bias provided in configuration.")

    def compute_metrics(self, val_df):
        """Computes statistical metrics comparing target and prediction columns."""
        if "target" not in val_df.columns or "prediction" not in val_df.columns:
            raise ValueError("DataFrame must contain 'target' and 'prediction' columns.")
            
        y_true = val_df["target"].values
        y_pred = val_df["prediction"].values
        
        metrics_config = self.config.get("validation", {}).get("metrics", [])
        results = {}
        
        for metric_info in metrics_config:
            name = metric_info.get("name")
            if not name:
                continue
                
            val = None
            if name == "MSE":
                val = float(np.mean((y_true - y_pred) ** 2))
            elif name == "MAE":
                val = float(np.mean(np.abs(y_true - y_pred)))
            elif name == "MAPE":
                denom = np.abs(y_true)
                denom = np.where(denom == 0, 1e-8, denom)
                val = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
            elif name == "R2":
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                val = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            elif name == "Pearson":
                if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
                    corr, _ = pearsonr(y_true, y_pred)
                    val = float(corr)
                    if np.isnan(val):
                        val = 0.0
                else:
                    val = 0.0
            elif name == "quantile_loss":
                q = metric_info.get("quantile", 0.5)
                diff = y_true - y_pred
                val = float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))
            else:
                val = float(np.mean(np.abs(y_true - y_pred)))
                
            results[name] = val
            
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(val)
            
            history_size = self.config.get("dynamic_weighting", {}).get("history_size", 10)
            if len(self.metrics_history[name]) > history_size:
                self.metrics_history[name].pop(0)
                
        return results

    def update_dynamic_weights(self):
        """Updates metric weights based on the trend in validation loss history."""
        dw_config = self.config.get("dynamic_weighting", {})
        if not dw_config.get("enabled", False):
            return
            
        metric_names = dw_config.get("metric_names", [])
        params = dw_config.get("parameters", {})
        lr = params.get("learning_rate", 0.1)
        threshold = params.get("threshold_percent", 0.15)
        
        metrics_list = self.config["validation"]["metrics"]
        metric_map = {m["name"]: m for m in metrics_list}
        updated_any = False
        
        for name in metric_names:
            if name not in self.metrics_history or len(self.metrics_history[name]) < 2:
                continue
                
            history = self.metrics_history[name]
            prev_val = history[-2]
            curr_val = history[-1]
            
            if prev_val == 0:
                continue
                
            relative_change = (curr_val - prev_val) / prev_val
            
            if abs(relative_change) > threshold:
                multiplier = 1.0 + lr * relative_change
                multiplier = max(0.5, min(2.0, multiplier))
                
                if name in metric_map:
                    metric_map[name]["weight"] = metric_map[name].get("weight", 1.0) * multiplier
                    updated_any = True
                    
        if updated_any:
            total_weight = sum(m.get("weight", 0.0) for m in metrics_list)
            if total_weight > 0:
                for m in metrics_list:
                    m["weight"] = float(m.get("weight", 0.0) / total_weight)

    def run_validation(self, data_path):
        """Runs the complete validation pipeline on a test dataset and outputs a JSON report."""
        df = pd.read_csv(data_path)
        
        if "prediction" not in df.columns:
            df["prediction"] = self.predict(df)
            
        metrics = self.compute_metrics(df)
        
        dw_config = self.config.get("dynamic_weighting", {})
        if dw_config.get("enabled", False):
            self.update_dynamic_weights()
            
        agg_config = self.config.get("validation", {}).get("aggregation", {})
        agg_method = agg_config.get("method", "weighted_sum")
        
        aggregate_score = 0.0
        metrics_list = self.config["validation"]["metrics"]
        
        if agg_method == "weighted_sum":
            for m in metrics_list:
                name = m["name"]
                weight = m.get("weight", 0.0)
                val = metrics.get(name, 0.0)
                aggregate_score += val * weight
        else:
            aggregate_score = float(np.mean(list(metrics.values())))
            
        passed = True
        failed_metrics = []
        for m in metrics_list:
            name = m["name"]
            threshold = m.get("threshold")
            val = metrics.get(name, 0.0)
            if threshold is not None:
                if name in ["R2", "Pearson"]:
                    if val < threshold:
                        passed = False
                        failed_metrics.append(name)
                else:
                    if val > threshold:
                        passed = False
                        failed_metrics.append(name)
                        
        status = "PASSED" if passed else "FAILED"
        
        results = {
            "metrics": metrics,
            "aggregate_score": aggregate_score,
            "status": status,
            "failed_metrics": failed_metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
            "configuration": self.config
        }
        
        output_path = self.config.get("validation", {}).get("output_path")
        if output_path:
            if not os.path.isabs(output_path):
                config_dir = os.path.dirname(self.config_path)
                output_path = os.path.join(config_dir, output_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
                
        return results
