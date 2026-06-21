import os
import yaml
import pytest
import json
from datetime import datetime, timedelta, timezone
from src.data_pipeline.data_loader import DataLoader

def test_fetch_macro_data_staleness(tmp_path):
    # Setup dummy config and snapshot
    config_path = tmp_path / "config.yaml"
    snapshot_path = tmp_path / "macro_snapshot.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump({
            "calibration": {"macro_snapshot_path": str(snapshot_path)},
            "run": {"max_staleness_days": 60}
        }, f)
        
    old_date = (datetime.now(timezone.utc).date() - timedelta(days=200)).strftime("%Y-%m-%d")
    with open(snapshot_path, "w") as f:
        yaml.dump({"repo_rate": {"value": 0.05, "as_of": old_date}}, f)
        
    loader = DataLoader(config_path=str(config_path))
    loader.cache_dir = str(tmp_path / "cache")
    os.makedirs(loader.cache_dir, exist_ok=True)
    
    with pytest.raises(ValueError, match="exceeding 180 days hard limit"):
        loader.fetch_macro_data(allow_stale=False)
        
    # Should pass with allow_stale
    data = loader.fetch_macro_data(allow_stale=True)
    assert data["repo_rate"]["staleness_days"] == 200

def test_fetch_tier1_fallback(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({
            "calibration": {"macro_snapshot_path": "dummy.yaml"},
            "run": {"max_staleness_days": 60}
        }, f)
        
    loader = DataLoader(config_path=str(config_path))
    loader.cache_dir = str(tmp_path / "cache")
    os.makedirs(loader.cache_dir, exist_ok=True)
    
    # No cache, should use dummy fallback
    data = loader.fetch_tier1_mca_aggregates()
    assert "state_aggregates" in data
    assert "MH" in data["state_aggregates"]

def test_fetch_tier1_with_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({
            "calibration": {"macro_snapshot_path": "dummy.yaml"},
            "run": {"max_staleness_days": 60}
        }, f)
        
    loader = DataLoader(config_path=str(config_path))
    loader.cache_dir = str(tmp_path / "cache")
    os.makedirs(loader.cache_dir, exist_ok=True)
    
    # Write a dummy cache file that is very recent
    recent_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    cache_file = os.path.join(loader.cache_dir, f"mca_cdm_{recent_date}.json")
    with open(cache_file, "w") as f:
        json.dump({"from_cache": True}, f)
        
    data = loader.fetch_tier1_mca_aggregates()
    assert data.get("from_cache") is True
