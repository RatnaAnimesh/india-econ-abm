import os
import yaml
import json
import logging
from datetime import datetime, timezone
import requests

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.macro_snapshot_path = self.config["calibration"]["macro_snapshot_path"]
        self.max_staleness_days = self.config["run"].get("max_staleness_days", 60)
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_macro_data(self, allow_stale=False):
        """Loads macro snapshot and attempts live refresh if stale."""
        if not os.path.exists(self.macro_snapshot_path):
            return {}
            
        with open(self.macro_snapshot_path, "r") as f:
            snapshot = yaml.safe_load(f) or {}
            
        today = datetime.now(timezone.utc).date()
        for key, data in snapshot.items():
            if "as_of" in data:
                as_of_date = datetime.strptime(data["as_of"], "%Y-%m-%d").date()
                staleness_days = (today - as_of_date).days
                data["staleness_days"] = staleness_days
                if staleness_days > self.max_staleness_days:
                    # Attempt live pull
                    success = self._attempt_live_macro_pull(key, data)
                    if not success and not allow_stale and staleness_days > 180:
                         raise ValueError(f"Data for {key} is {staleness_days} days stale, exceeding 180 days hard limit.")
        return snapshot

    def _attempt_live_macro_pull(self, key, data):
        """
        Stub for live pulls (RBI, MoSPI, etc.)
        In this implementation, we just catch exceptions and return False
        since actual scraping/API paths are out of scope without real URLs.
        """
        try:
            return False
        except Exception as e:
            logger.warning(f"Live pull failed for {key}: {e}")
            return False

    def fetch_tier1_mca_aggregates(self, allow_stale=False):
        """Fetches aggregate state×sector counts from MCA CDM or cache."""
        # Check cache first to see if we have recent data
        cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith("mca_cdm_")]
        cache_files.sort(reverse=True)
        
        staleness_days = 9999
        if cache_files:
            latest_cache = cache_files[0]
            date_str = latest_cache.replace("mca_cdm_", "").replace(".json", "")
            try:
                cache_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                staleness_days = (datetime.now(timezone.utc).date() - cache_date).days
                if staleness_days <= self.max_staleness_days:
                    with open(os.path.join(self.cache_dir, latest_cache), "r") as f:
                        return json.load(f)
            except ValueError:
                pass
                
        # Try live pull (Simulated failure here since we don't have the real dynamic URL)
        try:
            raise requests.exceptions.ConnectionError("Portal unreachable or URL not defined")
        except Exception as e:
            logger.warning(f"Failed to fetch live Tier 1 data: {e}")
            if cache_files:
                with open(os.path.join(self.cache_dir, cache_files[0]), "r") as f:
                    data = json.load(f)
                    if not allow_stale and staleness_days > 180:
                        raise ValueError(f"MCA data is {staleness_days} days stale, exceeding 180 days limit.")
                    return data
            else:
                # Provide a synthetic fallback if no cache exists for M1 testing
                return self._generate_dummy_mca_aggregates()

    def _generate_dummy_mca_aggregates(self):
        """Generates dummy fallback data for initial testing."""
        return {
            "state_aggregates": {
                "MH": {"n_active": 5000, "paidup_total": 5000000000},
                "DL": {"n_active": 3000, "paidup_total": 3000000000}
            },
            "sector_aggregates": {
                "C": {"n_active": 2000, "paidup_total": 2000000000},
                "G": {"n_active": 1500, "paidup_total": 1000000000}
            }
        }
