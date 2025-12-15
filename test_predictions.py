"""
Behavioral Equivalence Test for Repository Cleanup
===================================================

This script captures "golden outputs" from the current Streamlit implementation
BEFORE cleanup, then validates that predictions remain identical AFTER cleanup.

Run this script TWICE:
1. BEFORE cleanup: python test_predictions.py --capture
2. AFTER cleanup:  python test_predictions.py --validate

The script will fail if predictions differ by more than €1 (floating point tolerance).
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.stage3_utils import transform_stage3

# Test cases covering different property types and locations
TEST_CASES = [
    {
        "name": "house_brussels_villa",
        "description": "Luxury villa in Brussels",
        "input": {
            "property_type": "House",
            "property_subtype": "villa",
            "postal_code": 1000,
            "area": 150,
            "rooms": 3,
            "bathrooms": 2,
            "toilets": 2,
            "primary_energy_consumption": 200,
            "state": 2,
            "build_year": 2000,
            "facades_number": 2,
            "has_garage": 1,
            "has_garden": 1,
            "has_terrace": 0,
            "has_equipped_kitchen": 2,
            "has_swimming_pool": 0,
        }
    },
    {
        "name": "apartment_antwerp",
        "description": "Modern apartment in Antwerp",
        "input": {
            "property_type": "Apartment",
            "property_subtype": "apartment",
            "postal_code": 2000,
            "area": 85,
            "rooms": 2,
            "bathrooms": 1,
            "toilets": 1,
            "primary_energy_consumption": 180,
            "state": 4,
            "build_year": 2015,
            "facades_number": 1,
            "has_garage": 0,
            "has_garden": 0,
            "has_terrace": 1,
            "has_equipped_kitchen": 2,
            "has_swimming_pool": 0,
        }
    },
    {
        "name": "house_ghent_residence",
        "description": "Family residence in Ghent",
        "input": {
            "property_type": "House",
            "property_subtype": "residence",
            "postal_code": 9000,
            "area": 200,
            "rooms": 4,
            "bathrooms": 2,
            "toilets": 3,
            "primary_energy_consumption": 150,
            "state": 2,
            "build_year": 1995,
            "facades_number": 3,
            "has_garage": 1,
            "has_garden": 1,
            "has_terrace": 1,
            "has_equipped_kitchen": 2,
            "has_swimming_pool": 0,
        }
    }
]

# Feature order (must match Streamlit app exactly)
REDUCED_FEATURES = [
    "area",
    "postal_code_te_price",
    "locality_te_price",
    "bathrooms",
    "rooms",
    "primary_energy_consumption",
    "state",
    "province_benchmark_m2",
    "postal_code",
    "region_benchmark_m2",
    "property_subtype_te_price",
    "apt_avg_m2_region",
    "toilets",
    "property_type_te_price",
    "median_income",
    "build_year",
    "house_avg_m2_province",
    "has_garage",
    "apt_avg_m2_province",
    "has_garden",
    "has_terrace",
    "facades_number",
    "has_swimming_pool",
    "house_avg_m2_region",
    "has_equipped_kitchen",
]


def load_resources():
    """Load models, pipelines, and lookup data."""
    base = Path(__file__).parent
    
    data_path = base / "data" / "pre_processed" / "pre_processed_data_for_kaggle.csv"
    model_house_path = base / "models" / "model_xgb_house.pkl"
    model_apt_path = base / "models" / "model_xgb_apartment.pkl"
    pipeline_house_path = base / "models" / "stage3_pipeline_house.pkl"
    pipeline_apt_path = base / "models" / "stage3_pipeline_apartment.pkl"
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    lookup_cols = [
        "postal_code", "locality", "province", "region", "median_income",
        "province_benchmark_m2", "region_benchmark_m2", "national_benchmark_m2",
        "house_avg_m2_province", "apt_avg_m2_province",
        "house_avg_m2_region", "apt_avg_m2_region"
    ]
    lookup_df = df[lookup_cols].drop_duplicates(subset=["postal_code"]).set_index("postal_code")
    
    print(f"Loading models...")
    model_house = joblib.load(model_house_path)
    model_apt = joblib.load(model_apt_path)
    stage3_house = joblib.load(pipeline_house_path)
    stage3_apt = joblib.load(pipeline_apt_path)
    
    return lookup_df, model_house, model_apt, stage3_house, stage3_apt


def get_metadata(pc, lookup_df):
    """Get metadata for postal code (same logic as Streamlit app)."""
    if pc in lookup_df.index:
        return lookup_df.loc[pc].to_dict()
    all_pcs = lookup_df.index.values
    nearest = all_pcs[np.abs(all_pcs - pc).argmin()]
    return lookup_df.loc[nearest].to_dict()


def run_prediction(test_input, lookup_df, model_house, model_apt, stage3_house, stage3_apt):
    """Run prediction (exact copy of Streamlit logic)."""
    # Get metadata
    postal_code = test_input["postal_code"]
    metadata = get_metadata(postal_code, lookup_df)
    
    # Enrich input
    input_dict = {**test_input, **metadata}
    
    # Select model and pipeline
    if test_input["property_type"] == "House":
        pipeline = stage3_house
        model = model_house
    else:
        pipeline = stage3_apt
        model = model_apt
    
    # Create DataFrame and transform
    df_input = pd.DataFrame([input_dict])
    df_s3 = transform_stage3(df_input, pipeline)
    
    # Select features in exact order
    X = df_s3[[f for f in REDUCED_FEATURES if f in df_s3.columns]]
    
    # Predict
    prediction = float(model.predict(X)[0])
    
    return prediction


def capture_golden_outputs():
    """Capture golden outputs from current implementation."""
    print("=" * 60)
    print("CAPTURING GOLDEN OUTPUTS")
    print("=" * 60)
    
    lookup_df, model_house, model_apt, stage3_house, stage3_apt = load_resources()
    
    results = {}
    for case in TEST_CASES:
        print(f"\nTest: {case['name']}")
        print(f"  {case['description']}")
        
        prediction = run_prediction(
            case["input"],
            lookup_df,
            model_house,
            model_apt,
            stage3_house,
            stage3_apt
        )
        
        results[case["name"]] = {
            "description": case["description"],
            "predicted_price": prediction
        }
        
        print(f"  Predicted Price: €{prediction:,.2f}")
    
    # Save to file
    output_file = Path(__file__).parent / "golden_outputs.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Golden outputs saved to: {output_file}")
    print("=" * 60)


def validate_against_golden():
    """Validate current predictions against golden outputs."""
    print("=" * 60)
    print("VALIDATING AGAINST GOLDEN OUTPUTS")
    print("=" * 60)
    
    # Load golden outputs
    golden_file = Path(__file__).parent / "golden_outputs.json"
    if not golden_file.exists():
        print(f"❌ ERROR: Golden outputs file not found: {golden_file}")
        print("   Run with --capture first to create golden outputs.")
        sys.exit(1)
    
    with open(golden_file) as f:
        golden = json.load(f)
    
    # Run predictions
    lookup_df, model_house, model_apt, stage3_house, stage3_apt = load_resources()
    
    all_passed = True
    tolerance = 1.0  # €1 tolerance for floating point
    
    for case in TEST_CASES:
        print(f"\nTest: {case['name']}")
        print(f"  {case['description']}")
        
        prediction = run_prediction(
            case["input"],
            lookup_df,
            model_house,
            model_apt,
            stage3_house,
            stage3_apt
        )
        
        golden_price = golden[case["name"]]["predicted_price"]
        diff = abs(prediction - golden_price)
        
        print(f"  Golden:  €{golden_price:,.2f}")
        print(f"  Current: €{prediction:,.2f}")
        print(f"  Diff:    €{diff:,.2f}")
        
        if diff <= tolerance:
            print(f"  ✅ PASS (within €{tolerance} tolerance)")
        else:
            print(f"  ❌ FAIL (exceeds €{tolerance} tolerance)")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Behavioral equivalence confirmed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ TESTS FAILED - Predictions do not match!")
        print("=" * 60)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_predictions.py --capture   # Capture golden outputs")
        print("  python test_predictions.py --validate  # Validate against golden")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "--capture":
        capture_golden_outputs()
    elif mode == "--validate":
        validate_against_golden()
    else:
        print(f"Unknown mode: {mode}")
        print("Use --capture or --validate")
        sys.exit(1)


if __name__ == "__main__":
    main()
