import pandas as pd
import json
import os
import numpy as np

# Create directory if it doesn't exist
os.makedirs("result/entropy_vis_paper", exist_ok=True)

# Load data
df_base = pd.read_csv("result/entropy_vis_subset32_base_trace_v3/HAM10000_predictions.csv")
df_skinir = pd.read_csv("result/entropy_vis_subset32_skinir_trace_v3/HAM10000_predictions.csv")

# Function to extract base name from image path
def get_basename(path):
    return os.path.basename(path)

# Apply basename
df_base['basename'] = df_base['image'].apply(get_basename)
df_skinir['basename'] = df_skinir['image'].apply(get_basename)

# Merge
merged = pd.merge(df_base, df_skinir, on='basename', suffixes=('_base', '_skinir'))

all_pairs = []
all_long = []

for idx, row in merged.iterrows():
    basename = row['basename']
    gt = row['ground_truth_base'] # Should be the same
    
    # Comparison layer logic
    target_layer = row.get('memvr_target_layer_skinir')
    trigger_layer = row.get('memvr_trigger_layer_skinir')
    comp_layer = target_layer if pd.notna(target_layer) and target_layer != "" else trigger_layer
    
    trace_base = json.loads(row['memvr_entropy_trace_base'])
    trace_skinir = json.loads(row['memvr_entropy_trace_skinir'])
    
    # Map layers to entropy values
    entropy_map_base = {item['layer']: item['entropy'] for item in trace_base}
    entropy_map_skinir = {item['layer']: item['entropy'] for item in trace_skinir}
    
    layers = [item['layer'] for item in trace_skinir]
    
    # Collect long format
    for layer in layers:
        all_long.append({
            'basename': basename,
            'layer': layer,
            'entropy_base': entropy_map_base.get(layer),
            'entropy_skinir': entropy_map_skinir.get(layer),
            'ground_truth': gt,
            'prediction_base': row['predicted_answer_base'],
            'prediction_skinir': row['predicted_answer_skinir']
        })
    
    # Post-trigger pair
    ent_base_at_comp = entropy_map_base.get(comp_layer)
    ent_skinir_at_comp = entropy_map_skinir.get(comp_layer)
    
    # Check for peak/drop (local peak in base, drop in skinir)
    is_peak_base = False
    is_drop_skinir = False
    
    try:
        idx_comp = layers.index(comp_layer)
        if idx_comp > 0:
            prev_layer = layers[idx_comp - 1]
            if ent_base_at_comp > entropy_map_base.get(prev_layer):
                is_peak_base = True
            if ent_skinir_at_comp < entropy_map_skinir.get(prev_layer):
                is_drop_skinir = True
    except:
        pass

    all_pairs.append({
        'basename': basename,
        'comp_layer': comp_layer,
        'entropy_base': ent_base_at_comp,
        'entropy_skinir': ent_skinir_at_comp,
        'delta': ent_base_at_comp - ent_skinir_at_comp,
        'is_peak_base': is_peak_base,
        'is_drop_skinir': is_drop_skinir,
        'ground_truth': gt,
        'prediction_base': row['predicted_answer_base'],
        'prediction_skinir': row['predicted_answer_skinir']
    })

df_pairs = pd.DataFrame(all_pairs)
df_long = pd.DataFrame(all_long)

# Save main output
df_pairs.to_csv("result/entropy_vis_paper/corrected_subset_post_trigger_pairs.csv", index=False)
df_long.to_csv("result/entropy_vis_paper/corrected_subset_post_trigger_long.csv", index=False)

# Selection criteria
peak_drop_candidates = df_pairs[df_pairs['is_peak_base'] & df_pairs['is_drop_skinir']]
if not peak_drop_candidates.empty:
    selected = peak_drop_candidates.sort_values(by='delta', ascending=False).iloc[0]
else:
    selected = df_pairs.sort_values(by='delta', ascending=False).iloc[0]

# Generate single_case_entropy_curve.csv
selected_basename = selected['basename']
single_case_rows = df_long[df_long['basename'] == selected_basename].copy()
# Format ground_truth and prediction as clean strings
selected_row_in_merged = merged[merged['basename'] == selected_basename].iloc[0]
single_case_rows['ground_truth'] = str(selected_row_in_merged['ground_truth_base'])
single_case_rows['prediction_base'] = str(selected_row_in_merged['predicted_answer_base'])
single_case_rows['prediction_skinir'] = str(selected_row_in_merged['predicted_answer_skinir'])

# Also add the unified "prediction" column for the plot script if needed
single_case_rows['prediction'] = single_case_rows['prediction_skinir']

single_case_rows.to_csv("result/entropy_vis_paper/single_case_entropy_curve.csv", index=False)

# Summary
avg_diff = df_pairs['delta'].mean()
summary_text = f"Selected Basename: {selected_basename}\n"
summary_text += f"Comparison Layer: {selected['comp_layer']}\n"
summary_text += f"Base Entropy at Comp: {selected['entropy_base']}\n"
summary_text += f"SkinIR Entropy at Comp: {selected['entropy_skinir']}\n"
summary_text += f"Delta (Base - SkinIR): {selected['delta']}\n"
summary_text += f"Overall Mean Delta: {avg_diff}\n"

with open("result/entropy_vis_paper/summary.txt", "w") as f:
    f.write(summary_text)

print(summary_text)
