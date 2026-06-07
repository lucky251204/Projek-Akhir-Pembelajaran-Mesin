import re
import pandas as pd

INPUT_CSV  = "pexels_dataset/metadata.csv"
OUTPUT_CSV = "pexels_dataset/metadata_clean.csv"

COPYRIGHT_PATTERNS = [
    '©', 'copyright', 'all rights reserved', 'rights reserved',
    'getty', 'shutterstock', 'alamy', 'corbis', '®', '™'
]

def clean_caption(cap):
    if not isinstance(cap, str):
        return None, 'null'
    
    cap = cap.strip()
    
    for pat in COPYRIGHT_PATTERNS:
        if pat.lower() in cap.lower():
            return None, 'copyright'
    
    if re.match(r'^A photo of ', cap):
        return None, 'fallback'
    
    if re.search(r'\b(IMG|DSC|DSCN|IMGP|_MG_|MVI|MOV)\b', cap):
        return None, 'camera_filename'
    
    if cap.isupper() and len(cap) > 5:
        cap = cap.capitalize()
    
    cap = re.sub(r'[_\[\]{}|\\<>]', ' ', cap)
    cap = re.sub(r'\s+', ' ', cap).strip()
    
    if len(cap.split()) <= 2:
        return None, 'too_short'
    
    return cap, 'ok'

df = pd.read_csv(INPUT_CSV)
print(f"total data: {len(df)}")

results = df['caption'].apply(clean_caption)
df['caption_clean'] = results.apply(lambda x: x[0])
df['drop_reason']   = results.apply(lambda x: x[1])

drop_stats = df['drop_reason'].value_counts()
kept    = (df['drop_reason'] == 'ok').sum()
dropped = len(df) - kept

print("\ndrop report:")
for reason, count in drop_stats.items():
    if reason != 'ok':
        print(f"  {reason}: {count} ({count/len(df)*100:.1f}%)")
print(f"  total dropped: {dropped}")
print(f"  total kept: {kept}")

df_clean = df[df['drop_reason'] == 'ok'].copy()
df_clean['caption'] = df_clean['caption_clean']
df_clean = df_clean[['file', 'title', 'caption', 'keyword', 'source', 'id', 'hash']]
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\nsaved: {OUTPUT_CSV} ({len(df_clean)} rows)")