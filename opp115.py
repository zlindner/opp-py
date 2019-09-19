import os
import pandas as pd
import numpy as np
from glob import glob

def load():
    if not os.path.exists('./opp115.csv'):
        print('generating dataset...')

        generate_dataset().to_csv('./opp115.csv', sep=',', index=False)

    return pd.read_csv('./opp115.csv', sep=',', header=0)

def generate_dataset():
    p = load_policies()
    a = load_annotations()

    merged = pd.merge(a, p, on=['policy_id', 'segment_id'], how='outer')
    mode = merged.groupby(['policy_id', 'segment_id']).agg(lambda x: x.value_counts().index[0])
    mode.reset_index(inplace=True)

    return mode

def load_policies():
    policies = []

    for f in glob('opp-115/sanitized_policies/*.html'):
        with open(f, 'r') as policy:
            text = policy.read()
            segments = text.split('|||')

            p = pd.DataFrame(columns=['policy_id', 'segment_id', 'text'])
            p['segment_id'] = np.arange(len(segments))
            p['policy_id'] = f[27:-5].split('_')[0]
            p['text'] = segments

            policies.append(p)

    p = pd.concat(policies)
    p.reset_index(inplace=True, drop=True)

    return p

def load_annotations():        
    annotations = []

    for f in glob('opp-115/annotations/*.csv'):
        a = pd.read_csv(f, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attributes', 'date', 'url'])
        a['policy_id'] = f[20:-4].split('_')[0]
        a.drop(['annotation_id', 'batch_id', 'annotator_id', 'date', 'url'], axis=1, inplace=True)
        annotations.append(a)

    a = pd.concat(annotations)
    a.reset_index(inplace=True, drop=True)

    return a
