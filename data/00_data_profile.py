import pandas as pd
from pandas_profiling import ProfileReport

df_adni = pd.read_csv('raw_collated_freesurfer.csv', sep='\t')

# Minimal because otherwise file is too big
profile = ProfileReport(df_adni, title='ADNI Data Profiling Report', minimal=True)

profile.to_file("adni_freesurfer_report_raw_minimal.html")