import pandas as pd
from pandas_profiling import ProfileReport

for df_name in ['adni_train_corrected', 'adni_test_corrected']:
    print(df_name, '...')
    df_tmp = pd.read_csv(f'{df_name}.csv', index_col=0)
    # Minimal because otherwise file is too big
    profile = ProfileReport(df_tmp, title='ADNI Data Profiling Report', minimal=True)

    profile.to_file(f'adni_freesurfer_report_{df_name}_minimal.html')
