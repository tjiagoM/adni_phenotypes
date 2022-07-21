import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES_CORTICAL = ['thickness_bankssts_lh', 'thickness_caudalanteriorcingulate_lh',
                     'thickness_caudalmiddlefrontal_lh', 'thickness_cuneus_lh', 'thickness_entorhinal_lh',
                     'thickness_fusiform_lh', 'thickness_inferiorparietal_lh', 'thickness_inferiortemporal_lh',
                     'thickness_isthmuscingulate_lh', 'thickness_lateraloccipital_lh',
                     'thickness_lateralorbitofrontal_lh', 'thickness_lingual_lh', 'thickness_medialorbitofrontal_lh',
                     'thickness_middletemporal_lh', 'thickness_parahippocampal_lh', 'thickness_paracentral_lh',
                     'thickness_parsopercularis_lh', 'thickness_parsorbitalis_lh', 'thickness_parstriangularis_lh',
                     'thickness_pericalcarine_lh', 'thickness_postcentral_lh', 'thickness_posteriorcingulate_lh',
                     'thickness_precentral_lh', 'thickness_precuneus_lh', 'thickness_rostralanteriorcingulate_lh',
                     'thickness_rostralmiddlefrontal_lh', 'thickness_superiorfrontal_lh',
                     'thickness_superiorparietal_lh', 'thickness_superiortemporal_lh', 'thickness_supramarginal_lh',
                     'thickness_frontalpole_lh', 'thickness_temporalpole_lh', 'thickness_transversetemporal_lh',
                     'thickness_insula_lh', 'thickness_bankssts_rh', 'thickness_caudalanteriorcingulate_rh',
                     'thickness_caudalmiddlefrontal_rh', 'thickness_cuneus_rh', 'thickness_entorhinal_rh',
                     'thickness_fusiform_rh', 'thickness_inferiorparietal_rh', 'thickness_inferiortemporal_rh',
                     'thickness_isthmuscingulate_rh', 'thickness_lateraloccipital_rh',
                     'thickness_lateralorbitofrontal_rh', 'thickness_lingual_rh', 'thickness_medialorbitofrontal_rh',
                     'thickness_middletemporal_rh', 'thickness_parahippocampal_rh', 'thickness_paracentral_rh',
                     'thickness_parsopercularis_rh', 'thickness_parsorbitalis_rh', 'thickness_parstriangularis_rh',
                     'thickness_pericalcarine_rh', 'thickness_postcentral_rh', 'thickness_posteriorcingulate_rh',
                     'thickness_precentral_rh', 'thickness_precuneus_rh', 'thickness_rostralanteriorcingulate_rh',
                     'thickness_rostralmiddlefrontal_rh', 'thickness_superiorfrontal_rh',
                     'thickness_superiorparietal_rh', 'thickness_superiortemporal_rh', 'thickness_supramarginal_rh',
                     'thickness_frontalpole_rh', 'thickness_temporalpole_rh', 'thickness_transversetemporal_rh',
                     'thickness_insula_rh']

FEATURES_VOLUME = ['volume_bankssts_lh', 'volume_caudalanteriorcingulate_lh', 'volume_caudalmiddlefrontal_lh',
                   'volume_cuneus_lh', 'volume_entorhinal_lh', 'volume_fusiform_lh', 'volume_inferiorparietal_lh',
                   'volume_inferiortemporal_lh', 'volume_isthmuscingulate_lh', 'volume_lateraloccipital_lh',
                   'volume_lateralorbitofrontal_lh', 'volume_lingual_lh', 'volume_medialorbitofrontal_lh',
                   'volume_middletemporal_lh', 'volume_parahippocampal_lh', 'volume_paracentral_lh',
                   'volume_parsopercularis_lh', 'volume_parsorbitalis_lh', 'volume_parstriangularis_lh',
                   'volume_pericalcarine_lh', 'volume_postcentral_lh', 'volume_posteriorcingulate_lh',
                   'volume_precentral_lh', 'volume_precuneus_lh', 'volume_rostralanteriorcingulate_lh',
                   'volume_rostralmiddlefrontal_lh', 'volume_superiorfrontal_lh', 'volume_superiorparietal_lh',
                   'volume_superiortemporal_lh', 'volume_supramarginal_lh', 'volume_frontalpole_lh',
                   'volume_temporalpole_lh', 'volume_transversetemporal_lh', 'volume_insula_lh', 'volume_bankssts_rh',
                   'volume_caudalanteriorcingulate_rh', 'volume_caudalmiddlefrontal_rh', 'volume_cuneus_rh',
                   'volume_entorhinal_rh', 'volume_fusiform_rh', 'volume_inferiorparietal_rh',
                   'volume_inferiortemporal_rh', 'volume_isthmuscingulate_rh', 'volume_lateraloccipital_rh',
                   'volume_lateralorbitofrontal_rh', 'volume_lingual_rh', 'volume_medialorbitofrontal_rh',
                   'volume_middletemporal_rh', 'volume_parahippocampal_rh', 'volume_paracentral_rh',
                   'volume_parsopercularis_rh', 'volume_parsorbitalis_rh', 'volume_parstriangularis_rh',
                   'volume_pericalcarine_rh', 'volume_postcentral_rh', 'volume_posteriorcingulate_rh',
                   'volume_precentral_rh', 'volume_precuneus_rh', 'volume_rostralanteriorcingulate_rh',
                   'volume_rostralmiddlefrontal_rh', 'volume_superiorfrontal_rh', 'volume_superiorparietal_rh',
                   'volume_superiortemporal_rh', 'volume_supramarginal_rh', 'volume_frontalpole_rh',
                   'volume_temporalpole_rh', 'volume_transversetemporal_rh', 'volume_insula_rh']

FEATURES_VOLUME_EXTRA = ['volume_Left-Cerebellum-White-Matter', 'volume_Left-Cerebellum-Cortex',
                         'volume_Left-Thalamus-Proper', 'volume_Left-Caudate', 'volume_Left-Putamen',
                         'volume_Left-Pallidum', 'volume_Brain-Stem', 'volume_Left-Hippocampus', 'volume_Left-Amygdala',
                         'volume_Left-Accumbens-area', 'volume_Right-Cerebellum-White-Matter',
                         'volume_Right-Cerebellum-Cortex', 'volume_Right-Thalamus-Proper', 'volume_Right-Caudate',
                         'volume_Right-Putamen', 'volume_Right-Pallidum', 'volume_Right-Hippocampus',
                         'volume_Right-Amygdala', 'volume_Right-Accumbens-area']


class BrainFeaturesDataset(Dataset):
    """
    Class to load ADNI brain features dataset. It assumes all features were previously corrected/normalised
    """

    def __init__(self, csv_path: str, has_target: bool = True, keep_ids: bool = False):
        """
        :param csv_path: Location of CSV file. Column `diagnosis` assumed to have targetted label"
        """
        df_adni = pd.read_csv(csv_path, index_col=0)

        # Ensuring this order in all dataframes for correct ordering
        self.X = df_adni[FEATURES_CORTICAL + FEATURES_VOLUME + FEATURES_VOLUME_EXTRA].to_numpy(dtype=np.float32)

        assert self.X.shape[1] == len(
            FEATURES_CORTICAL + FEATURES_VOLUME + FEATURES_VOLUME_EXTRA), 'Something is wrong with dataframe shape!'

        if has_target:
            self.target = df_adni['diagnosis'].to_numpy(dtype=np.float32)
            assert sorted(np.unique(self.target)) == [0, 1], 'Something is wrong with dataframe shape!'
        else:
            self.target = [-1 for _ in range(len(self.X))]

        self.keep_ids = keep_ids
        if keep_ids:
            self.ids = df_adni.index.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        if not self.keep_ids:
            return self.X[idx], self.target[idx]
        else:
            return self.ids[idx], self.X[idx], self.target[idx]
