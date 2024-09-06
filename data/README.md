Folder containing raw and processed data for the model. In the following, a brief explanation of the raw data files: 

- File DTM (.ASC)
- File las originale (.las)
- File las con features (_F.las)

Lista features (oltre alle altre variabili di un normale las):
- 'ndvi'
- 'ndwi'
- 'ssi'
- 'l1_a'
- 'l2_a'
- 'l3_a'
- 'planarity_a'
- 'sphericity_a'
- 'linearity_a'
- 'entropy_a'
- 'theta_a'
- 'theta_variance_a'
- 'mad_a'
- 'delta_z_a'
- 'l1_b'
- 'l2_b'
- 'l3_b'
- 'planarity_b'
- 'sphericity_b'
- 'linearity_b'
- 'entropy_b'
- 'theta_b'
- 'theta_variance_b'
- 'mad_b'
- 'delta_z_b'
- 'N_h'
- 'delta_z_fl'

Le variabili che terminano con “_a” sono calcolate con un intorno di raggio 0.5m, quelle che terminano con “_b” raggio 1m. Quelle senza suffisso sono variabili puntuali.