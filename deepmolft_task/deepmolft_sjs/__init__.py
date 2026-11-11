"""DeepMOLFT late-fusion DNN + SHAP for Sjogren's salivary-gland RNA (PyTorch port of priyashdhakal/DeepMOLFT).

torch must be imported before shap in this environment (otherwise torch's shm.dll fails to load on Windows),
so it is imported here, first.
"""
import torch  # noqa: F401  (import order matters, see above)
