export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';
export const PAGES = ['Home', 'How it works', 'Try the predictor', 'Sample result', 'Model & science', 'About'];
export const MODELS = [
  ['Random Forest', 'balanced, fast'],
  ['XGBoost', 'highest accuracy'],
  ['Support Vector Regression (SVR)', 'smooth estimates'],
  ['Gaussian Process Regression (GPR)', 'gives confidence range'],
];
export const FALLBACK_PRESETS = [{ id: 'aspirin-abl1', drug_name: 'Aspirin', target_name: 'ABL1', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O', sequence: '', description: 'Aspirin paired with ABL1 for the sample result flow.' }];
