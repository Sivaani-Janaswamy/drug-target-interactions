import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function response(body, ok = true) { return Promise.resolve({ ok, json: () => Promise.resolve(body) }); }
const presets = [{ id: 'aspirin-abl1', drug_name: 'Aspirin', target_name: 'ABL1', smiles: 'CCO', sequence: 'ACDEFGHIKLMNPQRSTVWY', description: 'test' }];
const benchmarks = [{ Algorithm: 'Random Forest', Split: 'Random Split (Baseline)', RMSE: 1, CI: 0.5, 'Pearson r': 0.4 }];
const prediction = { drug: { name: 'Aspirin' }, target: { name: 'ABL1' }, score: 11.2, label: 'Moderate Binding Affinity', gauge_percent: 45, uncertainty: null, explanation: 'Moderate result.', features: [{ technical_name: 'Desc_MolWt', name: 'Molecular weight', value: -0.2, direction: 'negative', explanation: 'Pulled the score down.' }] };

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn((url, options = {}) => {
    if (url.endsWith('/presets')) return response(presets);
    if (url.endsWith('/benchmarks')) return response(benchmarks);
    if (url.endsWith('/predict')) return options.body.includes('protein_sequence') && JSON.parse(options.body).protein_sequence ? response(prediction) : response({ detail: 'A protein sequence is required.' }, false);
    return response({ answer: 'SHAP answer' });
  }));
});
afterEach(() => cleanup());

describe('DTI-ML app shell', () => {
  it('renders Home and navigates through all primary pages', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByRole('heading', { name: /Predicting whether a drug/i })).toBeInTheDocument();
    for (const page of ['How it works', 'Try the predictor', 'Sample result', 'Model & science', 'About']) {
      await user.click(screen.getByRole('button', { name: page }));
      expect(screen.getByRole('heading', { name: new RegExp(page === 'Sample result' ? 'No prediction yet' : page === 'Try the predictor' ? 'Try a prediction' : page === 'Model & science' ? 'How the models actually compare' : page === 'How it works' ? 'Five terms' : 'About this project', 'i') })).toBeInTheDocument();
    }
  });

  it('opens chat and submits a suggested question', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Ask about this result' }));
    expect(screen.getByText('Ask about this result')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Explain SHAP simply' }));
    expect(await screen.findByText('SHAP answer')).toBeInTheDocument();
  });

  it('supports predictor selection, validation, and successful result navigation', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Try the predictor' }));
    await user.click(screen.getByRole('button', { name: 'Aspirin' }));
    expect(screen.getByLabelText(/Drug molecule/i)).toHaveValue('CCO');
    await user.click(screen.getByRole('button', { name: /^SVR/ }));
    expect(screen.getByRole('button', { name: /^SVR/ })).toHaveClass('selected');
    await user.click(screen.getByRole('button', { name: 'Predict binding affinity →' }));
    expect(await screen.findByRole('heading', { name: /Aspirin × ABL1/i })).toBeInTheDocument();
  });

  it('shows validation feedback when the protein is empty', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Try the predictor' }));
    await user.click(screen.getByRole('button', { name: 'Predict binding affinity →' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/request could not be completed|protein/i);
  });
});
