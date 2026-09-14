export const pipeline = [
  ['01', 'Read the drug & protein', 'The drug is written as a text code (SMILES); the protein as its amino-acid sequence. No 3D lab data needed.'],
  ['02', 'Turn them into numbers', 'Chemistry software converts each into structured features — shape, composition, physical properties.'],
  ['03', 'Ask the model', 'Four different ML algorithms independently estimate how strong the interaction is.'],
  ['04', 'Explain the answer', 'The tool shows not just a score, but which features pushed it up or down.'],
];
export const whyCards = [
  ['clock', 'Faster than lab testing', 'Screening a candidate drug computationally takes seconds. Testing it physically can take weeks and real money.'],
  ['shield', 'Honestly evaluated', 'We test on drugs and proteins the model has never seen before — not just shuffled copies of its training data.'],
  ['signal', 'Not a black box', 'Every prediction comes with a plain-language reason, so the “why” is never hidden behind the score.'],
];
export const terms = [
  ["What's a kinase?", 'A kinase is a protein that acts like a molecular switch — it turns other proteins on or off inside a cell. When kinases misfire, cells can grow uncontrollably, which is why they are one of the most important drug targets in cancer treatment.', 'a light switch. The wrong drug flips it the wrong way; the right drug holds it exactly where you want it.'],
  ['What’s “binding affinity”?', "It's a number describing how strongly a drug molecule sticks to its target protein. Higher affinity generally means the drug is more effective at that target — and less likely to need a huge dose.", 'how snugly a key fits a lock. A loose fit barely turns it; a snug fit works reliably.'],
  ['Why “cold-split” evaluation?', 'Most simple tests let a model see similar drugs or proteins during both training and testing — which makes it look smarter than it is. A cold split removes that overlap entirely, so the reported accuracy reflects real-world performance on drugs it has genuinely never encountered.', 'testing a student on questions from a totally different textbook, not the one they studied from.'],
  ['What is SHAP, and why show it?', 'SHAP is a method that scores how much each input feature (a molecule shape, a protein property) pushed the final prediction up or down. Instead of trusting the model blindly, you can see exactly what it noticed.', 'a teacher showing their working, not just the final answer on the exam.'],
];
export const modelDescriptions = [
  ['Random Forest', 'Many decision trees voting together. Robust, hard to fool, good default baseline.'],
  ['XGBoost / LightGBM', "Trees built one after another, each correcting the last one's mistakes. Usually the most accurate."],
  ['Support Vector Regression', 'Finds the smoothest boundary that still respects the data. Good with smaller, cleaner feature sets.'],
  ['Gaussian Process', 'Slower, but tells you how confident it is in each prediction — not just the number itself.'],
];
export const team = [['A', 'Data & Drug Features', 'Dataset cleaning, split design, fingerprint & descriptor pipeline.'], ['B', 'Protein Features & Models', 'Protein feature pipeline, training and tuning all four algorithms.'], ['C', 'Evaluation & Interpretability', 'Metrics, SHAP analysis, pharmacophore sanity checks.'], ['D', 'App & Paper', 'Web app build, IEEE paper compilation, presentation.']];
