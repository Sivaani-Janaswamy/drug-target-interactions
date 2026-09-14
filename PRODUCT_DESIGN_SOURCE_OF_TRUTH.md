# DTI-ML Product Design Source of Truth

**Status:** Approved for implementation
**Product:** DTI-ML Drug-Target Binding Predictor
**Authoritative visual reference:** [`dti-ml-frontend-mockup.html`](dti-ml-frontend-mockup.html)
**Frontend direction:** React
**Backend direction:** Python API serving the existing ML pipeline
**Date:** 2026-09-14

## 1. Decision

The HTML mockup is the required production product design. It is not a loose visual reference or an optional prototype.

The production application must reproduce the mockup's:

- Information architecture
- Page names and navigation
- Layout structure
- Typography
- Color system
- Spacing and sizing
- Components
- Forms and controls
- Animations
- Chat widget behavior
- Responsive behavior
- Content hierarchy
- Visual states

The production implementation will use React for the browser interface and a Python API for prediction, benchmark, and chatbot functionality. Streamlit may remain temporarily as a development or fallback interface, but it is not the production frontend and must not redefine the product design.

## 2. Why React Is Required

The mockup directly controls its DOM, CSS, JavaScript state, transitions, responsive breakpoints, and page-level interactions. Streamlit generates its own widget DOM and performs full script reruns, which prevents reliable pixel-level and behavior-level parity with the mockup.

React is required because it gives the product direct control over:

- The sticky horizontal header
- Client-side page navigation
- Custom preset chips
- Custom model-selection cards
- Docking animation and replay
- Animated affinity gauge
- Chat panel state
- Loading and error states
- Responsive layouts
- Exact markup and accessibility attributes

The existing Python feature extraction, model loading, prediction, SHAP, benchmark, and chatbot logic should be reused behind an API rather than rewritten in JavaScript.

## 3. Authority Rules

1. This document defines the production design requirements.
2. [`dti-ml-frontend-mockup.html`](dti-ml-frontend-mockup.html) is the pixel and interaction reference.
3. The React implementation must follow the mockup unless a change is explicitly recorded in this document.
4. Backend limitations must not silently change the design. If a backend value is unavailable, the frontend must show a defined loading, unavailable, or error state that preserves the mockup structure.
5. Streamlit-specific layouts, widgets, sidebars, rerun behavior, and generated markup must not be copied into the React product when they conflict with the mockup.
6. New visual patterns, pages, cards, navigation items, color themes, or content hierarchies require an explicit design decision before implementation.

## 4. Product Architecture

```text
React frontend
    |
    | HTTP/JSON
    v
Python API
    |
    +-- RDKit drug features
    +-- Protein features
    +-- Trained model checkpoints
    +-- Affinity prediction
    +-- SHAP attribution
    +-- Benchmark results
    +-- Scoped chatbot service
```

### 4.1 Frontend

The frontend should be a React application with:

- Componentized page sections
- CSS variables matching the mockup tokens
- Client-side navigation or a router preserving the six mockup pages
- Local UI state for navigation, form values, selected model, animations, chat, loading, and errors
- API client functions isolated from presentation components
- Responsive behavior matching the mockup breakpoint at approximately `820px`

### 4.2 Backend

The backend should expose the existing Python functionality through a small API. Recommended endpoints:

```text
GET  /api/health
GET  /api/presets
GET  /api/benchmarks
POST /api/predict
POST /api/chat
```

The backend must:

- Validate SMILES and protein input
- Validate the selected model
- Reuse `features/featurizer.py`
- Reuse `app/helpers.py` model loading and score interpretation where appropriate
- Reuse `evaluation/shap_utils.py`
- Load models once and cache them
- Return stable JSON response shapes
- Return explicit HTTP errors for invalid input
- Avoid exposing Streamlit session state as an API dependency

## 5. Required Pages

The production app must contain exactly these primary navigation items, in this order:

1. Home
2. How it works
3. Try the predictor
4. Sample result
5. Model & science
6. About

The navigation must match the mockup's horizontal header. A permanent Streamlit-style sidebar is not acceptable as the production navigation.

## 6. Global Visual System

The following design tokens are authoritative and must be centralized in the React styles:

```css
--ink: #12201F;
--paper: #F4F7F5;
--paper-alt: #EAEFEA;
--panel: #FFFFFF;
--line: #D7DED9;
--teal: #1F6F63;
--teal-deep: #0F4A41;
--amber: #DE9A34;
--amber-deep: #B87A1F;
--graphite: #5B655F;
--danger: #B84B3A;
--radius: 10px;
--shadow: 0 1px 2px rgba(18,32,31,0.06), 0 6px 20px rgba(18,32,31,0.05);
```

### 6.1 Typography

Use the same font families as the mockup:

- `Space Grotesk` for headings, display text, brand text, and major labels
- `Inter` for body copy, controls, and general UI text
- `IBM Plex Mono` for technical labels, feature names, metadata, and score scales

Do not replace these with browser defaults, Arial, Roboto, or a generic system stack unless the fonts fail to load, in which case the fallback must remain visually close.

### 6.2 Page foundation

- Background: `var(--paper)`
- Primary text: `var(--ink)`
- Secondary text: `var(--graphite)`
- Panels: white with a light border
- Border radius: generally 8px to 16px as specified by the mockup
- Content max width: approximately `1180px`
- Main horizontal padding: approximately `24px`
- Navigation header: sticky at the top with translucent paper background and backdrop blur
- Page changes: fade and slight upward entrance transition

## 7. Header and Navigation Requirements

The header must match the mockup:

- Sticky top position
- Translucent `var(--paper)` background
- Backdrop blur
- Bottom border using `var(--line)`
- Content max width approximately `1180px`
- Brand mark with the three-circle connected-node visual
- Brand name `DTI-ML`
- Subtitle `binding affinity predictor`
- Six horizontal navigation links
- Active link styled with teal text, pale teal background, and pale teal border
- Hover state using `var(--paper-alt)`
- Links must remain usable on narrow screens with wrapping behavior matching the mockup

Navigation must update the active page without losing the current frontend state unless the user explicitly starts a new prediction.

## 8. Home Page Requirements

The Home page must contain the following sections in order.

### 8.1 Hero

- Two-column desktop layout
- One-column layout below the mockup breakpoint
- Eyebrow: `VTU Major Project · AI & Data Science`
- Heading: `Predicting whether a drug fits its target — before it's ever tested in a lab.`
- The word `fits` uses teal emphasis and an amber underline
- Supporting copy: `Feed it a drug and a protein. It predicts how tightly they'd bind — tested only on pairs it's never seen before.`
- Primary action: `Try a prediction →`
- Secondary action: `What does this actually mean?`
- Hero visual with the animated drug/protein lock-and-key illustration
- Hero visual must use the same proportions, colors, labels, and motion as the mockup

### 8.2 Pipeline

Heading eyebrow: `The pipeline, in four steps`

Four steps, in order:

1. `Read the drug & protein`
2. `Turn them into numbers`
3. `Ask the model`
4. `Explain the answer`

The pipeline must retain the bordered horizontal strip, step dividers, numbered technical labels, connector arrows, hover state, and two-column mobile behavior defined by the mockup.

### 8.3 Why this matters

Three cards, in order:

1. `Faster than lab testing`
2. `Honestly evaluated`
3. `Not a black box`

The cards must retain their icons, copy, border, radius, hover lift, and three-column-to-two-column responsive behavior.

## 9. How It Works Page Requirements

The page must contain:

- Eyebrow: `Plain-language explainer`
- Heading: `Five terms this project relies on — explained without the jargon`
- Supporting copy matching the mockup
- Four explanatory cards in a two-column grid:
  - `What's a kinase?`
  - `What's "binding affinity"?`
  - `Why "cold-split" evaluation?`
  - `What is SHAP, and why show it?`
- Each card must include its explanatory paragraph and `Think of it as:` analogy block
- Section eyebrow: `The four models we compare`
- Supporting copy matching the mockup
- Model cards for:
  - `Random Forest`
  - `XGBoost / LightGBM`
  - `Support Vector Regression`
  - `Gaussian Process`

The text must remain understandable to a non-specialist and retain the mockup's content hierarchy.

## 10. Try the Predictor Page Requirements

The page must contain:

- Eyebrow: `Interactive demo`
- Heading: `Try a prediction`
- Supporting copy matching the mockup
- Two-column desktop layout
- Stacked layout below the mockup breakpoint

### 10.1 Input panel

The left panel must contain:

- Drug molecule field labeled `Drug molecule — as a SMILES string`
- Drug textarea using technical monospace styling
- Preset chips:
  - `Aspirin`
  - `Imatinib`
  - `Gefitinib`
  - `Paste my own …`
- Target protein field labeled `Target protein — kinase sequence`
- Protein textarea using technical monospace styling
- Preset chips:
  - `ABL1 (leukemia)`
  - `EGFR (lung cancer)`
  - `Paste my own …`
- Model selector with four selectable cards:
  - `Random Forest` / `balanced, fast`
  - `XGBoost` / `highest accuracy`
  - `SVR` / `smooth estimates`
  - `Gaussian Process` / `gives confidence range`
- `Random Forest` selected by default
- Full-width primary action: `Predict binding affinity →`
- Informational KIBA note matching the mockup

Native browser inputs may be used, but the layout and styling must match the mockup. Native Streamlit selectboxes are not acceptable in the production frontend.

### 10.2 Explanation panel

The right panel must contain the heading:

`What happens after you click predict`

It must show the four numbered steps from the mockup, in order, with the same spacing and amber technical step numbers.

### 10.3 Input behavior

- Preset chips fill the relevant field values
- Custom input remains editable
- Model cards update selection visually
- Invalid input displays a clear inline error without destroying the form
- Submit displays a loading state while the API request runs
- Successful prediction navigates to Sample result and preserves the response
- Failed prediction preserves user input and displays an actionable error

## 11. Sample Result Page Requirements

The page must contain:

- Eyebrow: `Sample output`
- A result heading using the selected drug and target, for example `Aspirin × ABL1 — prediction`
- Supporting copy matching the mockup
- Docking scene with the same puzzle-piece geometry, colors, labels, and animation
- Caption states:
  - `Approaching ABL1…`
  - `Testing the fit…`
  - `Bound — moderate affinity`
- `Replay docking` control
- Animated affinity gauge
- Score with `predicted KIBA score` label
- Qualitative binding label
- Weak, Moderate, Strong scale labels
- Plain-language result explanation
- Section eyebrow: `Why the model said this`
- Four reason rows with positive teal or negative red indicator bars
- Informational SHAP note

### 11.1 Result data

The page must render live API output when a prediction has been run. If no prediction exists, it must show a designed empty state with an action to return to Try the predictor while retaining the page structure and visual style.

The API response must distinguish drug and target names. It must never assign the same preset label to both fields by default.

### 11.2 SHAP display

The result must use human-readable feature labels wherever available. Raw names such as `Morgan_Bit_742` may be retained as technical metadata, but they must not be the only explanation shown to the user.

## 12. Model & Science Page Requirements

The page must contain:

- Eyebrow: `For evaluators & the curious`
- Heading: `How the models actually compare`
- Supporting copy matching the mockup
- A styled results table with columns:
  - `Algorithm`
  - `Split`
  - `RMSE ↓`
  - `Concordance Index ↑`
  - `Pearson r ↑`
- Split tags styled like the mockup
- Best values highlighted in teal
- Section eyebrow: `The honest headline`
- Honest headline card matching the mockup
- Responsive table behavior for narrow screens

The table must use actual saved benchmark results from `models/results.csv` or another verified backend source. The placeholder values and placeholder disclaimer in the mockup are design references only and must not be presented as actual results.

## 13. About Page Requirements

The page must contain:

- Eyebrow: `VTU Major Project`
- Heading: `About this project`
- Supporting project description matching the mockup
- Four team cards:
  - `Data & Drug Features`
  - `Protein Features & Models`
  - `Evaluation & Interpretability`
  - `App & Paper`
- Section eyebrow: `References this project builds on`
- Supporting references copy matching the mockup

## 14. Chat Widget Requirements

The chat widget must match the mockup's floating design:

- Fixed circular launcher at bottom-right
- Teal-deep background
- White chat icon
- Floating motion on the icon/launcher consistent with the mockup
- Slide-up panel
- Header: `Ask about this result`
- Status text indicating whether the current prediction is available
- Close control
- Scrollable message body
- Bot and user message styles matching the mockup
- Suggested question chips
- Input field with `Ask a question…` placeholder
- `Send` button

The production chat must use the backend API rather than query-parameter-driven Streamlit reruns. It must preserve the mockup's visual behavior while adding:

- Loading/typing state
- API error fallback
- Scoped answers about DTI-ML, the current prediction, cold splits, models, and SHAP
- Rate limiting on the backend
- No medical advice or unrelated answers

## 15. Responsive Requirements

The React product must implement the mockup's responsive behavior at approximately `820px`:

- Hero changes from two columns to one
- Pipeline changes from four columns to two
- Pipeline connector arrows disappear
- Three-card grids change to two columns
- Two-column grids become one column
- Predictor panels stack vertically
- Hero heading becomes smaller
- Team grid changes to two columns
- Chat panel width is constrained to the viewport
- No text, controls, or cards overflow horizontally

The following viewport sizes must be manually verified:

- 1440 × 900
- 1280 × 800
- 768 × 1024
- 390 × 844

## 16. API Contract

### 16.1 Prediction request

```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "protein_sequence": "MKTAYIAKQR...",
  "model": "Random Forest",
  "drug_name": "Aspirin",
  "target_name": "ABL1"
}
```

### 16.2 Prediction response

```json
{
  "drug": {
    "name": "Aspirin",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
  },
  "target": {
    "name": "ABL1"
  },
  "model": "Random Forest",
  "score": 12.4,
  "label": "Moderate predicted binding",
  "explanation": "Moderate interaction — weaker than known ABL1 inhibitors like Imatinib.",
  "gauge_percent": 52,
  "uncertainty": null,
  "features": [
    {
      "name": "Aromatic ring count",
      "technical_name": "Morgan_Bit_742",
      "value": 0.9,
      "direction": "positive",
      "explanation": "The molecule's ring structure resembles known kinase-binding scaffolds — pushed the score up."
    }
  ]
}
```

The exact numeric values will come from the selected model. The response shape must remain stable even when optional uncertainty or explanation fields are unavailable.

## 17. Accessibility Requirements

- Use semantic headings in page order
- Use real buttons for actions
- Use real links or router links for navigation
- Every input has an associated label
- Chat launcher and close controls have accessible names
- Keyboard users can operate navigation, forms, model cards, chat, and replay
- Focus states must remain visible
- Color must not be the only way to communicate positive versus negative attribution
- Motion should respect `prefers-reduced-motion`

## 18. Implementation Checklist

### Design contract

- [ ] Add this document to the repository
- [ ] Treat `dti-ml-frontend-mockup.html` as the authoritative visual reference
- [ ] Record any intentional deviation in this document before implementation
- [ ] Confirm the final product uses the six mockup pages and no Streamlit sidebar

### Backend extraction

- [ ] Create a reusable prediction service independent of Streamlit
- [ ] Create stable request and response schemas
- [ ] Add FastAPI or Flask backend entrypoint
- [ ] Add `GET /api/health`
- [ ] Add `GET /api/presets`
- [ ] Add `GET /api/benchmarks`
- [ ] Add `POST /api/predict`
- [ ] Add `POST /api/chat`
- [ ] Validate SMILES, protein sequences, and model names
- [ ] Make model paths independent of the process working directory
- [ ] Cache loaded model checkpoints
- [ ] Return separate drug and target names
- [ ] Return human-readable SHAP explanations where possible
- [ ] Add backend error handling and structured error responses
- [ ] Add CORS configuration for local React development

### React application

- [ ] Create React frontend structure
- [ ] Create shared design-token stylesheet
- [ ] Load Space Grotesk, Inter, and IBM Plex Mono
- [ ] Implement exact sticky header and brand mark
- [ ] Implement six-page navigation
- [ ] Implement Home page
- [ ] Implement How it works page
- [ ] Implement Try the predictor page
- [ ] Implement Sample result page
- [ ] Implement Model & science page
- [ ] Implement About page
- [ ] Implement floating chat widget
- [ ] Implement page transitions
- [ ] Implement preset chips
- [ ] Implement model-selection cards
- [ ] Implement loading states
- [ ] Implement validation and error states
- [ ] Implement docking animation
- [ ] Implement docking replay
- [ ] Implement animated affinity gauge
- [ ] Implement SHAP reason rows
- [ ] Implement responsive table
- [ ] Implement mobile responsive layouts
- [ ] Implement reduced-motion behavior
- [ ] Implement keyboard and screen-reader support

### Data and content parity

- [ ] Match all mockup page titles and section labels
- [ ] Match all mockup button labels
- [ ] Match all mockup explanatory copy unless a factual correction is approved
- [ ] Match all model names consistently across frontend, API, and saved results
- [ ] Match preset names to separate drug and target metadata
- [ ] Replace mockup placeholder benchmark values with actual verified results
- [ ] Replace raw SHAP-only labels with user-readable descriptions
- [ ] Ensure the result heading uses the actual selected drug and target

### Visual and behavior verification

- [ ] Compare React screenshots against the mockup at 1440 × 900
- [ ] Compare at 1280 × 800
- [ ] Compare at 768 × 1024
- [ ] Compare at 390 × 844
- [ ] Verify no horizontal overflow
- [ ] Verify header active state
- [ ] Verify navigation transitions to the correct page
- [ ] Verify hero buttons navigate correctly
- [ ] Verify preset chips populate fields
- [ ] Verify model cards select correctly
- [ ] Verify invalid input errors are visible and useful
- [ ] Verify successful prediction reaches Sample result
- [ ] Verify gauge reflects the API score
- [ ] Verify docking animation and replay
- [ ] Verify SHAP positive and negative states
- [ ] Verify benchmark table uses API data
- [ ] Verify chat launcher, panel, chips, send, loading, and fallback behavior
- [ ] Verify reduced-motion mode
- [ ] Verify backend health and API failure states

### Retirement of Streamlit production role

- [ ] Keep Streamlit working during the migration as a fallback
- [ ] Document that Streamlit is not the production design authority
- [ ] Remove or archive Streamlit-specific UI code after React parity is accepted
- [ ] Update README startup and deployment instructions
- [ ] Update paper and project documentation to describe the final React/API architecture

## 19. Definition of Done

The migration is complete only when:

1. The React app renders the same six-page product structure as the mockup.
2. The visual system and responsive behavior match the mockup at the required viewport sizes.
3. The predictor performs real inference through the Python API.
4. The result page renders real score, uncertainty, and SHAP data.
5. The science page renders verified benchmark results.
6. The chatbot retains the mockup's appearance and provides scoped backend answers.
7. All required loading, empty, validation, error, and reduced-motion states are implemented.
8. The app passes the visual and behavior checklist above.
9. Any intentional deviation is documented here and approved before release.

This document and the mockup together are the source of truth for production product design.
