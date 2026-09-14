import { useEffect, useState } from 'react';
import { getBenchmarks, getPresets } from './api';
import { FALLBACK_PRESETS, PAGES } from './constants';
import Header from '../components/Header';
import ChatWidget from '../components/ChatWidget';
import HomePage from '../pages/HomePage';
import HowItWorksPage from '../pages/HowItWorksPage';
import PredictorPage from '../pages/PredictorPage';
import ResultPage from '../pages/ResultPage';
import SciencePage from '../pages/SciencePage';
import AboutPage from '../pages/AboutPage';

export default function App() { const [page,setPage]=useState('Home');const [result,setResult]=useState(null);const [presets,setPresets]=useState(FALLBACK_PRESETS);const [benchmarks,setBenchmarks]=useState([]);useEffect(()=>{getPresets().then(setPresets).catch(()=>{});getBenchmarks().then(setBenchmarks).catch(()=>{});},[]);const navigate=next=>{setPage(next);window.scrollTo({top:0,behavior:'smooth'});};const renderPage=()=>({Home:<HomePage navigate={navigate}/>, 'How it works':<HowItWorksPage/>, 'Try the predictor':<PredictorPage presets={presets} onPrediction={setResult} navigate={navigate}/>, 'Sample result':<ResultPage result={result} navigate={navigate}/>, 'Model & science':<SciencePage benchmarks={benchmarks}/>, About:<AboutPage/>}[page]);return <><Header page={page} navigate={navigate} pages={PAGES}/><main>{renderPage()}</main><footer>DTI-ML · VTU Major Project · production interface</footer><ChatWidget result={result}/></>; }
