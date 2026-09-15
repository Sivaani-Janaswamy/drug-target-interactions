import { useState } from 'react';
import { askChat } from '../app/api';

export default function ChatWidget({ result }) {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hi — I can explain any term or number on this page in plain English. Tap a question below, or type your own.' }
  ]);

  const send = async (preset = '') => {
    const value = (preset || question).trim();
    if (!value || busy) return;
    
    setQuestion('');
    setMessages(current => [
      ...current, 
      { role: 'user', text: value },
      { role: 'typing' }
    ]);
    setBusy(true);
    
    try {
      const response = await askChat(value, result || {});
      const sourceText = response.sources && response.sources.length > 0 
        ? `\n\nSource: ${response.sources.map(s => s.title).join(', ')}` 
        : '';
      setMessages(current => [
        ...current.filter(item => item.role !== 'typing'),
        { role: 'bot', text: response.answer + sourceText }
      ]);
    } catch {
      setMessages(current => [
        ...current.filter(item => item.role !== 'typing'),
        { role: 'bot', text: 'I could not reach the explanation service just now — try again in a moment.' }
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button 
        className="chat-launcher" 
        onClick={() => setOpen(value => !value)} 
        aria-label="Ask about this result"
      >
        <span aria-hidden="true">▱</span>
      </button>
      
      {open && (
        <div className="chat-panel">
          <div className="chat-head">
            <div>
              <h4>Ask about this result</h4>
              <span>{result ? 'grounded to your current prediction' : 'run a prediction for grounded answers'}</span>
            </div>
            <button 
              className="chat-close" 
              onClick={() => setOpen(false)} 
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
          
          <div className="chat-body">
            {messages.map((message, index) => 
              message.role === 'typing' ? (
                <div className="chat-typing" key={index}>
                  <i /><i /><i />
                </div>
              ) : (
                <div className={`chat-msg ${message.role}`} key={index}>
                  {message.text}
                </div>
              )
            )}
          </div>
          
          <div className="chat-suggest">
            {result && (
              <button className="chip" onClick={() => send('What does this score mean?')}>
                What does this score mean?
              </button>
            )}
            <button className="chip" onClick={() => send('Why lower on cold split?')}>
              Why lower on cold split?
            </button>
            <button className="chip" onClick={() => send('Explain SHAP simply')}>
              Explain SHAP simply
            </button>
            <button className="chip" onClick={() => send('What dataset does this project use?')}>
              What dataset does this project use?
            </button>
            <button className="chip" onClick={() => send('What are the feature dimensions?')}>
              What are the feature dimensions?
            </button>
          </div>
          
          <form 
            className="chat-inputrow" 
            onSubmit={event => {
              event.preventDefault();
              send();
            }}
          >
            <input 
              type="text" 
              value={question} 
              onChange={event => setQuestion(event.target.value)} 
              placeholder="Ask a question…" 
            />
            <button type="submit" disabled={busy}>Send</button>
          </form>
        </div>
      )}
    </>
  );
}