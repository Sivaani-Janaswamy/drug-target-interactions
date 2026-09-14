export function PageIntro({ eyebrow, title, children }) { return <section className="page-intro"><div className="eyebrow">{eyebrow}</div><h2>{title}</h2>{children && <p className="lede">{children}</p>}</section>; }
export function Eyebrow({ children }) { return <div className="eyebrow">{children}</div>; }
export function ActionButton({ children, onClick, ghost = false, className = '', type = 'button' }) { return <button type={type} className={`btn ${ghost ? 'btn-ghost' : 'btn-primary'} ${className}`} onClick={onClick}>{children}</button>; }
