import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

function App() {
  return (
    <main>
      <p>CMSC 672 · Computer Vision</p>
      <h1>Playing Card Analysis</h1>
      <p>The project workspace is ready. Image upload and analysis are the next step.</p>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
