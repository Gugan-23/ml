import React from 'react';
import ReactDOM from 'react-dom';
import App from './App'; // Importing the main App component
import './index.css'; // Optional: Importing a CSS file for styles

// Rendering the App component into the root element in index.html
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
