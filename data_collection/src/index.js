import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router } from 'react-router-dom';

import App from './App';

const element = (
  <Router>
    <App />
  </Router>
);

const container = document.getElementById('root');
ReactDOM.render(element, container);
