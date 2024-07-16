import React, { useState } from 'react';
import Main from "./Main";
import Experiment from "./page/experiment";

const App = () => {
  const [currentPage, setCurrentPage] = useState('main');

  const renderPage = () => {
    switch (currentPage) {
      case 'main':
        return <Main />;
      case 'experiment':
        return <Experiment />;
      default:
        return <Main />;
    }
  };

  const renderButton = () => {
    switch (currentPage) {
      case 'main':
        return <div class="button-container"><button className="btn" onClick={() => setCurrentPage('experiment')}>Start Experiment</button></div>;
      case 'experiment':
        return <div class="button-container"></div>;
      default:
        return <div class="button-container"><button className="btn" onClick={() => setCurrentPage('main')}>HOME</button></div>;
    }
  }

  return (
    <div>
      {renderPage()}
      {renderButton()}
    </div>
  );
};

export default App;
