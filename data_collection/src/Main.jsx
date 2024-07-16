import React, { useState } from 'react';
import "./styles/main.css";

const Main = () => {
  const [clickedGoogleForm, setClickedGoogleForm] = useState(false);

  const openGoogleFormInNewTab = () => {
    const google_form_url = "https://forms.gle/1euGXApzzDAnQcwj8";
    window.open(google_form_url, '_blank', 'noopener,noreferrer');
  };

  const clickGoogleForm = () => {
    openGoogleFormInNewTab();
    setClickedGoogleForm(true);
  }

  return (
    <div>
      {!clickedGoogleForm ? (
        <div className="main-div">
          <div>
            <h1>Thanks for joining our experiment!</h1>
            <p>Once you finish filling a consent form, please click "Google Form" button to fill in more information.</p>
            <p>Once you finish please come back to this tab. <u>No need to reload the page</u>. </p>
            <br></br>
            {!clickedGoogleForm ? (
              <button className="btn" onClick={() => clickGoogleForm()}>Google Form</button>
            ) : (
              <button className="btn-disable">Google Form</button>
            )}
          </div>
        </div>
      ) : (
        <div className="about-div">
          <div>
            <h2>About Experiment</h2>
            <p>In this experiment you will be asked to click a button while looking a the screen.</p>
            <p style={{ font: 'bold', color: 'red' }}> Please make sure to look at the screen and do not look away.</p>
            <br></br>
            <p>We will show you some different color and brightness screen to collect various pupil data.</p>
            <p>Feel free to stop if you are not feeling well while looking at the screen.</p>
            <p>Please click "Start Experiment" to begin the data collection.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Main;
