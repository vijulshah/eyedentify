import React, { useEffect, useRef, useState } from 'react';
import { CSVLink } from "react-csv";
import "../styles/experiment.css";

const Experiment = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [currentSession, setCurrentSession] = useState(0);
  const [timestamps, setTimestamps] = useState(["session_id", "start", "end"]);
  const currentSessionCountRef = useRef(1);
  const [timestampArray, setTimestampArray] = useState([]);
  const [downloadCsvEnableStatus, setDownloadCsvEnableStatus] = useState(false);
  let interval;

  const addTimestamp = (sessionName) => {
    const newTimestamp = new Date();

    if (sessionName) {
      setTimestamps(prevTimestamps => [...prevTimestamps, sessionName, newTimestamp.getTime()]);
    } else {
      setTimestamps(prevTimestamps => [...prevTimestamps, newTimestamp.getTime()]);
    }
  };

  const chunkArray = (array, size) => {
    const chunkedArr = [];
    for (let i = 0; i < array.length; i += size) {
      chunkedArr.push(array.slice(i, i + size));
    }

    setDownloadCsvEnableStatus(true);
    return chunkedArr;
  };

  const getVideo = async () => {
    const constraints = { video: true };
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
    } catch (err) {
      console.error('Error accessing the camera:', err);
    }
  };

  useEffect(() => {
    getVideo();
    setCurrentSession("start");
    return () => {
      videoRef.current.srcObject?.getTracks().forEach(track => track.stop());
    };
  }, []);

  const handleDataAvailable = ({ data }) => {
    if (data.size > 0) {
      setRecordedChunks(prev => [...prev, data]);
    }
  };

  const startRecording = () => {
    setRecordedChunks([]);
    let options = { mimeType: 'video/mp4' };

    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      console.log(`${options.mimeType} not supported, falling back to default WebM.`);
      options = { mimeType: 'video/webm; codecs=vp8' };
    }

    try {
      mediaRecorderRef.current = new MediaRecorder(videoRef.current.srcObject, options);
      mediaRecorderRef.current.ondataavailable = handleDataAvailable;
      mediaRecorderRef.current.start();
      setCapturing(true);
      addTimestamp(currentSessionCountRef.current);

      if (!interval) {
        interval = setInterval(stopRecording, 3000);
      }

      if (currentSessionCountRef.current <= 10) {
        setCurrentSession("collecting");
      } else if (11 <= currentSessionCountRef.current && currentSessionCountRef.current <= 15) {
        setCurrentSession("blackBg");
      } else if (16 <= currentSessionCountRef.current && currentSessionCountRef.current <= 20) {
        setCurrentSession("redBg");
      } else if (21 <= currentSessionCountRef.current && currentSessionCountRef.current <= 25) {
        setCurrentSession("blueBg");
      } else if (26 <= currentSessionCountRef.current && currentSessionCountRef.current <= 30) {
        setCurrentSession("yellowBg");
      } else if (31 <= currentSessionCountRef.current && currentSessionCountRef.current <= 35) {
        setCurrentSession("greenBg");
      } else if (36 <= currentSessionCountRef.current && currentSessionCountRef.current <= 40) {
        setCurrentSession("grayBg");
      } else {
        setCurrentSession("collecting");
      }

    } catch (e) {
      console.error('Failed to create MediaRecorder:', e);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    clearInterval(interval);
    setCapturing(false);
    setCurrentSession("save");
    addTimestamp();
  };

  const saveRecording = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentSessionCountRef.current}.webm`;
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }, 100);

    mediaRecorderRef.current.ondataavailable = null;
    currentSessionCountRef.current += 1;

    // Collecting 50 images
    if (currentSessionCountRef.current <= 50) {
      setCurrentSession("start");
    } else {
      setCurrentSession("finish");
    }

    console.log(timestampArray);
  };

  const renderSession = () => {
    switch (currentSession) {
      case "save":
        return (
          <div className="experiment-div">
            <div>{recordedChunks.length > 0 && <button className="btn" onClick={saveRecording}>Save Recording</button>}</div>
          </div>
        );
      case "start":
        return (
          <div className="experiment-div">
            <button className="btn" onClick={startRecording}>Start Recording</button>
          </div>
        );
      case "collecting":
        return (
          <div className="experiment-div">
            {currentSessionCountRef.current} / 50
          </div>
        );
      case "blackBg":
        return (
          <div className="black-bg-div">
            {currentSessionCountRef.current} / 50
          </div>
        );
      case "redBg":
        return (
          <div className="red-bg-div">
            <div>
              {currentSessionCountRef.current} / 50
            </div>
          </div>
        );
      case "blueBg":
        return (
          <div className="blue-bg-div">
            <div>
              {currentSessionCountRef.current} / 50
            </div>
          </div>
        );
      case "yellowBg":
        return (
          <div className="yellow-bg-div">
            <div>
              {currentSessionCountRef.current} / 50
            </div>
          </div>
        );
      case "greenBg":
        return (
          <div className="green-bg-div">
            <div>
              {currentSessionCountRef.current} / 50
            </div>
          </div>
        );
      case "grayBg":
        return (
          <div className="gray-bg-div">
            <div>
              {currentSessionCountRef.current} / 50
            </div>
          </div>
        );
      case "finish":
        return (
          <div className="next-page-div">
            {downloadCsvEnableStatus ? (
              <div>
                <CSVLink className="btn" data={timestampArray} filename={"timestamp.csv"}>DOWNLOAD CSV</CSVLink>
              </div>
            ) : (
              <div>
                <h1>Thank you for participating. Please inform experiment conductor.</h1>
                <button className="btn" onClick={() => setTimestampArray(chunkArray(timestamps, 3))}>CHUNK DATA</button>
              </div>
            )}
          </div>
        )
      default:
        return (
          <div className="experiment-div">
            <button className="btn" onClick={stopRecording}>Stop Recording</button>
          </div>
        );
    }
  };

  return (
    <div>
      <div className='video-container'>
        <video ref={videoRef} autoPlay playsInline muted />
      </div>
      {renderSession()}
    </div >
  );
};

export default Experiment;
