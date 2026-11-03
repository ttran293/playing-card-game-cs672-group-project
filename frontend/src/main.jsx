import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const MAX_BYTES = 20 * 1024 * 1024;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [camera, setCamera] = useState(false);
  const [openingCamera, setOpeningCamera] = useState(false);
  const [ready, setReady] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [dragging, setDragging] = useState(false);
  const video = useRef(null);
  const stream = useRef(null);
  const cameraRequest = useRef(0);
  const upload = useRef(null);

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (camera && video.current && stream.current) {
      video.current.srcObject = stream.current;
      video.current.play().catch(() => setError('Could not start camera preview. Please try again.'));
    }
  }, [camera]);

  useEffect(() => () => {
    cameraRequest.current += 1;
    stream.current?.getTracks().forEach(track => track.stop());
  }, []);

  function stopCamera() {
    cameraRequest.current += 1;
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    setCamera(false);
    setOpeningCamera(false);
    setReady(false);
  }

  function chooseFile(next) {
    if (!next) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type)) {
      setError('Choose a JPEG, PNG, or WebP image.'); return;
    }
    if (next.size > MAX_BYTES) { setError('Choose an image smaller than 20 MB.'); return; }
    stopCamera();
    setFile(next);
    setResult(null);
    setShowOriginal(false);
    setError('');
  }

  async function openCamera() {
    setError('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access needs localhost or HTTPS and a supported browser. You can upload a photo instead.'); return;
    }
    const requestId = ++cameraRequest.current;
    setOpeningCamera(true);
    try {
      const next = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
      if (cameraRequest.current !== requestId) { next.getTracks().forEach(track => track.stop()); return; }
      stream.current = next;
      setCamera(true);
      setReady(false);
    } catch (err) {
      if (cameraRequest.current === requestId) setError(err.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access in your browser or upload a photo.'
        : 'Could not open a camera. Check that one is connected and available.');
    } finally {
      if (cameraRequest.current === requestId) setOpeningCamera(false);
    }
  }

  function capture() {
    const element = video.current;
    if (!element?.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = element.videoWidth;
    canvas.height = element.videoHeight;
    canvas.getContext('2d').drawImage(element, 0, 0);
    canvas.toBlob(blob => {
      if (blob) chooseFile(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
      else setError('Could not capture the photo. Please try again.');
    }, 'image/jpeg', 0.95);
  }

  async function detect() {
    if (!file || busy) return;
    setBusy(true);
    setError('');
    setResult(null);
    setShowOriginal(false);
    try {
      const response = await fetch('/api/detect', {
        method: 'POST', headers: { 'Content-Type': file.type }, body: file,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(data?.detail || 'Detection failed. Check that the backend is running.');
      if (!data?.annotated_image || !Array.isArray(data.cards)) throw new Error('The backend returned an unexpected response.');
      setResult(data);
    } catch (err) {
      setError(err.message === 'Failed to fetch' ? 'Cannot reach the backend. Check that FastAPI is running on port 8000.' : err.message);
    } finally { setBusy(false); }
  }

  return (
    <main>
      <h1>Card detection</h1>
      <div className="controls">
        <input ref={upload} type="file" accept="image/jpeg,image/png,image/webp" hidden
          onChange={event => { chooseFile(event.target.files[0]); event.target.value = ''; }} />
        <button disabled={busy || openingCamera} onClick={() => upload.current.click()}>Upload image</button>
        {!camera && !openingCamera && <button disabled={busy} onClick={openCamera}>Open camera</button>}
        {openingCamera && <button onClick={stopCamera}>Cancel camera request</button>}
        {camera && <>
          <button disabled={!ready} onClick={capture}>Take photo</button>
          <button onClick={stopCamera}>Close camera</button>
        </>}
        <button disabled={!file || busy || camera || openingCamera} onClick={detect}>
          {busy ? 'Detecting…' : 'Detect'}
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {busy && <p role="status">Detecting cards. The first run may take a few minutes.</p>}
      <div className={`preview ${dragging ? 'dragging' : ''}`}
        onDragOver={event => { event.preventDefault(); if (!busy && !camera) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={event => { event.preventDefault(); setDragging(false); if (!busy && !camera) chooseFile(event.dataTransfer.files[0]); }}>
        {camera ? (
          <video ref={video} autoPlay muted playsInline onLoadedData={() => setReady(true)} aria-label="Live camera preview" />
        ) : preview ? (
          <img src={result && !showOriginal ? result.annotated_image : preview}
            alt={result && !showOriginal ? 'Card detection result' : 'Selected image'}
            onError={() => setError('Could not display this image. Choose another photo.')} />
        ) : <p>Upload an image or take a photo.</p>}
      </div>
      {file && !camera && <p className="filename">{file.name}</p>}
      {result && !camera && <section aria-label="Detection results">
        <button onClick={() => setShowOriginal(!showOriginal)}>{showOriginal ? 'Show detections' : 'Show original'}</button>
        <p>Game: {result.game} · Cards: {result.cards.length} · Clusters: {result.clusters.length}</p>
        {result.clusters.map(cluster => <p key={cluster.id}>Cluster {cluster.id}: {cluster.cards.join(', ')}</p>)}
      </section>}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
