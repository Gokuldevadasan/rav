"""Single-page browser UI for the local phone camera prototype."""

CAMERA_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#10151f">
  <title>RAV — Reasoning-Aware Vision</title>
  <style>
    :root { color-scheme: dark; font: 16px/1.5 system-ui, sans-serif; }
    body { margin: 0; background: #10151f; color: #edf2fa; }
    main {
      box-sizing: border-box; width: min(100% - 2rem, 760px); margin: 0 auto;
      padding: 1.25rem 0 3rem;
    }
    header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    h1 { margin: 0; font-size: 1.45rem; }
    h1 span { color: #98a9bf; font-size: .82rem; font-weight: 500; }
    #connection { color: #b9c9dc; font-size: .85rem; }
    video {
      display: block; width: 100%; max-height: 62vh; margin-top: 1rem;
      background: #070a10; border: 1px solid #303c4d; border-radius: 14px;
      object-fit: contain;
    }
    button, input {
      box-sizing: border-box; min-height: 48px; border: 1px solid #44536a;
      border-radius: 10px; font: inherit;
    }
    button { padding: .65rem 1rem; background: #263b59; color: white; cursor: pointer; }
    button:disabled { cursor: wait; opacity: .65; }
    #start-camera { margin-top: .8rem; }
    label { display: block; margin: 1.2rem 0 .35rem; font-weight: 600; }
    input { width: 100%; padding: .65rem .8rem; background: #171f2c; color: white; }
    #ask {
      width: 100%; margin-top: .65rem; background: #4176c5;
      border-color: #4176c5; font-weight: 700;
    }
    #response {
      min-height: 4rem; margin-top: .8rem; padding: .85rem 1rem;
      white-space: pre-wrap; overflow-wrap: anywhere; background: #171f2c;
      border: 1px solid #303c4d; border-radius: 10px;
    }
    #loading { min-height: 1.5rem; margin-top: .35rem; color: #b9c9dc; }
    .hint { color: #aab8ca; font-size: .85rem; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>RAV <span>— Reasoning-Aware Vision</span></h1>
      <div id="connection" role="status">Checking connection…</div>
    </header>
    <video id="preview" autoplay playsinline muted aria-label="Live camera preview"></video>
    <button id="start-camera" type="button">Start camera</button>
    <p id="camera-hint" class="hint"></p>
    <label for="question">Question</label>
    <input id="question" maxlength="500" value="What is around me?" autocomplete="off">
    <button id="ask" type="button" disabled>Capture frame &amp; ask RAV</button>
    <div id="loading" role="status" aria-live="polite"></div>
    <section aria-label="RAV response">
      <strong>Response</strong>
      <div id="response">Start the camera, then ask a question about the current frame.</div>
    </section>
  </main>
  <script>
    const preview = document.getElementById("preview");
    const connection = document.getElementById("connection");
    const cameraHint = document.getElementById("camera-hint");
    const startButton = document.getElementById("start-camera");
    const askButton = document.getElementById("ask");
    const question = document.getElementById("question");
    const response = document.getElementById("response");
    const loading = document.getElementById("loading");
    let cameraStream;

    async function checkConnection() {
      try {
        const result = await fetch("/health");
        if (!result.ok) throw new Error("Server returned " + result.status);
        connection.textContent = "Connected to local RAV";
      } catch (error) {
        connection.textContent = "RAV server unavailable";
      }
    }

    startButton.addEventListener("click", async () => {
      if (!window.isSecureContext) {
        cameraHint.textContent =
          "Phone browsers require HTTPS on the local network. Open the HTTPS " +
          "address with a certificate trusted by this phone.";
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraHint.textContent = "This browser does not provide camera access.";
        return;
      }
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } }, audio: false
        });
        preview.srcObject = cameraStream;
        askButton.disabled = false;
        cameraHint.textContent = "Camera ready. Frames are sent only when you press the button.";
      } catch (error) {
        cameraHint.textContent = "Camera could not be opened: " + error.message;
      }
    });

    askButton.addEventListener("click", async () => {
      if (!cameraStream || !preview.videoWidth) {
        response.textContent = "Camera is not ready. Start the camera and try again.";
        return;
      }
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 1024 / Math.max(preview.videoWidth, preview.videoHeight));
      canvas.width = Math.round(preview.videoWidth * scale);
      canvas.height = Math.round(preview.videoHeight * scale);
      canvas.getContext("2d").drawImage(preview, 0, 0, canvas.width, canvas.height);
      askButton.disabled = true;
      loading.textContent = "Analyzing this frame…";
      response.textContent = "";
      try {
        const frame = await new Promise((resolve, reject) =>
          canvas.toBlob(
            blob => blob ? resolve(blob) : reject(new Error("Could not capture camera frame.")),
            "image/jpeg",
            0.88
          )
        );
        const url = "/analyze?prompt=" + encodeURIComponent(question.value.trim());
        const result = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: frame
        });
        const data = await result.json();
        if (!result.ok) throw new Error(data.detail || "Analysis failed.");
        response.textContent = data.response;
      } catch (error) {
        response.textContent = "RAV could not analyze this frame: " + error.message;
      } finally {
        loading.textContent = "";
        askButton.disabled = false;
      }
    });

    window.addEventListener("beforeunload", () => {
      if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    });
    checkConnection();
  </script>
</body>
</html>
"""
