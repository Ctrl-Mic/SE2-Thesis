import { useEffect, useRef, useState } from "react";

export default function DevelopmentPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [imageUrl, setImageUrl] = useState(null);
  const [features, setFeatures] = useState(null);
  const [state, setState] = useState(null);
  const [belief, setBelief] = useState(null);

  useEffect(() => {
    let stream;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera: ", err);
      }
    };

    const CaptureFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) return;

      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        async (blob) => {
          if (!blob) return;

          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const response = await fetch(
              "http://localhost:8000/detect", 
              {
                method: "POST",
                body: formData,
              }
            );

            const data = await response.json();

            console.log("Detection result:", data);

            //  update UI
            setImageUrl(data.image_url);
            setFeatures(data.features);
            setState(data.state);
            setBelief(data.belief);

          } catch (error) {
            console.error("Error sending frame:", error);
          }
        },
        "image/jpeg",
        0.8
      );
    };

    startCamera();

    intervalRef.current = setInterval(CaptureFrame, 3000); // every 3 sec

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return (
    <div className="w-full h-full flex flex-col justify-center items-center gap-6">
      <div className="w-100 flex flex-row justify-center items-center gap-6">
        {/* Live camera */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="rounded-xl shadow-lg"
        ></video>

        {/* Annotated result */}
        {imageUrl && (
          <img
            src={imageUrl}
            alt="annotated frame"
            className="rounded-xl shadow-lg"
          />
        )}
      </div>

      {/* Feature display */}
      {features && (
        <div className="text-center">
          <p>People: {features.people_count}</p>
          <p>Motion: {features.motion_level.toFixed(2)}</p>
          <p>Exit Activity: {features.exit_activity ? "Yes" : "No"}</p>
          <p>Confidence: {features.avg_confidence.toFixed(2)}</p>
        </div>
      )}

      {belief && (
        <div className="text-center">
          <p>Belief:</p>
          <ul>
            <li>Occupied: {belief.Occupied.toFixed(2)}</li>
            <li>Leaving: {belief.Leaving.toFixed(2)}</li>
            <li>Empty: {belief.Empty.toFixed(2)}</li>
          </ul>
        </div>
      )}
      
      {state && (
        <div className="text-center">
          <p>Inferred State: {state}</p>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden"></canvas>

      <h2 className="primary-text">
        THIS IS A DEVELOPMENT PAGE USED FOR TESTING PURPOSES ONLY hehe
      </h2>
    </div>
  );
}