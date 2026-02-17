import { useEffect, useRef } from "react";

export default function DevelopmentPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

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
      const imageData = canvas.toDataURL("image/png");

      canvas.toBlob(
        async (blob) => {
          if (!blob) return;

          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const response = await fetch(
              "http://localhost:4000/Thesis/home/people-count",
              {
                method: "POST",
                body: formData,
              },
            );
            const data = await response.json();
            console.log("YOLO people count:", data);
          } catch (error) {
            console.error("Error sending frame:", error);
          }
        },
        "image/jpeg",
        0.8,
      );
    };

    startCamera();

    intervalRef.current = setInterval(CaptureFrame, 5000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return (
    <div className="w-full h-full flex justify-center items-center">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="rounded-xl shadow-lg"
      ></video>

      <canvas ref={canvasRef} className="hidden"></canvas>
    </div>
  );
}
