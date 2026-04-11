import { useEffect, useRef, useState } from "react";
import { getRooms } from "../../shared/services/roomService";

export default function DevelopmentPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [rooms, setRooms] = useState([]);
  const [roomResults, setRoomResults] = useState({});
  const [frame, setFrame] = useState(null);

  // 1. FETCH ROOMS FROM SERVER
  useEffect(() => {
    const fetchRooms = async () => {
        try {
          const data = await getRooms();
          console.log("DATA GATHERED FOR DEVELOPMENT:", data)
          setRooms(data);
        } catch (err) {
          console.error(err);
        }
      };

    fetchRooms();
  }, []);

  // 2. CAMERA + DETECTION LOOP
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
        console.error("Camera error:", err);
      }
    };

    const captureFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || rooms.length === 0) return;

      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;

        try {
          // SEND FRAME TO ALL ROOMS CONCURRENTLY
          await Promise.all(
            rooms.map((room) => {
              const formData = new FormData();
              formData.append("file", blob, "frame.jpg");
              formData.append("room_id", room._id);

              return fetch("http://localhost:8000/detect", {
                method: "POST",
                body: formData,
              })
                .then((res) => res.json())
                .then((data) => {
                  console.log(`Room ${room.room_name}:`, data);
                
                  setFrame(data.image_url); // show last captured frame from results folder
                  // STORE RESULT PER ROOM
                  setRoomResults((prev) => ({
                    ...prev,
                    [room._id]: data,
                  }));
                })
                .catch((err) => {
                  console.error(`Error for room ${room._id}`, err);
                });
            })
          );
        } catch (error) {
          console.error("Error sending frames:", error);
        }
      }, "image/jpeg", 0.8);
    };

    startCamera();

    intervalRef.current = setInterval(captureFrame, 3000); // every 3 sec

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [rooms]); // rerun when rooms are loaded

  return (
    <div className="w-full h-full flex flex-col items-center gap-6 p-6">
      <h1 className="text-2xl font-bold">Development Page</h1>

      {/* CAMERA */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="rounded-xl shadow-lg w-[400px]"
      />
      <img src={frame} alt="Placeholder" className="rounded-xl shadow-lg w-[400px]" />
      {/* RESULTS PER ROOM */}
      <div className="grid grid-cols-3 gap-4 w-full">
        {rooms.map((room) => {
          const result = roomResults[room._id];

          return (
            <div
              key={room._id}
              className="p-4 bg-gray-100 rounded-xl shadow-md"
            >
              <h2 className="font-bold text-lg">{room.room_name}</h2>

              {result ? (
                <>
                  <p>State: {result.state}</p>
                  <p>People: {result.features?.people_count}</p>
                  <p>
                    Motion:{" "}
                    {result.features?.motion_level?.toFixed(2)}
                  </p>
                  <p>
                    Occupancy:{" "}
                    {result.features?.estimated_occupancy}
                  </p>
                </>
              ) : (
                <p className="text-gray-500">Waiting for data...</p>
              )}
            </div>
          );
        })}
      </div>

      <canvas ref={canvasRef} className="hidden"></canvas>
    </div>
  );
}