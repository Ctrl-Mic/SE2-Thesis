import { createContext, useContext, useRef, useEffect } from "react";
import { detectFrame } from "../shared/services/roomService.js";
import { getROIPoints } from "../shared/services/pointService.js";

const CameraContext = createContext();

export const useCamera = () => useContext(CameraContext);

export const CameraProvider = ({ children }) => {
  //! This will store all active camera streams
  const streamsRef = useRef({});
  const captureIntervalsRef = useRef({});
  const captureResourcesRef = useRef({});

  useEffect(() => {
    startCamera();
    startFrameCapture();
    getROIPoints();
    initializeRoomCamera();
  }, []);

  //! Starts the camera for a specific room
  const startCamera = async (roomId, deviceId) => {
    try {
      //! if already running, skip
      if (streamsRef.current[roomId]) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId } },
      });

      streamsRef.current[roomId] = stream;

      console.log("STREAM STARTED FOR ROOM:", roomId, deviceId);
      console.log("CAMERA CONTEXT FUNCTIONS:", { startCamera, getStream });
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  //! Unmounts the camera stream for a specific room when user turned off the device
  const stopCamera = (roomId) => {
    const stream = streamsRef.current[roomId];
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      delete streamsRef.current[roomId];
    }

    if (captureIntervalsRef.current[roomId]) {
      clearInterval(captureIntervalsRef.current[roomId]);
      delete captureIntervalsRef.current[roomId];
    }

    if (captureResourcesRef.current[roomId]) {
      delete captureResourcesRef.current[roomId];
    }

    console.log("STREAM AND CAPTURE STOPPED FOR ROOM:", roomId);
  };

  const stopAllCameras = () => {
    // stop all active streams
    Object.keys(streamsRef.current).forEach((roomId) => {
      const stream = streamsRef.current[roomId];
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    });

    // clear all frame capture intervals
    Object.keys(captureIntervalsRef.current).forEach((roomId) => {
      clearInterval(captureIntervalsRef.current[roomId]);
    });

    // clear all stored refs/objects
    streamsRef.current = {};
    captureIntervalsRef.current = {};
    captureResourcesRef.current = {};

    console.log("All camera streams and capture resources cleared");
  };

  //! Get stream for a room
  const getStream = (roomId) => {
    return streamsRef.current[roomId];
  };

  const initializeRoomCamera = async (roomId, rooms, availableCameras) => {
    try {
      const room = rooms.find((r) => r._id === roomId);
      const devices = await getDevice(roomId);

      if (!devices.length) {
        console.log("No saved device for room:", room?.room_name || roomId);
        stopCamera(roomId);
        return;
      }

      const savedDevice = devices[0];

      const matchedCamera = availableCameras.find(
        (cam) => cam.label === savedDevice.device_label,
      );

      if (!matchedCamera) {
        console.warn(
          "No matching browser camera for room:",
          room?.room_name || roomId,
          savedDevice.device_label,
        );
        stopCamera(roomId);
        return;
      }

      stopCamera(roomId);
      await startCamera(roomId, matchedCamera.deviceId);
      startFrameCapture(roomId);
    } catch (err) {
      console.error("Error initializing room camera:", err);
    }
  };

  //! Frame capture
  const startFrameCapture = async (roomId) => {
    const stream = streamsRef.current[roomId];

    if (!stream) {
      console.warn("No stream found for room:", roomId);
      return;
    }

    if (captureIntervalsRef.current[roomId]) {
      console.log("Frame capture already running for room:", roomId);
      return;
    }

    let roiPayload = [];

    try {
      const roiData = await getROIPoints(roomId);

      //! group DB rows into ROI sets
      const groupedROIs = {};

      roiData.forEach((point) => {
        if (!groupedROIs[point.point_index]) {
          groupedROIs[point.point_index] = [];
        }

        groupedROIs[point.point_index].push({
          point_x: point.point_x,
          point_y: point.point_y,
          point_order: point.point_order,
        });
      });

      roiPayload = Object.keys(groupedROIs)
        .sort((a, b) => Number(a) - Number(b))
        .map((key) => ({
          roi_index: Number(key),
          points: groupedROIs[key].sort(
            (a, b) => a.point_order - b.point_order,
          ),
        }));

      console.log("ROI PAYLOAD FOR ROOM:", roomId, roiPayload);
    } catch (error) {
      console.error("Failed to fetch ROI for room:", roomId, error);
      return;
    }

    const video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    captureResourcesRef.current[roomId] = {
      video,
      canvas,
      ctx,
    };

    video
      .play()
      .then(() => {
        captureIntervalsRef.current[roomId] = setInterval(() => {
          if (video.videoWidth === 0 || video.videoHeight === 0) {
            console.warn("Video not ready for room:", roomId);
            return;
          }

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(
            async (blob) => {
              if (!blob) return;

              const formData = new FormData();
              formData.append("file", blob, "frame.jpg");
              formData.append("roomId", String(roomId));
              formData.append("rois", JSON.stringify(roiPayload));

              for (const pair of formData.entries()) {
                console.log("FORMDATA:", pair[0], pair[1]);
              }

              try {
                const data = await detectFrame(formData);
                // console.log("BACKEND RESPONSE:", data);
              } catch (error) {
                console.error("Error sending frame for room:", roomId, error);
              }
            },
            "image/jpeg",
            0.8,
          );
        }, 3000);

        console.log("Frame capture started for room:", roomId);
      })
      .catch((err) => {
        console.error("Error playing hidden video for room:", roomId, err);
      });
  };

  return (
    <CameraContext.Provider
      value={{
        startCamera,
        stopCamera,
        stopAllCameras,
        getStream,
        startFrameCapture,
        initializeRoomCamera,
      }}
    >
      {children}
    </CameraContext.Provider>
  );
};
