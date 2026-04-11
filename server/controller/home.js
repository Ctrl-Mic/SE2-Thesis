const axios = require("axios");
const path = require("path");

const imagePath = path.join(__dirname, "../images/sample2.jpg");

// TODO: This need a complete overhaul like wtf is this. Spaghetti ah shi (シ_ _)シ

// !This needs a dedicated controller (I know this is in development, but shit it giving me headaches)
// TODO: Well refactor then move it to its dedicated controller, then add comments on this shit too
const getPeopleCount = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded" });
    }

    const FormData = require("form-data");
    const axios = require("axios");

    console.log("Received file:", req.file);

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: "frame.jpg",
      contentType: req.file.mimetype,
    });

    const response = await axios.post(
      "http://localhost:8000/detect",
      formData,
      {
        headers: formData.getHeaders(),
      },
    );

    res.status(200).json(response.data);
  } catch (error) {
    console.error(
      "Error in getPeopleCount:",
      error.response?.data || error.message,
    );
    res.status(500).json({ error: "Failed to process image" });
  }
};

const detectRoomFrame = async (req, res) => {
  try {
    console.log("===== FRAME DETAILS  =====");

    const { roomId } = req.body;
    const rois = req.body.rois ? JSON.parse(req.body.rois) : [];

    if (!roomId) {
      return res.status(400).json({ error: "roomId is required" });
    }

    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded" });
    }

    console.log(" - Room ID:", roomId);
    console.log(" - Name:", req.file.originalname);
    console.log(" - Type:", req.file.mimetype);
    console.dir(rois, { depth: null });

    return res.status(200).json({
      message: "Frame received successfully",
      roomId,
      rois,
    });
  } catch (error) {
    console.error("Error receiving frame:", error.message);
    return res.status(500).json({ error: "Failed to receive frame" });
  }
};

// ! Well theres nothing much to change in this
const getHealth = async (req, res) => {
  try {
    const response = await axios.get("http://localhost:8000/health");
    res.status(200).json({ status: response.data });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch health status" });
  }
};

module.exports = {
  getPeopleCount,
  detectRoomFrame,
  getHealth,
};
