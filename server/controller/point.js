require("dotenv").config();
const logger = require("../utils/logger");
const { getIO } = require("../config/socket");
const ROIPoint = require("../models/points_model");

exports.addPoints = async (req, res, next) => {
  try {
    const { roomId, rois } = req.body;

    console.dir(req.body, { depth: null });

    if (!roomId) {
      return res.status(400).json({ message: "roomId is required" });
    }

    if (!Array.isArray(rois) || rois.length === 0) {
      return res.status(400).json({ message: "rois is required" });
    }

    const pointsToInsert = [];

    for (const roi of rois) {
      if (!roi.roi_index || !Array.isArray(roi.points)) {
        return res.status(400).json({
          message: "Each ROI must contain roi_index and points array",
        });
      }

      for (const point of roi.points) {
        if (
          typeof point.point_x !== "number" ||
          typeof point.point_y !== "number" ||
          typeof point.point_order !== "number"
        ) {
          return res.status(400).json({
            message: "Invalid point data detected",
          });
        }

        pointsToInsert.push({
          room_id: roomId,
          point_x: point.point_x,
          point_y: point.point_y,
          point_index: roi.roi_index,
          point_order: point.point_order,
        });
      }
    }

    // Optional: remove old ROI points for this room first
    // await ROIPoint.deleteMany({ room_id: roomId });

    const savedPoints = await ROIPoint.insertMany(pointsToInsert);

    logger.info({
      message: `ROI ADD -- ROI points added for room ${roomId}: With status code 201`,
      method: req.method,
      ip: req.ip,
    });

    return res.status(201).json({
      message: "ROI points saved successfully",
      count: savedPoints.length,
      data: savedPoints,
    });
  } catch (error) {
    logger.error({
      message: `ROI ADD -- ${error.message}`,
      method: req.method,
      ip: req.ip,
    });

    return res.status(500).json({ message: error.message });
  }
};

exports.getPointsByRoom = async (req, res) => {
  try {
    const { roomId } = req.params;

    const points = await ROIPoint.find({ room_id: roomId }).sort({
      point_index: 1,
      point_order: 1,
    });

    return res.status(200).json(points);
  } catch (error) {
    return res.status(500).json({ message: error.message });
  }
};

exports.deleteROIByIndex = async (req, res) => {
  try {
    const { roomId, roiIndex } = req.params;

    await ROIPoint.deleteMany({
      room_id: roomId,
      point_index: Number(roiIndex),
    });

    return res.status(200).json({
      message: `ROI ${roiIndex} deleted successfully`,
    });
  } catch (error) {
    console.error("Failed to delete ROI:", error.message);
    return res.status(500).json({ message: error.message });
  }
};
