const express = require("express");
const router = express.Router();
const authMiddleware = require("../middleware/authMiddleware")
const { addPoints, getPointsByRoom, deleteROIByIndex } = require("../controller/point")

router.post("/add", authMiddleware, addPoints);
router.get("/:roomId", authMiddleware, getPointsByRoom);
router.delete("/:roomId/:roiIndex", authMiddleware, deleteROIByIndex);

module.exports = router;