const express = require("express");
const router = express.Router();
const authMiddleware = require("../middleware/authMiddleware")
const { registerUser, loginUser, userData, isFirstTime, updateOrganization } = require("../controller/auth");

router.post("/register", registerUser);
router.post("/login", loginUser);
router.get("/me", authMiddleware, userData);
router.get("/first-login", authMiddleware, isFirstTime);
router.patch("/update-organization", authMiddleware, updateOrganization);

module.exports = router;