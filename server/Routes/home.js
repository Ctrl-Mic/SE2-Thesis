const express = require("express");
const HomeRouter = express.Router();
const multer = require("multer");

const upload = multer();


const { getPeopleCount, home, getHealth } = require("../controller/home");

HomeRouter.post("/people-count", upload.single("file"), getPeopleCount);
HomeRouter.get("/", home);
HomeRouter.get("/health", getHealth);

module.exports = HomeRouter;
