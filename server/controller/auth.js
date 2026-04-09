require("dotenv").config();
const bcrypt = require("bcrypt");
const User = require("../models/user_model");
const jwt = require("jsonwebtoken");
const logger = require("../utils/logger");

// Okay so many shit is going on here, but ill try to comment as much as possible to help for debugging and stuff
// Gosh this turned into monster code. Okay so imma list down my todo here right now
// TODO: Refactor as much as possible
// TODO: Label functions kay nakakalito minsan

exports.registerUser = async (req, res) => {
  try {
    // Takes the data from frontend
    const { first_name, last_name, email, password, user_organization } =
      req.body;
    // Check if user already exist thru email
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      // Logs the process
      logger.error({
        message: `AUTH REGISTER -- Attempted to register an existing email: With status code 400 with the email of ${req.body.email}`,
        method: req.method,
        ip: req.ip,
      });
      return res
        .status(400)
        .json({
          message:
            "The user with this email already exist. Try logging in to your account.",
        });
    }
    // Hashes the password
    const salt = await bcrypt.genSalt(10);
    const hashed_password = await bcrypt.hash(password, salt);
    // Creates the user data using the user model
    const user = new User({
      first_name,
      last_name,
      email,
      hashed_password,
      user_organization,
    });

    await user.save();
    logger.info({
      message: `AUTH REGISTER -- Registered the user ${req.body.email}: With status code 201`,
      method: req.method,
      ip: req.ip,
    });
    res.status(201).json({
      message: "User registered successfully",
    });
  } catch (error) {
    logger.error({
      message: `AUTH REGISTER -- ${error.message}`,
      method: req.method,
      ip: req.ip,
    });
    res.status(500).json({ message: error.message });
  }
};

exports.loginUser = async (req, res) => {
  try {
    // Basically same format as the register user. It takes the data fro the request first then store it
    // Then with our user object, it'll check the email if it exist in the User table. After that, it'll
    // return a corresponding status code with the status message back to the frontend. If it exist, it
    // will then try to vaidate the password then same cycle. If eveything matches, we then get the user
    // data then using our secret key (basically admin key), we will store the user id as a token for
    // use na through out the website. So this basically prevents constant user request sa database since
    // naka token naman na. And then yeah same send lang status code. Then i added a new function which is
    // yung logger. Self explanatory nayon. Better development practice nalang yon.
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    console.log(user);
    if (!user) {
      logger.error({
        message: `AUTH LOGIN -- User ${user.email} could not be found with status code (400)`,
        method: req.method,
        ip: req.ip,
      });
      return res.status(400).json({
        message:
          "The login information is incorrect. Ensure that you enter correct email and password",
      });
    }

    const isMatch = await bcrypt.compare(password, user.hashed_password);
    if (!isMatch) {
      logger.error({
        message: `AUTH LOGIN -- User ${user.email} password do not match with status code (400)`,
        method: req.method,
        ip: req.ip,
      });
      return res.status(400).json({
        message:
          "The login information is incorrect. Ensure that you enter correct email and password",
      });
    }

    const token = jwt.sign({ userID: user._id }, process.env.JWT_SECRET, {
      expiresIn: "1d",
    });

    logger.info({
      message: `AUTH LOGIN -- Email ${req.body.email} Login Succesfull`,
      method: req.method,
      ip: req.ip,
    });

    res.json({ message: "Login Succesful", token });
  } catch (error) {
    logger.error({
      message: `AUTH LOGIN -- ${error.message} with status code (500)`,
      method: req.method,
      ip: req.ip,
    });
    res.status(500).json({
      message:
        "The login information is incorrect. Ensure that you enter correct email and password",
    });
  }
};

exports.userData = async (req, res, next) => {
  try {
    const user = await User.findById(req.userID).select(
      "first_name last_name is_authorized is_firsttime user_organization is_admin",
    );
    logger.info({
      message: `AUTH USERDATA -- ${user.first_name} data requested from frontend: Accepted`,
      method: req.method,
      ip: req.ip,
    });
    res.json(user);
  } catch (error) {
    res
      .status(500)
      .json({
        message: "It seems like there was a problem connecting to server",
      });
    logger.error({
      message: `AUTH USERDATA -- ${error.message} with status code (500)`,
      method: req.method,
      ip: req.ip,
    });
  }
};

exports.isFirstTime = async (req, res, next) => {
  try {
    const user = await User.findById(req.userID).select(
      "first_name is_firsttime",
    );
    logger.info({
      message: `AUTH FIRSTLOGIN -- ${user.first_name} checked: is_firsttime = ${user.is_firsttime}`,
      method: req.method,
      ip: req.ip,
    });
    res.json(user);
  } catch (error) {
    res
      .status(500)
      .json({
        message: "It seems like there was a problem connecting to server",
      });
    logger.error({
      message: `AUTH FIRSTLOGIN -- ${error.message} with status code (500)`,
      method: req.method,
      ip: req.ip,
    });
  }
};

exports.updateOrganization = async (req, res) => {
  try {
    const { user_organization } = req.body;
    const user = await User.findByIdAndUpdate(
      req.userID,
      { user_organization },
      { new: true }
    ).select("first_name user_organization");

    logger.info({
      message: `AUTH UPDATE ORG -- ${user.first_name} updated organization to ${user_organization}`,
      method: req.method,
      ip: req.ip,
    });

    res.json({ message: "Organization updated successfully", user });
  } catch (error) {
    logger.error({
      message: `AUTH UPDATE ORG -- ${error.message} with status code (500)`,
      method: req.method,
      ip: req.ip,
    });
    res.status(500).json({ message: error.message });
  }
};