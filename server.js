const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const bcrypt = require('bcrypt');  // For password hashing
const path = require('path');

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

// MongoDB connection
mongoose.connect('mongodb://127.0.0.1:27017/login', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
});

const db = mongoose.connection;

db.on('error', () => {
    console.log("Could not connect to MongoDB");
});

db.once('open', () => {
    console.log("Connected to MongoDB");
});

// MongoDB Schema and Model
const userSchema = new mongoose.Schema({
    username: String,
    email: { type: String, unique: true },  // Ensure email uniqueness
    password: String,
});
const userModel = mongoose.model('logins', userSchema);

// Endpoint for home page
app.get("/", (req, res) => {
    console.log("Server running on port " + port);
    res.sendFile(path.join(__dirname, 'public', 'homepage2.html'));
});

// Endpoint for user registration
app.post("/registered", async (req, res) => {
    try {
        const { username, email, password } = req.body;

        // Check if the email is already registered
        const existingUser = await userModel.findOne({ email });

        if (existingUser) {
            console.log('Email already registered');
            return res.send('Email already registered');
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create a new user document
        const newUser = new userModel({
            username,
            email,
            password: hashedPassword,  // Store the hashed password
        });

        // Save the user in the database
        await newUser.save();

        console.log("User registered successfully");
        res.redirect('/login.html');  // Redirect to login page
    
    } catch (error) {
        console.log("Error during registration", error);
        res.status(500).send('Internal Server Error');
    }
});

// Endpoint for user login
app.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        // Find the user by email
        const user = await userModel.findOne({ email });

        if (!user) {
            console.log('User not found');
            return res.send('User not found');
        }

        // Compare the provided password with the stored hashed password
        const isPasswordMatch = await bcrypt.compare(password, user.password);

        if (isPasswordMatch) {
            console.log("Login success");
            return res.sendFile(path.join(__dirname, 'public', 'homepage1.html'));
        } else {
            console.log("Password does not match");
            return res.send('Password does not match');
        }

    } catch (error) {
        console.log("Error during login", error);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
