const dataService = require("../services/dataService")



// Non-Hive Users

const express = require('express');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { createHiveAccount, generateHiveKeys } = require('../utils/hiveService');
const { encrypt } = require('..utils/encryption'); // import encryption utility.

const router = express.Router();
const USER_DATA_FILE = path.join(__dirname, '../UserData.json');

// -- Helper Functions -- //

/**
 * Generates a secure, random token.
 * @ returns {string} - The generated token.
 */

function generateToken(){
  return crypto.randomBytes(32).toString('hex'); // 64 character hex string
}

/**
 * Sends a magic link email (Resebd API Integration)
 * @param {string} email - The recipient's email address.
 * @param {string} magicLink - The magic link URL.
 */

async function sendMagicLinkEmail(email, magicLink){
  console.log('\n--- SIMULATING EMAIL SEND ---');
  console.log(`To: ${email}`);
  console.log(`Subject: Your Slakenet Magic Link`);
  console.log(`Body: Click this link to log in: ${mgaicLink}`);
  console.log('----------------------------------------');
// --- RESEND API INTEGRATION EXAMPLE (UNCOMMENT AND CONFIGURE) ---
    /*
    const { Resend } = require('resend');
    const resend = new Resend('YOUR_RESEND_API_KEY'); // Replace with your actual Resend API Key

    try {
        const { data, error } = await resend.emails.send({
            from: 'Slakenet <onboarding@yourdomain.com>', // Replace with your verified sender domain
            to: [email],
            subject: 'Your Slakenet Magic Link',
            html: `<p>Click this link to log in to Slakenet:</p><p><a href="${magicLink}">${magicLink}</a></p>`,
        });

        if (error) {
            console.error('Resend API error:', error);
            return false;
        }
        console.log('Resend API success:', data);
        return true;
    } catch (error) {
        console.error('Error sending email via Resend:', error);
        return false;
    }
    */

    return true; // Simulate success if Resend block is commented out
}

// --- Authentication Routes ---

/**
 * POST /auth/send-magic-link
 * Initiates the magic link login/signup process.
 */
router.post('/send-magic-link', async (req, res) => {
    const { email } = req.body;

    if (!email) {
        return res.status(400).json({ success: false, message: 'Email is required.' });
    }

    try {
        let userData = JSON.parse(fs.readFileSync(USER_DATA_FILE, 'utf8'));
        let user = userData.users.find(u => u.email === email);

        const token = generateToken();
        const tokenExpiry = Date.now() + (15 * 60 * 1000); // Token valid for 15 minutes

        if (user) {
            // Existing user, update token
            user.magicLinkToken = token;
            user.tokenExpiry = tokenExpiry;
            console.log(`Updated magic link token for existing user: ${email}`);
        } else {
            // New user, create a temporary entry. Hive account will be created upon verification.
            user = {
                id: crypto.randomUUID(), // Generate a unique ID for new users
                email: email,
                magicLinkToken: token,
                tokenExpiry: tokenExpiry,
                slakeBalance: 0, // Placeholder
                createdAt: new Date().toISOString(),
                lastLoginAt: null,
                hiveUsername: null,
                hivePublicKeys: null, // Store public keys
                encryptedHiveKeys: null // Store encrypted private keys
            };
            userData.users.push(user);
            console.log(`Created temporary entry for new user: ${email}`);
        }

        fs.writeFileSync(USER_DATA_FILE, JSON.stringify(userData, null, 2));

        const magicLink = `${req.protocol}://${req.get('host')}/auth/verify-magic-link?token=${token}&email=${encodeURIComponent(email)}`;
        const emailSent = await sendMagicLinkEmail(email, magicLink);

        if (emailSent) {
            res.json({ success: true, message: 'Magic link sent to your email.' });
        } else {
            res.status(500).json({ success: false, message: 'Failed to send magic link email.' });
        }

    } catch (err) {
        console.error('Error sending magic link:', err);
        res.status(500).json({ success: false, message: 'Failed to send magic link.' });
    }
});

/**
 * GET /auth/verify-magic-link
 * Verifies the magic link and logs in/signs up the user.
 */
router.get('/verify-magic-link', async (req, res) => {
    const { token, email } = req.query;

    if (!token || !email) {
        return res.redirect('/auth/login-failure?message=Invalid link.');
    }

    try {
        let userData = JSON.parse(fs.readFileSync(USER_DATA_FILE, 'utf8'));
        let userIndex = userData.users.findIndex(u => u.email === email && u.magicLinkToken === token);

        if (userIndex === -1) {
            return res.redirect('/auth/login-failure?message=Invalid or expired link.');
        }

        let user = userData.users[userIndex];

        // Check token expiry
        if (Date.now() > user.tokenExpiry) {
            // Invalidate token after expiry check
            user.magicLinkToken = null;
            user.tokenExpiry = null;
            fs.writeFileSync(USER_DATA_FILE, JSON.stringify(userData, null, 2));
            return res.redirect('/auth/login-failure?message=Magic link expired.');
        }

        // Invalidate the token immediately after successful use
        user.magicLinkToken = null;
        user.tokenExpiry = null;

        // Check if Hive account needs to be created (for new users)
        if (!user.hiveUsername) {
            console.log(`Creating Hive account for new user: ${user.email}`);

            // Generate a unique Hive username
            const baseHiveUsername = user.email.split('@')[0].replace(/[^a-z0-9-]/g, '').toLowerCase();
            let hiveUsername = `slakenet-${baseHiveUsername.substring(0, Math.min(baseHiveUsername.length, 8))}`; // Max 16 chars for Hive username
            let counter = 0;
            // Ensure uniqueness (simple check, improve for production by querying Hive blockchain)
            while (userData.users.some(u => u.hiveUsername === hiveUsername)) {
                counter++;
                hiveUsername = `slakenet-${baseHiveUsername.substring(0, Math.min(baseHiveUsername.length, 6))}${counter}`;
            }

            // Generate Hive keypairs
            const hiveKeys = generateHiveKeys();
            console.log(`Generated Hive keys for ${hiveUsername}.`);

            // Encrypt private keys for storage
            const encryptedOwner = encrypt(hiveKeys.privateKeys.owner);
            const encryptedActive = encrypt(hiveKeys.privateKeys.active);
            const encryptedPosting = encrypt(hiveKeys.privateKeys.posting);
            const encryptedMemo = encrypt(hiveKeys.privateKeys.memo);

            // Store public keys and encrypted private keys in userData.json
            user.hivePublicKeys = hiveKeys.publicKeys;
            user.encryptedHiveKeys = {
                owner: encryptedOwner,
                active: encryptedActive,
                posting: encryptedPosting,
                memo: encryptedMemo
            };

            // Attempt to create Hive account (simulated or via API)
            const hiveAccountCreationStatus = await createHiveAccount(hiveUsername, hiveKeys.publicKeys);
            console.log(`Hive account creation status for ${hiveUsername}: ${hiveAccountCreationStatus}`);

            user.hiveUsername = hiveUsername;
            user.slakeBalance = 0; // Confirm initial Slake balance
            console.log(`Hive account ${user.hiveUsername} created for ${user.email}.`);
        }

        user.lastLoginAt = new Date().toISOString();
        fs.writeFileSync(USER_DATA_FILE, JSON.stringify(userData, null, 2));

        // Set user in session (only store non-sensitive info)
        req.session.user = {
            id: user.id,
            email: user.email,
            hiveUsername: user.hiveUsername,
            slakeBalance: user.slakeBalance
        };

        console.log(`User ${user.email} successfully authenticated and session set.`);

        res.redirect('/dashboard'); // Redirect to the dashboard after successful login/signup

    } catch (err) {
        console.error('Error verifying magic link:', err);
        res.redirect('/auth/login-failure?message=An unexpected error occurred.');
    }
});

// Route for successful signup/login
// This route is now mostly handled by /dashboard
router.get('/signup-success', (req, res) => {
    // This page is now largely deprecated as we redirect to dashboard
    res.redirect('/dashboard');
});

// Route for login failure (remains same)
router.get('/login-failure', (req, res) => {
    const message = req.query.message || 'There was an issue with your login.';
    res.send(`
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Login Failed</title>
            <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>
            <link href="[https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap)" rel="stylesheet">
            <style>
                body { font-family: 'Inter', sans-serif; }
            </style>
        </head>
        <body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
            <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-md w-full">
                <h1 class="text-3xl font-bold text-red-600 mb-4 rounded-md">Login Failed 😔</h1>
                <p class="text-gray-700 mb-6">${message}</p>
                <a href="/" class="inline-block bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300 ease-in-out shadow-md">
                    Try Again
                </a>
            </div>
        </body>
        </html>
    `);
});

module.exports = router;

// End of Non-Hive Users Code



// Get signup page
exports.getSignup = (req, res) => {
  const step = req.query.step || 1
  const userData = req.session.signupData || {}

  res.render("auth/signup", {
    title: "Sign Up | Slakenet Aurix",
    step: Number.parseInt(step),
    userData: userData,
  })
}

// Process signup step one
exports.postSignupStepOne = (req, res) => {
  const { username, email, password, authType } = req.body

  // Validate inputs (simplified validation)
  if (authType === "email" && (!username || !email || !password)) {
    req.flash("error", "Please fill in all required fields")
    return res.redirect("/auth/signup")
  }

  // Save to session
  req.session.signupData = {
    ...req.session.signupData,
    username,
    email,
    password,
    authType: authType || "email",
  }

  res.redirect("/auth/signup?step=2")
}



// Process signup step two
exports.postSignupStepTwo = (req, res) => {
  const { displayName, bio } = req.body

  // Save to session
  req.session.signupData = {
    ...req.session.signupData,
    displayName,
    bio,
  }

  // Note: In a real app, you would handle avatar upload here

  res.redirect("/auth/signup?step=3")
}

// Process signup step three
exports.postSignupStepThree = (req, res) => {
  let interests = req.body.interests

  // Convert to array if it's a single value
  if (!Array.isArray(interests)) {
    interests = interests ? [interests] : []
  }

  // Validate
  if (interests.length < 3) {
    req.flash("error", "Please select at least 3 interests")
    return res.redirect("/auth/signup?step=3")
  }

  // Save to session
  req.session.signupData = {
    ...req.session.signupData,
    interests,
  }

  res.redirect("/auth/signup?step=4")
}

// Complete signup
exports.completeSignup = (req, res) => {
  const userData = req.session.signupData

  if (!userData || !userData.username) {
    return res.redirect("/auth/signup")
  }

  // Create user
  const newUser = {
    id: Date.now().toString(),
    username: userData.username,
    email: userData.email,
    displayName: userData.displayName || userData.username,
    bio: userData.bio || "",
    interests: userData.interests || [],
    avatar: userData.avatar || "",
    authType: userData.authType || "email",
    createdAt: new Date().toISOString(),
  }

  // Save user to database (or in-memory store in this case)
  dataService.createUser(newUser)

  // Set user session
  req.session.user = newUser

  // Clear signup data
  delete req.session.signupData

  req.flash("success", "Welcome to Slakenet Aurix!")
  res.redirect("/dashboard")
}

// Verify keychain
exports.keychainVerify = (req, res) => {
  const { username } = req.body

  // In a real app, you would verify the user's Hive keychain
  // For demo, we'll just accept the username

  req.session.signupData = {
    ...req.session.signupData,
    username,
    authType: "keychain",
    keychainVerified: true,
  }

  res.json({ success: true })
}

// Logout
exports.logout = (req, res) => {
  req.session.destroy()
  res.redirect("/")
}
