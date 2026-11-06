# FairTransCP Interactive Web Application ⚖️
**Target Venue:** AISTATS 2027  
**Author:** Aarush Dubey (B.Tech 3rd Year)  
**Paper:** *Fair Conformal Classification for Financial Transactions: Balancing Coverage and Set-Size Equity Across Demographic Groups*

---

## 🚀 Quick Start: Running Locally

To run the interactive web application on your local machine:

1. **Navigate to the repository root:**
   ```bash
   cd ~/Downloads/AISTAT2027/AISTAT2027/fair-conformal-finance
   ```

2. **Activate the environment:**
   ```bash
   source venv_mac/bin/activate
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run webapp/app.py
   ```
   The application will automatically open in your browser at `http://localhost:8501`.

---

## ☁️ Step-by-Step Deployment on Render

You can host this web application for **free** on Render with a public URL to share with your professor or link in your submission.

### Step 1: Sign Up / Log In to Render
- Visit [render.com](https://render.com) and log in using your GitHub account (`aarushdubey`).

### Step 2: Create a New Web Service
1. Click the **"New +"** button in the top right and select **"Web Service"**.
2. Under "Connect a repository", search for and select **`fair-conformal-finance`**.
3. Fill in the service configuration:
   - **Name:** `fairtranscp-demo` (or any name you choose)
   - **Region:** Any (e.g. *Oregon (US West)* or *Frankfurt (EU)*)
   - **Branch:** `main`
   - **Root Directory:** Leave blank (or `fair-conformal-finance` if deploying from a parent mono-repo)
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r webapp/requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run webapp/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Plan:** Select the **Free** tier ($0/month).

4. Click **"Deploy Web Service"**!
5. Within 2-3 minutes, Render will build and provide a public URL like:
   `https://fairtranscp-demo.onrender.com`

> **Note on Render Free Tier:** Free services spin down after 15 minutes of inactivity. When presenting to your professor, open the link 1 minute beforehand so it wakes up, or keep the local version running as a fail-safe backup.

---

## ⚡ Alternative Free 1-Click Hosting: Streamlit Community Cloud
Because this is a pure Streamlit app, you can also deploy it directly to **Streamlit Community Cloud** (which doesn't spin down as aggressively):
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **"New App"**
3. Select Repository: `aarushdubey/fair-conformal-finance`
4. Main file path: `webapp/app.py`
5. Click **"Deploy!"**

---

## 🎯 How to Use This Web App with Your Professor

When you sit down with your professor, follow this 4-step walkthrough:

1. **Tab 1 ("The Paradox & Concept"):**
   - Explain what conformal prediction is: replacing single uncertain predictions with guaranteed sets.
   - Show the 3 colored cards. Point to the **orange card (Mondrian CP)** and explain the **Coverage-Equity Paradox**.
   - Show Proposition 1: explain that when a group's quantile is higher, the ambiguous range $[1-q, q]$ mathematically expands.

2. **Tab 2 ("Live Applicant Simulator"):**
   - Pick an applicant from the dropdown (e.g. Applicant #4, young borrower).
   - Show the **Mondrian CP output: `{Bad, Good}`**. Say: *"Professor, look — the model gave up and outputted both classes. In a retail bank under the FCRA, this borrower is stuck in manual review limbo."*
   - Show **FairTransCP output: `{Good}`**. Say: *"Our method resolves the ambiguity without violating coverage safety."*

3. **Tab 3 ("Empirical Benchmarks"):**
   - Switch between German Credit, Taiwan Credit, and Adult Income.
   - Point out the Adult Income XGBoost finding: Mondrian CP set-size disparity spikes to **1.637** (+57.4% surge), while FairTransCP keeps it controlled at **1.040**.

4. **Tab 4 ("The $\lambda$ Pareto Dial"):**
   - Drag the $\lambda$ slider from 0 to 1 live.
   - Show how it moves along the Pareto frontier, balancing coverage equity and set-size parity.
   - Point to **Theorem 1** for the formal safety lower bound.

5. **Tab 5 ("Defense Cheatsheet"):**
   - Contains direct answers to tough oral questions your professor might ask (why no deep learning, why not just Mondrian, etc.).

