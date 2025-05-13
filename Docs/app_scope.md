## 🎯 Demo Deployment Rules (Streamlit + Ngrok)

### 1. User Interface (Frontend Requirements)

* Use **Streamlit** to build the web interface.
* Users must fill in the following information:

  * **Location**: select from a map (map selector) or a dropdown list of addresses.
  * **Area**: input as a number (in square meters).
  * **Number of Rooms**: input as a number.
  * **Legal Status**, **House Direction**, **Year of Construction**: input manually or select from suggestions.
* Input validation:

  * The form **cannot be submitted** if any required fields are missing.
  * Clear error messages must be displayed for missing or invalid inputs.

### 2. Data Submission to Server (API Request Requirements)

* User data must be sent **via API** to the backend server:

  * Data should be structured as a **JSON** object.
  * Validate the inputs before sending (e.g., area must be > 0, construction year must be reasonable).
* A **loading indicator** or "predicting..." status must be shown while waiting for the server’s response.

### 3. Server-side Processing (Backend Requirements)

* The server must:

  * Receive data from the frontend.
  * Run the **pre-trained real estate price prediction model**.
  * Return the predicted result:

    * **Price per square meter** or
    * **Total property price**.

* Server-side validation:

  * Ensure all incoming data is valid.
  * Handle errors gracefully (e.g., if the model crashes or data is invalid, return a clear and informative error message).

### 4. Result Display (Frontend Result Display)

* Display the prediction results clearly to the user:

  * Show both the **price per square meter** and **total price** if needed.
* Display a **chart of the average prices** for surrounding areas:

  * Use `st.bar_chart`, `st.line_chart`, or third-party libraries like Plotly.
  * Include additional information such as minimum, average, and maximum prices in the neighborhood.

### 5. Supporting Tools and Technical Requirements

* **Streamlit**: for creating a simple and interactive web interface.
* **Ngrok**:

  * Use Ngrok to expose the local server to a public URL for demonstration.
  * Ensure the Ngrok link remains active during the entire demo.
* **Machine Learning Model**:

  * The model must be pre-loaded when the server starts (no retraining on every request).
* **Minimum Security Standards**:

  * Do not log sensitive user information (like detailed addresses) to the console or logs.

---

## ✨ Bonus – Extra Points for Demo

* Show a success message like **"Prediction completed successfully!"** after the result is returned.
* Add smart input warnings such as:

  * "Area too small" (e.g., under 10m²)
  * "Unusual construction year" (e.g., year > current year)
* Light UI decoration:

  * Use Streamlit’s layout and components to add a polished look (icons like 🏠, gentle color themes).
