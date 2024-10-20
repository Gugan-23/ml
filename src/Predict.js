// src/components/Predict.js
import React, { useState } from 'react';
import axios from 'axios';

const Predict = () => {
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/capture_image'); // Call the capture_image route
            setResult(response.data); // Store the response data in state
            setError(null); // Clear any previous errors
        } catch (err) {
            console.error('Prediction error:', err);
            setError('An error occurred while predicting. Please try again.');
            setResult(null); // Clear the previous result
        }
    };

    return (
        <div>
            <h1>Predict</h1>
            <button onClick={handlePredict}>Predict</button>
            {result && (
                <div>
                    <h2>Prediction Results:</h2>
                    <p>{result.message}</p>
                    {result.image_url && <img src={`http://127.0.0.1:5000/${result.image_url}`} alt="Captured" />}
                    {result.details && (
                        <pre>{JSON.stringify(result.details, null, 2)}</pre>
                    )}
                </div>
            )}
            {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
    );
};

export default Predict;
