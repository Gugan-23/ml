// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './components/Home'; // Assuming you have a Home component
import Predict from './components/Predict'; // Import your Predict component
import Header from './components/Header'; // Your header component

const App = () => {
    return (
        <Router>
            <Header />
            <Switch>
                <Route path="/" exact component={Home} />
                <Route path="/predict" component={Predict} />
            </Switch>
        </Router>
    );
};

export default App;
