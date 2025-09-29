import React from 'react';
import Hero from '../components/home/Hero.jsx';
import BackToTopButton from "../components/BackToTopButton.jsx";
import ModelStructure from "../components/home/ModelStructure.jsx";

const Home = () => {
    return (
        <div>
            <Hero />
            <ModelStructure />

            <BackToTopButton />
        </div>
    );
};

export default Home;
