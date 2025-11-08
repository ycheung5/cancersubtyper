import React from "react";
import heroBg from "../../assets/hero-bg.png";
import { useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { RouteConstants } from "../../shared/constants/RouteConstants.js";

const Hero = () => {
    const navigate = useNavigate();
    const { user } = useSelector((state) => state.auth);

    const handleLoginClick = () => navigate(RouteConstants.login);
    const handleDashboardClick = () => navigate(RouteConstants.dashboard);
    const handleDemoClick = () => navigate(RouteConstants.demoDashboard); // NEW

    return (
        <div
            className="hero h-screen flex items-center justify-center"
            style={{
                backgroundImage: `url(${heroBg})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
            }}
        >
            <div className="hero-overlay absolute"></div>

            <div className="hero-content text-center text-neutral-content relative z-10">
                <div className="max-w-3xl">
                    <h1 className="mb-5 text-6xl font-extrabold tracking-tight">
                        Cancer<span className="text-red-400">Subtyper</span>
                    </h1>

                    <p className="mb-6 text-xl font-medium leading-relaxed">
                        A Web-Based <span className="text-red-400 font-bold">Deep Learning</span> Platform for <span className="text-red-400 font-bold">Cancer Subtyping</span> Through DNA Methylation Data
                    </p>

                    <div className="flex justify-center flex-wrap gap-4">
                        {!user ? (
                            <button
                                onClick={handleLoginClick}
                                className="btn btn-primary btn-lg px-6 transition duration-300 transform hover:scale-105"
                            >
                                Get Started
                            </button>
                        ) : (
                            <button
                                onClick={handleDashboardClick}
                                className="btn btn-primary btn-lg px-6 transition duration-300 transform hover:scale-105"
                            >
                                Go to Dashboard
                            </button>
                        )}

                        <button
                            onClick={handleDemoClick}
                            className="btn btn-accent btn-lg px-6 transition duration-300 transform hover:scale-105"
                        >
                            Try Demo
                        </button>

                        <a
                            href="https://www.youtube.com/watch?v=xw05Zq01-EQ"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-info btn-lg px-6 transition duration-300 transform hover:scale-105"
                        >
                            Tutorial
                        </a>

                        <button
                            onClick={() => {
                                document.getElementById("model-structure")?.scrollIntoView({ behavior: "smooth" });
                            }}
                            className="btn btn-outline btn-secondary btn-lg px-6 transition duration-300 transform hover:scale-105"
                        >
                            Learn More
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Hero;
