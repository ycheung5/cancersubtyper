import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import BackToTopButton from "../../components/BackToTopButton.jsx";
import { RouteConstants } from "../../shared/constants/RouteConstants.js";

// Example containers
import BCTyperFinderExample from "../../components/visualization/examples/BCTyperFinderExample.jsx";
import CancerSubminerExample from "../../components/visualization/examples/CancerSubminerExample.jsx";

const ExampleVisualization = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const modelName = location.state?.modelName || "BCtypeFinder";

    return (
        <div className="flex justify-center">
            <div className="w-full max-w-7xl p-6 bg-base-100 rounded-lg shadow-lg space-y-6 my-5">
                <div className="breadcrumbs text-sm">
                    <ul>
                        <li>
                            <a
                                onClick={() => navigate(RouteConstants.demoDashboard)}
                                className="cursor-pointer text-primary"
                            >
                                Demo Dashboard
                            </a>
                        </li>
                        <li>
                            <a
                                onClick={() => navigate(RouteConstants.demoProject)}
                                className="cursor-pointer text-primary"
                            >
                                Breast Cancer Subtyping (DEMO)
                            </a>
                        </li>
                        <li className="text-base-content">Breast Cancer Subtyping (DEMO) Visualization</li>
                    </ul>
                </div>

                {/* Render model-specific example */}
                {modelName === "BCtypeFinder" && <BCTyperFinderExample />}
                {modelName === "CancerSubminer" && <CancerSubminerExample />}
            </div>

            <BackToTopButton />
        </div>
    );
};

export default ExampleVisualization;
