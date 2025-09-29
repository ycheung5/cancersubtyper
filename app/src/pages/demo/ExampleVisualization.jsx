import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import BackToTopButton from "../../components/BackToTopButton.jsx";
import { RouteConstants } from "../../shared/constants/RouteConstants.js";

// Example containers
import BCTyperFinderExample from "../../components/visualization/examples/BCTyperFinderExample.jsx";
// import CancerSubminerExample from "../components/visualization/examples/CancerSubminerExample.jsx";

const ExampleVisualization = () => {
    const navigate = useNavigate();
    const [modelName, setModelName] = useState("BCtypeFinder"); // use state instead of params

    // const handleModelChange = (e) => {
    //     setModelName(e.target.value);
    // };

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
                                Demo Project
                            </a>
                        </li>
                        <li className="text-base-content">Demo Visualization</li>
                    </ul>
                </div>

                {/* Model selector */}
                {/*<div className="flex items-center gap-3">*/}
                {/*    <label className="font-medium">Choose Model:</label>*/}
                {/*    <select*/}
                {/*        className="select select-bordered"*/}
                {/*        value={modelName}*/}
                {/*        onChange={handleModelChange}*/}
                {/*    >*/}
                {/*        <option value="">-- Select --</option>*/}
                {/*        <option value="BCtypeFinder">BCtypeFinder</option>*/}
                {/*        <option value="CancerSubminer">CancerSubminer</option>*/}
                {/*    </select>*/}
                {/*</div>*/}

                {/* Render model-specific example */}
                {modelName === "BCtypeFinder" && <BCTyperFinderExample />}
                {/*{modelName === "CancerSubminer" && <CancerSubminerExample />}*/}
            </div>

            <BackToTopButton />
        </div>
    );
};

export default ExampleVisualization;
