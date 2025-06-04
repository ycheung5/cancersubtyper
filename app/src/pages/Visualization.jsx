import React from "react";
import BCTyperFinderVisualization from "../components/visualization/bctypefinder/BCTyperFinderVisualization.jsx";
import { useParams } from "react-router-dom";
import { RouteConstants } from "../shared/constants/RouteConstants";
import { useNavigate } from "react-router-dom";
import BackToTopButton from "../components/BackToTopButton";
import CancerSubminerVisualization from "../components/visualization/cancersubminer/CancerSubminerVisualization.jsx"; // Import

const Visualization = () => {
    const { project_id, job_id, model_name} = useParams();
    const navigate = useNavigate();


    return (
        <div className="flex justify-center">
            <div className="w-full max-w-7xl p-6 bg-base-100 rounded-lg shadow-lg space-y-6 my-5">
                <div className="breadcrumbs text-sm">
                    <ul>
                        <li>
                            <a onClick={() => navigate(RouteConstants.dashboard)} className="cursor-pointer text-primary">
                                Dashboard
                            </a>
                        </li>
                        <li>
                            <a onClick={() => navigate(`${RouteConstants.project}/${project_id}`)} className="cursor-pointer text-primary">
                                Project
                            </a>
                        </li>
                        <li className="text-base-content">Visualization</li>
                    </ul>
                </div>
                {
                    model_name === "BCtypeFinder" && <BCTyperFinderVisualization id={job_id} />
                }
                {
                    model_name === "CancerSubminer" && <CancerSubminerVisualization id={job_id} />
                }
            </div>

            <BackToTopButton />
        </div>
    );
};

export default Visualization;
