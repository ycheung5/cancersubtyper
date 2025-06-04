import React, { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { getProjects } from "../redux/projectSlice.jsx";
import ProjectDetail from "../components/project/ProjectDetail.jsx";
import SampleDetail from "../components/project/SampleDetail.jsx";
import JobDetail from "../components/project/job/JobDetail.jsx";
import { RouteConstants } from "../shared/constants/RouteConstants.js";
import BackToTopButton from "../components/BackToTopButton.jsx";

const Project = () => {
    const { id } = useParams();
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { projectList } = useSelector((state) => state.project);
    const { jobList } = useSelector((state) => state.job);

    const project = projectList.find((project) => project.id === Number(id));

    useEffect(() => {
        dispatch(getProjects());
    }, [dispatch, jobList]);

    return (
        <div className="flex justify-center">
            <div className="card w-full max-w-7xl shadow-lg p-5 bg-base-100 space-y-10 my-5">
                <div className="card-body">
                    {project ? (
                        <div className="space-y-8">
                            <div className="breadcrumbs text-md">
                                <ul>
                                    <li>
                                        <button onClick={() => navigate(RouteConstants.dashboard)} className="cursor-pointer text-primary">
                                            Dashboard
                                        </button>
                                    </li>
                                    <li className="text-base-content">Project</li>
                                </ul>
                            </div>

                            <ProjectDetail project={project} />
                            <SampleDetail projectId={project.id} target={project.target_file} source={project.source_file} />
                            <JobDetail projectId={project.id} />
                        </div>
                    ) : (
                        <p className="text-lg text-gray-500">Project data is not available.</p>
                    )}
                </div>
            </div>
            <BackToTopButton />
        </div>
    );
};

export default Project;
