import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { AiOutlineProject, AiOutlinePlusCircle } from "react-icons/ai";
import ProjectRow from "./ProjectRow";
import { getProjects } from "../../redux/projectSlice.jsx";
import CreateProjectDialog from "./CreateProject.jsx";

const ProjectTable = () => {
    const dispatch = useDispatch();
    const { projectList, status, error } = useSelector((state) => state.project);
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    useEffect(() => {
        if (status === "idle") {
            dispatch(getProjects());
        }
    }, [dispatch, status]);

    const onCreateProject = () => setIsDialogOpen(true);

    const handleCloseDialog = () => {
        setIsDialogOpen(false);
    };

    const isLoading = status === "loading";
    const isFailed = status === "failed";

    return (
        <div className="bg-base-100 shadow-lg rounded-lg p-6 border border-gray-200 w-7xl">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-semibold text-base-content flex items-center gap-2">
                    <AiOutlineProject className="text-primary text-3xl" />
                    Your Projects
                </h2>
                <button
                    className="btn btn-primary px-6 py-2 font-semibold flex items-center gap-2 shadow-md hover:shadow-lg transition-all"
                    onClick={onCreateProject}
                >
                    <AiOutlinePlusCircle className="text-lg" />
                    Create Project
                </button>
                <CreateProjectDialog isOpen={isDialogOpen} onClose={handleCloseDialog} />
            </div>

            {/* Handle Loading & Error States */}
            {isLoading ? (
                <p className="text-base-content text-center">Loading projects...</p>
            ) : isFailed ? (
                <p className="text-error text-center">Failed to load projects: {error}</p>
            ) : projectList.length === 0 ? (
                <div className="p-6 text-center text-base-content">
                    <p>No projects found. Start by creating one.</p>
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="table w-full border border-gray-200 shadow-sm">
                        <thead className="bg-base-200">
                        <tr>
                            <th className="px-6 py-3 text-left text-base-content w-[20%]">Project Name</th>
                            <th className="px-6 py-3 text-left text-base-content w-[25%]">Tumor Type</th>
                            <th className="px-6 py-3 text-left text-base-content w-[25%]">Description</th>
                            <th className="px-6 py-3 text-left text-base-content w-[15%]">Status</th>
                            <th className="px-6 py-3 text-left text-base-content w-[15%]">Last Edited</th>
                            <th className="px-6 py-3 text-left text-base-content w-[10%]">Actions</th>
                        </tr>
                        </thead>
                        <tbody>
                        {projectList.map((project) => (
                            <ProjectRow key={project.id} project={project} />
                        ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default ProjectTable;
