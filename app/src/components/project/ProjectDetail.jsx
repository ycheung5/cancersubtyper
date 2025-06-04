import React, { useState } from "react";
import { FaEdit, FaTrash, FaFileUpload, FaFileCsv, FaClock, FaCalendarAlt } from "react-icons/fa";
import { MdCheckCircle, MdCancel } from "react-icons/md";
import { useDispatch, useSelector } from "react-redux";
import { showToast } from "../../redux/toastSlice.jsx";
import { deleteProject, getProjects, uploadMetadataFile } from "../../redux/projectSlice.jsx";
import { formatDate } from "../../shared/utils/utils.jsx";
import { useNavigate } from "react-router-dom";
import { RouteConstants } from "../../shared/constants/RouteConstants.js";
import EditProject from "./EditProject.jsx";

const ProjectDetail = ({ project }) => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isEditOpen, setIsEditOpen] = useState(false);
    const used = useSelector((state) => state.job.jobList.length > 0);

    const handleMetadataUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        dispatch(uploadMetadataFile({ project_id: project.id, metadataFile: file }))
            .unwrap()
            .then(() => dispatch(showToast({ message: "Metadata uploaded successfully", type: "success" })))
            .catch((err) => dispatch(showToast({ message: err || "Upload failed", type: "error" })));
    };

    const deleteProjectHandler = () => {
        dispatch(deleteProject(project.id))
            .unwrap()
            .then(async () => {
                await dispatch(getProjects());
                navigate(RouteConstants.dashboard);
                dispatch(showToast({ message: "Project deleted successfully", type: "success" }));
            })
            .catch((err) => {
                dispatch(showToast({ message: err || "Failed to delete project", type: "error" }));
            });
    };

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-base-content truncate max-w-[70%]">{project.name}</h1>
                <div className="flex space-x-3">
                    <button className="btn btn-outline btn-info flex items-center" onClick={() => setIsEditOpen(true)}>
                        <FaEdit className="mr-2" /> Edit
                    </button>
                    <button className="btn btn-outline btn-error flex items-center" onClick={() => setIsModalOpen(true)}>
                        <FaTrash className="mr-2" /> Delete
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Project Info */}
                <div className="p-5 bg-base-200 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold text-base-content mb-3">Project Information</h2>
                    <p className="text-lg font-medium flex items-center gap-2 pb-2">
                        <strong>Tumor Type:</strong>
                        {project.tumor_type ? (
                            <span className="badge badge-accent text-base-content">
                                {project.tumor_type}
                            </span>
                        ) : (
                            <span className="text-gray-400 italic">Unknown</span>
                        )}
                    </p>
                    <p className="text-lg font-medium"><strong>Description:</strong></p>
                    <div className="p-3 bg-base-100 rounded-lg border border-base-300 max-h-40 overflow-y-auto break-words mb-3">
                        {project.description || "No description provided."}
                    </div>

                    <p className="text-lg font-medium mt-2 flex items-center gap-2">
                        <strong>Status:</strong>
                        {project.active ? (
                            <span className="flex items-center gap-1 text-green-500">
                                <MdCheckCircle className="text-xl" /> Active
                            </span>
                        ) : (
                            <span className="flex items-center gap-1 text-red-500">
                                <MdCancel className="text-xl" /> Inactive
                            </span>
                        )}
                    </p>
                </div>

                {/* Timestamps */}
                <div className="p-5 bg-base-200 rounded-lg shadow-md space-y-2">
                    <h2 className="text-xl font-semibold text-base-content mb-3">Timestamps</h2>
                    <p className="text-lg font-medium flex items-center gap-2">
                        <FaCalendarAlt className="text-info" />
                        <strong>Created At:</strong>
                        {formatDate(project.created_at)}
                    </p>
                    <p className="text-lg font-medium flex items-center gap-2">
                        <FaClock className="text-info" />
                        <strong>Last Edited:</strong>
                        {formatDate(project.edited_at)}
                    </p>
                </div>
            </div>

            <div className="p-5 bg-base-200 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold text-base-content mb-3">Metadata File</h2>
                <div className="flex items-center gap-4">
                    <label
                        htmlFor="metadata-upload"
                        className={`btn btn-outline btn-primary flex items-center cursor-pointer ${used ? 'btn-disabled' : ''}`}
                    >
                        <FaFileUpload className="mr-2" />
                        <span>Upload Metadata</span>
                    </label>
                    <input
                        id="metadata-upload"
                        type="file"
                        className="hidden"
                        accept=".csv"
                        onChange={handleMetadataUpload}
                        disabled={used}
                    />

                    {project.metadata_file ? (
                        <div className="flex items-center gap-2 text-green-500">
                            <FaFileCsv className="text-lg" />
                            <span>{project.metadata_file}</span>
                        </div>
                    ) : (
                        <span className="text-error">No metadata file uploaded</span>
                    )}
                </div>
            </div>

            <dialog className="modal" open={isModalOpen}>
                <div className="modal-box bg-base-100 shadow-lg border border-base-300 rounded-lg">
                    <h3 className="font-bold text-xl text-base-content">Confirm Deletion</h3>
                    <p className="text-base-content mt-2">
                        Are you sure you want to delete this project? This action cannot be undone.
                    </p>
                    <div className="modal-action">
                        <button className="btn btn-outline" onClick={() => setIsModalOpen(false)}>
                            Cancel
                        </button>
                        <button className="btn btn-error" onClick={() => {
                            setIsModalOpen(false);
                            deleteProjectHandler();
                        }}>
                            Confirm Delete
                        </button>
                    </div>
                </div>
            </dialog>

            <EditProject isOpen={isEditOpen} onClose={() => setIsEditOpen(false)} project={project} />
        </div>
    );
};

export default ProjectDetail;
