import React from "react";
import { useDispatch } from 'react-redux';
import JobStatusBadge from "./JobStatusBadge";
import { formatDate } from "../../../shared/utils/utils.jsx";
import {
    FaRegCalendarAlt, FaPlay, FaStopwatch, FaFileDownload, FaChartBar
} from "react-icons/fa";
import {useNavigate} from "react-router-dom";
import {RouteConstants} from "../../../shared/constants/RouteConstants.js";
import {downloadResults} from "../../../redux/jobSlice.jsx";
import {showToast} from "../../../redux/toastSlice.jsx";

const JobList = ({ jobs }) => {
    const dispatch = useDispatch();
    const navigate = useNavigate();

    return (
        <div className="space-y-6">
            {jobs.map((job) => (
                <div key={job.id} className="p-6 bg-base-100 rounded-lg shadow-md border border-base-300">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-semibold text-base-content flex items-center gap-3">
                            <span className="text-primary">Job ID: {job.id}</span>
                            <span className="text-gray-500 font-medium">| Model: {job.model_name || "Not Specified"}</span>
                        </h3>
                        <div className="flex items-center gap-3">
                            <JobStatusBadge status={job.status}/>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <p className="flex items-center gap-2 text-base-content">
                            <FaRegCalendarAlt className="text-primary"/>
                            <strong>Created:</strong> {formatDate(job.created_at)}
                        </p>
                        <p className="flex items-center gap-2 text-base-content">
                            <FaPlay className="text-green-500"/>
                            <strong>Started:</strong> {job.started_at ? formatDate(job.started_at) : "N/A"}
                        </p>
                        <p className="flex items-center gap-2 text-base-content">
                            <FaStopwatch className="text-yellow-500"/>
                            <strong>Finished:</strong> {job.finished_at ? formatDate(job.finished_at) : "N/A"}
                        </p>
                    </div>

                    {job.status === "Completed" && (
                        <div className="flex justify-end gap-3 mt-5">
                            <button
                                className="btn btn-sm btn-outline btn-success flex items-center"
                                onClick={() =>
                                    dispatch(downloadResults(job.id))
                                        .unwrap()
                                        .catch(err => dispatch(showToast({ message: err, type: "error" })))
                                }
                            >
                                <FaFileDownload className="mr-2"/>
                                Download Result
                            </button>
                            <button
                                className="btn btn-sm btn-outline btn-info flex items-center"
                                onClick={() => navigate(`${RouteConstants.project}/${job.project_id}/${RouteConstants.visualization}/${job.id}/${job.model_name}`)}
                            >
                                <FaChartBar className="mr-2"/>
                                Visualize
                            </button>
                        </div>
                    )}
                </div>
            ))}
        </div>
    )
};

export default JobList;
