import React, { useEffect, useState, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { FaPlus, FaRegFileAlt } from "react-icons/fa";
import JobFeedback from "./JobFeedback";
import { createJob, getJobList, getModelList } from "../../../redux/jobSlice.jsx";
import { showToast } from "../../../redux/toastSlice.jsx";
import JobList from "./JobList.jsx";

const POLL_PENDING = parseInt(import.meta.env.VITE_POLL_PENDING) || 15000;
const POLL_PREPROCESSING = parseInt(import.meta.env.VITE_POLL_PREPROCESSING) || 60000;
const POLL_RUNNING = parseInt(import.meta.env.VITE_POLL_RUNNING) || 60000;

const JobDetail = ({ projectId }) => {
    const dispatch = useDispatch();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [creatingJob, setCreatingJob] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null);
    const [autoEstimate, setAutoEstimate] = useState("1");
    const [subtypeCount, setSubtypeCount] = useState(2);
    const pollingRef = useRef(null);

    const { jobList, modelList } = useSelector((state) => state.job);

    const selectedModelName = modelList.find(m => m.id === Number(selectedModel))?.name || "";

    const fetchJobs = async () => {
        setLoading(true);
        try {
            await dispatch(getJobList(projectId)).unwrap();
            setError(null);
        } catch (error) {
            setError(error);
            dispatch(showToast({ message: error, type: "error" }));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        dispatch(getModelList());
    }, [dispatch]);

    useEffect(() => {
        fetchJobs();

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
            }
        };
    }, [dispatch, projectId]);

    useEffect(() => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }

        const hasPendingJobs = jobList.some(job => job.status === "Pending");
        const hasPreprocessingJobs = jobList.some(job => job.status === "Preprocessing");
        const hasRunningJobs = jobList.some(job => job.status === "Running");

        let interval = 0;
        if (hasPendingJobs) {
            interval = POLL_PENDING;
        } else if (hasPreprocessingJobs) {
            interval = POLL_PREPROCESSING;
        } else if (hasRunningJobs) {
            interval = POLL_RUNNING;
        }

        if (interval > 0) {
            pollingRef.current = setInterval(() => {
                dispatch(getJobList(projectId)).unwrap().catch(() => {});
            }, interval);
        }

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
            }
        };
    }, [jobList, dispatch, projectId]);

    const createJobHandler = () => {
        if (!selectedModel) {
            dispatch(showToast({ message: "Please select a model before creating a job.", type: "warning" }));
            return;
        }

        const modelParameters = selectedModelName === "CancerSubminer"
            ? [Number(autoEstimate), Number(subtypeCount)]
            : [];

        const jobData = {
            project_id: projectId,
            model_id: selectedModel,
            model_parameters: modelParameters,
        };

        setCreatingJob(true);

        dispatch(createJob(jobData))
            .unwrap()
            .then(() => {
                dispatch(showToast({ message: "Job created successfully.", type: "success" }));
            })
            .catch((err) => {
                dispatch(showToast({ message: err || "Failed to create job.", type: "error" }));
            })
            .finally(() => {
                setCreatingJob(false);
            });
    };

    return (
        <div className="p-6 bg-base-200 rounded-lg shadow-md space-y-5">
            <div className="flex flex-wrap justify-between items-center gap-4">
                <h2 className="text-xl font-semibold text-base-content flex items-center gap-2">
                    <FaRegFileAlt className="text-primary" />
                    Job Details
                </h2>

                <div className="flex items-center gap-3">
                    <select
                        className="select select-bordered text-base-content w-56"
                        value={selectedModel || ""}
                        onChange={(e) => setSelectedModel(e.target.value)}
                    >
                        <option value="" disabled>Select a Model</option>
                        {modelList.map((model) => (
                            <option key={model.id} value={model.id}>{model.name}</option>
                        ))}
                    </select>

                    <button
                        className="btn btn-outline btn-primary flex items-center"
                        disabled={
                            creatingJob ||
                            !selectedModel ||
                            (selectedModelName === "CancerSubminer" && autoEstimate === "0" && subtypeCount <= 2)
                        }
                        onClick={createJobHandler}
                    >
                        <FaPlus className="mr-2" /> Create Job
                    </button>
                </div>
            </div>

            {selectedModelName === "CancerSubminer" && (
                <div className="flex flex-col gap-4 text-sm text-base-content mt-2">
                    <div>
                        <label className="font-medium mb-1 block">
                            Estimate Number of Subtypes:
                        </label>
                        <div className="flex flex-wrap items-center gap-6">
                            <label className="flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="autoEstimate"
                                    checked={autoEstimate === "1"}
                                    value="1"
                                    onChange={() => setAutoEstimate("1")}
                                    className="radio radio-sm"
                                />
                                Automatically
                            </label>
                            <label className="flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="autoEstimate"
                                    checked={autoEstimate === "0"}
                                    value="0"
                                    onChange={() => setAutoEstimate("0")}
                                    className="radio radio-sm"
                                />
                                Manually specify
                            </label>
                        </div>
                    </div>

                    {autoEstimate === "0" && (
                        <div className="flex flex-col w-48">
                            <label className="font-medium mb-1">
                                Number of Subtypes:
                            </label>
                            <input
                                type="number"
                                className="input input-bordered"
                                value={subtypeCount}
                                min={2}
                                onChange={(e) => setSubtypeCount(Number(e.target.value))}
                            />
                        </div>
                    )}
                </div>
            )}

            <JobFeedback loading={loading} error={error} jobs={jobList} />

            {!loading && !error && jobList.length > 0 && <JobList jobs={jobList} />}
        </div>
    );
};

export default JobDetail;
