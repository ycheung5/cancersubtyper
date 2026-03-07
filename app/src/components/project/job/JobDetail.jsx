import React, { useEffect, useMemo, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { FaPlus, FaRegFileAlt } from "react-icons/fa";
import { FaCircleInfo } from "react-icons/fa6";
import JobFeedback from "./JobFeedback";
import { createJob, getJobList, getModelList } from "../../../redux/jobSlice.jsx";
import { showToast } from "../../../redux/toastSlice.jsx";
import JobList from "./JobList.jsx";

const POLL_PENDING = parseInt(import.meta.env.VITE_POLL_PENDING) || 15000;
const POLL_PREPROCESSING = parseInt(import.meta.env.VITE_POLL_PREPROCESSING) || 60000;
const POLL_RUNNING = parseInt(import.meta.env.VITE_POLL_RUNNING) || 60000;

const createField = (key, min, max, step, defaultValue, description, label) => ({
    key,
    label,
    min,
    max,
    step,
    defaultValue,
    description,
});

const COMMON_FIELDS = [
    createField("num_cpg_clusters", 100, 10000, 1, 3000, "How many CpG feature groups are created during preprocessing. Larger values keep more detail but usually increase runtime.", "Number of CpG Clusters"),
    createField("batch_size", 1, 1024, 1, 128, "How many source samples are processed together in one training step.", "Batch Size"),
    createField("target_batch_size", 1, 1024, 1, 128, "How many target samples are processed together in one step.", "Target Batch Size"),
    createField("num_hidden_nodes1_feature_extractor", 8, 8192, 1, 1024, "Size of the first hidden layer in the feature extractor network.", "Feature Extractor Hidden Layer 1"),
    createField("num_hidden_nodes2_feature_extractor", 8, 8192, 1, 512, "Size of the second hidden layer in the feature extractor network.", "Feature Extractor Hidden Layer 2"),
    createField("num_hidden_nodes1_classifier", 8, 4096, 1, 256, "Size of the first hidden layer in the subtype classifier.", "Classifier Hidden Layer 1"),
    createField("num_hidden_nodes2_classifier", 8, 4096, 1, 64, "Size of the second hidden layer in the subtype classifier.", "Classifier Hidden Layer 2"),
    createField("num_hidden_nodes1_discriminator", 8, 4096, 1, 256, "Size of the first hidden layer in the domain discriminator.", "Discriminator Hidden Layer 1"),
    createField("num_hidden_nodes2_discriminator", 8, 4096, 1, 64, "Size of the second hidden layer in the domain discriminator.", "Discriminator Hidden Layer 2"),
    createField("learning_rate_feature_extractor", 0.0000001, 1, 0.000001, 0.0001, "Step size used when updating the feature extractor weights.", "Feature Extractor Learning Rate"),
    createField("learning_rate_classifier", 0.0000001, 1, 0.000001, 0.00001, "Step size used when updating the classifier weights.", "Classifier Learning Rate"),
    createField("learning_rate_discriminator", 0.00000001, 1, 0.0000001, 0.000001, "Step size used when updating the discriminator weights.", "Discriminator Learning Rate"),
];

const MODEL_FIELDS = {
    BCtypeFinder: [
        ...COMMON_FIELDS,
        createField("num_epochs_pretraining", 1, 5000, 1, 800, "Number of rounds used for the initial training stage before adaptation starts.", "Pretraining Epochs"),
        createField("num_epochs_adversarial_training", 1, 5000, 1, 500, "Number of rounds used for domain-adversarial training.", "Adversarial Training Epochs"),
        createField("num_epochs_semi_supervised_learning", 1, 5000, 1, 500, "Number of rounds used for semi-supervised learning with target pseudo-labels.", "Semi-supervised Learning Epochs"),
        createField("num_epochs_fine_tuning", 1, 5000, 1, 800, "Number of rounds used for the final fine-tuning stage.", "Fine-tuning Epochs"),
    ],
    CancerSubminer: [
        ...COMMON_FIELDS,
        createField("num_epochs_pretraining", 1, 5000, 1, 800, "Number of rounds used for the initial training stage before subtype discovery begins.", "Pretraining Epochs"),
        createField("num_epochs_adversarial_training", 1, 5000, 1, 500, "Number of rounds used for domain-adversarial training.", "Adversarial Training Epochs"),
        createField("num_epochs_semi_supervised_learning", 1, 5000, 1, 300, "Number of rounds used for semi-supervised learning with target pseudo-labels.", "Semi-supervised Learning Epochs"),
        createField("num_epochs_fine_tuning", 1, 5000, 1, 300, "Number of rounds used for the final fine-tuning stage.", "Fine-tuning Epochs"),
    ],
};

const buildInitialValues = (modelName) =>
    (MODEL_FIELDS[modelName] || []).reduce((acc, field) => {
        acc[field.key] = String(field.defaultValue);
        return acc;
    }, {});

const validateField = (field, rawValue) => {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
        return `${field.key} must be a valid number.`;
    }
    if (value < field.min || value > field.max) {
        return `${field.key} must be between ${field.min} and ${field.max}.`;
    }
    return null;
};

const JobDetail = ({ project }) => {
    const dispatch = useDispatch();
    const projectId = project.id;
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [creatingJob, setCreatingJob] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null);
    const [autoEstimate, setAutoEstimate] = useState("1");
    const [subtypeCount, setSubtypeCount] = useState("2");
    const [usePretrainedModel, setUsePretrainedModel] = useState("1");
    const [parameterValues, setParameterValues] = useState({});
    const pollingRef = useRef(null);

    const { jobList, modelList } = useSelector((state) => state.job);
    const selectedModelName = modelList.find((m) => m.id === Number(selectedModel))?.name || "";
    const parameterFields = useMemo(() => MODEL_FIELDS[selectedModelName] || [], [selectedModelName]);

    const fetchJobs = async () => {
        setLoading(true);
        try {
            await dispatch(getJobList(projectId)).unwrap();
            setError(null);
        } catch (fetchError) {
            setError(fetchError);
            dispatch(showToast({ message: fetchError, type: "error" }));
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
        if (!selectedModelName) {
            setParameterValues({});
            return;
        }
        setParameterValues(buildInitialValues(selectedModelName));
        setAutoEstimate("1");
        setSubtypeCount("2");
        setUsePretrainedModel("1");
    }, [selectedModelName]);

    useEffect(() => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }

        const hasPendingJobs = jobList.some((job) => job.status === "Pending");
        const hasPreprocessingJobs = jobList.some((job) => job.status === "Preprocessing");
        const hasRunningJobs = jobList.some((job) => job.status === "Running");

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

    const getValidationErrors = () => {
        const errors = [];

        if (!selectedModel) {
            errors.push("Please select a model before creating a job.");
        }
        if (!project.target_file) {
            errors.push("Upload a target file before creating a job.");
        }
        if (usePretrainedModel === "0" && !project.source_file) {
            errors.push("Upload a source file or enable the pretrained model before creating a job.");
        }
        if (selectedModelName === "CancerSubminer") {
            const subtypeValue = Number(subtypeCount);
            if (!Number.isInteger(subtypeValue) || subtypeValue < 2 || subtypeValue > 50) {
                errors.push("num_subtype_user_defined must be between 2 and 50.");
            }
        }

        parameterFields.forEach((field) => {
            const fieldError = validateField(field, parameterValues[field.key]);
            if (fieldError) {
                errors.push(fieldError);
            }
        });

        return errors;
    };

    const createJobHandler = () => {
        const errors = getValidationErrors();
        if (errors.length > 0) {
            dispatch(showToast({ message: errors[0], type: "warning" }));
            return;
        }

        const numericParameters = parameterFields.map((field) => Number(parameterValues[field.key]));
        const modelParameters = selectedModelName === "CancerSubminer"
            ? [Number(autoEstimate), Number(subtypeCount), Number(usePretrainedModel), ...numericParameters]
            : [Number(usePretrainedModel), ...numericParameters];

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

    const validationErrors = getValidationErrors();

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
                        disabled={creatingJob || validationErrors.length > 0}
                        onClick={createJobHandler}
                    >
                        <FaPlus className="mr-2" /> Create Job
                    </button>
                </div>
            </div>

            {selectedModelName && (
                <div className="space-y-4 text-sm text-base-content">
                    <div>
                        <label className="font-medium mb-1 block">Use pretrained model:</label>
                        <div className="flex flex-wrap items-center gap-6">
                            <label className="flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="usePretrainedModel"
                                    checked={usePretrainedModel === "1"}
                                    value="1"
                                    onChange={() => setUsePretrainedModel("1")}
                                    className="radio radio-sm"
                                />
                                Yes
                            </label>
                            <label className="flex items-center gap-2">
                                <input
                                    type="radio"
                                    name="usePretrainedModel"
                                    checked={usePretrainedModel === "0"}
                                    value="0"
                                    onChange={() => setUsePretrainedModel("0")}
                                    className="radio radio-sm"
                                />
                                No
                            </label>
                        </div>
                        <p className="mt-2 text-xs text-base-content/70">
                            For the hyperparameter setting, the below default values are provided. Most users should not need to change them.
                        </p>
                    </div>

                    {selectedModelName === "CancerSubminer" && (
                        <div className="card border border-base-300 bg-base-100 shadow-sm">
                            <div className="card-body gap-4 p-5">
                                <div className="flex items-center gap-2">
                                    <h3 className="card-title text-base">Subtype Estimation</h3>
                                    <div
                                        className="tooltip tooltip-right"
                                        data-tip="Choose whether CancerSubminer should estimate the number of subtypes automatically or use a manually specified value."
                                    >
                                        <button type="button" className="btn btn-ghost btn-xs btn-circle text-info">
                                            <FaCircleInfo />
                                        </button>
                                    </div>
                                </div>

                                <div className="grid gap-4 md:grid-cols-2">
                                    <div>
                                        <label className="font-medium mb-1 block">Estimate Number of Subtypes</label>
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

                                    <div className="rounded-2xl border border-base-300 bg-base-200/60 p-4">
                                        <div className="mb-2 flex items-center gap-2">
                                            <label className="font-medium">Number of Subtypes</label>
                                            <div
                                                className="tooltip tooltip-left"
                                                data-tip="Used only when automatic estimation is turned off. This is the number of subtypes CancerSubminer will try to separate."
                                            >
                                                <button type="button" className="btn btn-ghost btn-xs btn-circle text-info">
                                                    <FaCircleInfo />
                                                </button>
                                            </div>
                                        </div>
                                        <div className="text-xs text-base-content/60">Parameter name: <span className="font-mono">num_subtype_user_defined</span></div>
                                        <input
                                            type="number"
                                            className="input input-bordered mt-3"
                                            value={subtypeCount}
                                            min={2}
                                            max={50}
                                            step={1}
                                            disabled={autoEstimate === "1"}
                                            onChange={(e) => setSubtypeCount(e.target.value)}
                                        />
                                        <span className="mt-2 block text-xs text-base-content/60">Allowed range: 2 to 50</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="card border border-base-300 bg-gradient-to-br from-base-100 to-base-200 shadow-sm">
                        <div className="card-body gap-5 p-5">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <h3 className="card-title text-base">Hyperparameters</h3>
                                    <p className="text-xs text-base-content/60">
                                        The defaults are suitable for most runs. Hover over the info icons for plain-language explanations.
                                    </p>
                                </div>
                                <div className="badge badge-outline badge-primary">Default-ready</div>
                            </div>
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                            {parameterFields.map((field) => (
                                <div key={field.key} className="rounded-2xl border border-base-300 bg-base-100 p-4 shadow-sm transition hover:border-primary/40 hover:shadow-md">
                                    <div className="mb-3 flex items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <label className="font-semibold leading-5">{field.label}</label>
                                            <div className="mt-1 text-xs text-base-content/60 font-mono break-all">{field.key}</div>
                                        </div>
                                        <div
                                            className="tooltip tooltip-left max-w-xs"
                                            data-tip={field.description}
                                        >
                                            <button type="button" className="btn btn-ghost btn-xs btn-circle text-info">
                                                <FaCircleInfo />
                                            </button>
                                        </div>
                                    </div>
                                    <input
                                        type="number"
                                        className="input input-bordered w-full bg-base-100"
                                        value={parameterValues[field.key] ?? ""}
                                        min={field.min}
                                        max={field.max}
                                        step={field.step}
                                        onChange={(e) => setParameterValues((prev) => ({ ...prev, [field.key]: e.target.value }))}
                                    />
                                    <div className="mt-3 flex items-center justify-between gap-2 text-xs text-base-content/60">
                                        <span>Range: {field.min} to {field.max}</span>
                                        <span>Default: {field.defaultValue}</span>
                                    </div>
                                    <p className="mt-2 text-xs leading-5 text-base-content/70">
                                        {field.description}
                                    </p>
                                </div>
                            ))}
                        </div>
                        </div>
                    </div>

                    {validationErrors.length > 0 && (
                        <div className="alert alert-warning shadow-sm">
                            <span>{validationErrors[0]}</span>
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
