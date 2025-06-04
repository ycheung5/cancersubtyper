import React, { useState } from "react";
import { FaFileUpload, FaTimes, FaSpinner, FaTimesCircle } from "react-icons/fa";
import { useDispatch } from "react-redux";
import { uploadSampleFile } from "../../redux/projectSlice";
import { showToast } from "../../redux/toastSlice";
import axios from "axios";

const UploadSample = ({ projectId, onClose }) => {
    const dispatch = useDispatch();
    const [sourceFile, setSourceFile] = useState(null);
    const [targetFile, setTargetFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState("");
    const [cancelTokenSource, setCancelTokenSource] = useState(null);
    const [abortController, setAbortController] = useState(null);

    const handleFileChange = (e, type) => {
        const file = e.target.files[0];
        if (file && file.name.endsWith(".gz")) {
            type === "source" ? setSourceFile(file) : setTargetFile(file);
            setError("");
        } else {
            setError("Only .gz compressed files are supported.");
        }
    };

    const handleDragOver = (e) => e.preventDefault();

    const handleDrop = (e, type) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith(".gz")) {
            type === "source" ? setSourceFile(file) : setTargetFile(file);
            setError("");
        } else {
            setError("Only .gz compressed files are supported.");
        }
    };

    const handleUpload = async () => {
        if (!sourceFile || !targetFile) {
            setError("Both source and target files are required.");
            return;
        }

        setLoading(true);
        setError("");
        setProgress(0);

        const cancelSource = axios.CancelToken.source();
        const controller = new AbortController();
        setCancelTokenSource(cancelSource);
        setAbortController(controller);

        dispatch(
            uploadSampleFile({
                project_id: projectId,
                sourceFile,
                targetFile,
                setProgress,
                cancelTokenSource: cancelSource,
                abortSignal: controller.signal,
            })
        )
            .unwrap()
            .then(() => {
                setProgress(100);
                dispatch(showToast({ message: "Sample uploaded successfully!", type: "success" }));
            })
            .catch((err) => {
                console.error(err);
                dispatch(showToast({ message: err || "Sample upload failed", type: "error" }));
            })
            .finally(() => {
                setLoading(false);
                onClose();
            });
    };

    const handleCancelUpload = () => {
        abortController?.abort();
        cancelTokenSource?.cancel("User canceled the upload.");
        setLoading(false);
        setProgress(0);
    };

    return (
        <dialog className="modal modal-open">
            <div className="modal-box bg-base-100 border border-base-300 shadow-lg rounded-lg">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-base-content">Upload Sample Files</h2>
                    <button onClick={onClose} className="btn btn-sm btn-circle btn-outline" disabled={loading}>
                        <FaTimes />
                    </button>
                </div>

                <div className="space-y-4">
                    <div
                        className={`p-6 border-2 border-dashed border-base-300 rounded-lg flex flex-col items-center justify-center ${loading ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-base-200 transition"}`}
                        onDragOver={(e) => !loading && handleDragOver(e)}
                        onDrop={(e) => !loading && handleDrop(e, "source")}
                    >
                        <FaFileUpload className="text-primary text-4xl mb-2" />
                        <p className="text-base-content text-center">Drag & drop your <strong>Source File</strong> here</p>
                        <input
                            type="file"
                            className="hidden"
                            id="sourceInput"
                            accept=".gz"
                            onChange={(e) => handleFileChange(e, "source")}
                            disabled={loading}
                        />
                        <label htmlFor="sourceInput" className={`btn btn-outline btn-primary mt-2 ${loading ? "btn-disabled" : ""}`}>
                            Choose Source File
                        </label>
                        {sourceFile && <p className="text-sm text-green-500 mt-2">✔ {sourceFile.name}</p>}
                    </div>

                    <div
                        className={`p-6 border-2 border-dashed border-base-300 rounded-lg flex flex-col items-center justify-center ${loading ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-base-200 transition"}`}
                        onDragOver={(e) => !loading && handleDragOver(e)}
                        onDrop={(e) => !loading && handleDrop(e, "target")}
                    >
                        <FaFileUpload className="text-primary text-4xl mb-2" />
                        <p className="text-base-content text-center">Drag & drop your <strong>Target File</strong> here</p>
                        <input
                            type="file"
                            className="hidden"
                            id="targetInput"
                            accept=".gz"
                            onChange={(e) => handleFileChange(e, "target")}
                            disabled={loading}
                        />
                        <label htmlFor="targetInput" className={`btn btn-outline btn-primary mt-2 ${loading ? "btn-disabled" : ""}`}>
                            Choose Target File
                        </label>
                        {targetFile && <p className="text-sm text-green-500 mt-2">✔ {targetFile.name}</p>}
                    </div>
                </div>

                {loading && (
                    <div className="mt-4">
                        <progress className="progress progress-primary w-full" value={progress} max="100"></progress>
                        <p className="text-sm text-base-content mt-2 text-center">{progress}%</p>
                    </div>
                )}

                {error && (
                    <div className="alert alert-error mt-3 flex items-center gap-2">
                        <FaTimesCircle className="text-red-500" />
                        {error}
                    </div>
                )}

                <div className="modal-action">
                    <button className="btn btn-outline" hidden={loading} onClick={onClose}>
                        Cancel
                    </button>
                    {loading ? (
                        <button className="btn btn-error flex items-center" onClick={handleCancelUpload} disabled={progress >= 100}>
                            <FaTimesCircle className="mr-2" />
                            Cancel Upload
                        </button>
                    ) : (
                        <button
                            className="btn btn-primary flex items-center"
                            onClick={handleUpload}
                            disabled={!sourceFile || !targetFile}
                        >
                            <FaSpinner className={`${loading ? "animate-spin" : "hidden"} mr-2`} />
                            Upload
                        </button>
                    )}
                </div>
            </div>
        </dialog>
    );
};

export default UploadSample;
