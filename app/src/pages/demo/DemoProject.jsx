import React from 'react';
import {RouteConstants} from "../../shared/constants/RouteConstants.js";
import BackToTopButton from "../../components/BackToTopButton.jsx";
import {useNavigate} from "react-router-dom";
import {
    FaCalendarAlt, FaChartBar,
    FaClock,
    FaEdit,
    FaFileCsv, FaFileDownload,
    FaFileUpload, FaPlay,
    FaPlus, FaRegCalendarAlt,
    FaRegFileAlt, FaStopwatch,
    FaTrash,
    FaUpload
} from "react-icons/fa";
import {MdCancel} from "react-icons/md";
import EditProject from "../../components/project/EditProject.jsx";
import {AiOutlineFileText} from "react-icons/ai";
import JobStatusBadge from "../../components/project/job/JobStatusBadge.jsx";
import {useDispatch, useSelector} from "react-redux";
import {showToast} from "../../redux/toastSlice.jsx";
import {downloadExampleResults} from "../../redux/jobSlice.jsx";

const DemoProject = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const jobStatus = useSelector((state) => state.job.status);

    const editProjectHandler = () => {
        dispatch(showToast({ message: "Demo project cannot be edited" , type: "error" }));
    }

    const deleteProjectHandler = () => {
        dispatch(showToast({ message: "Demo project cannot be deleted" , type: "error" }));
    }

    const downloadResultHandler = (model) => {
        dispatch(downloadExampleResults(model))
            .unwrap()
            .then(() => {
                dispatch(showToast({
                    message: `${model} example results downloaded successfully`,
                    type: "success"
                }));
            })
            .catch((error) => {
                dispatch(showToast({
                    message: error || "Failed to download example results",
                    type: "error"
                }));
            });
    }

    const handleGetTemplate = () => {
        // Create CSV template content
        const csvContent = "sample_id,os_time,status\n" +
            "sample_001,365,1\n" +
            "sample_002,180,2\n" +
            "sample_003,730,1";

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'metadata_template.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const handleGetSourceTemplate = () => {
        // Create source template - CpG methylation data format with subtype labels
        const csvContent = "sample_id,cg00000029,cg00000108,cg00000109,cg00000165,cg00000236,subtype\n" +
            "sample_001,0.1234,0.5678,0.9012,0.3456,0.7890,LumA\n" +
            "sample_002,0.2345,0.6789,0.0123,0.4567,0.8901,LumB\n" +
            "sample_003,0.3456,0.7890,0.1234,0.5678,0.9012,Her2\n" +
            "sample_004,0.4567,0.8901,0.2345,0.6789,0.0123,Basal\n" +
            "sample_005,0.5678,0.9012,0.3456,0.7890,0.1234,Normal-like";

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'source_template.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const handleGetTargetTemplate = () => {
        // Create target template - CpG methylation data format with batch labels
        const csvContent = "sample_id,cg00000029,cg00000108,cg00000109,cg00000165,cg00000236,Batch\n" +
            "sample_001,0.2345,0.6789,0.0123,0.4567,0.8901,Batch_1\n" +
            "sample_002,0.3456,0.7890,0.1234,0.5678,0.9012,Batch_1\n" +
            "sample_003,0.4567,0.8901,0.2345,0.6789,0.0123,Batch_2\n" +
            "sample_004,0.5678,0.9012,0.3456,0.7890,0.1234,Batch_2\n" +
            "sample_005,0.6789,0.0123,0.4567,0.8901,0.2345,Batch_3";

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'target_template.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };


    return (
        <div className="flex justify-center">
            <div className="card w-full max-w-7xl shadow-lg p-5 bg-base-100 space-y-10 my-5">
                <div className="card-body">
                    <div className="space-y-8">
                        <div className="breadcrumbs text-md">
                            <ul>
                                <li>
                                    <button onClick={() => navigate(RouteConstants.demoDashboard)} className="cursor-pointer text-primary">
                                        Demo Dashboard
                                    </button>
                                </li>
                                <li className="text-base-content">Breast Cancer Subtyping (DEMO)</li>
                            </ul>
                        </div>

                        <div className="space-y-8">
                            <div className="flex justify-between items-center">
                                <h1 className="text-3xl font-bold text-base-content truncate max-w-[70%]">Breast Cancer Subtyping (DEMO)</h1>
                                <div className="flex space-x-3">
                                    <button
                                        className="btn btn-outline btn-info flex items-center"
                                        onClick={editProjectHandler}
                                    >
                                        <FaEdit className="mr-2" /> Edit
                                    </button>
                                    <button
                                        className="btn btn-outline btn-error flex items-center"
                                        onClick={deleteProjectHandler}
                                    >
                                        <FaTrash className="mr-2" /> Delete
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Project Info */}
                                <div className="p-5 bg-base-200 rounded-lg shadow-md">
                                    <h2 className="text-xl font-semibold text-base-content mb-3">Project Information</h2>
                                    <p className="text-lg font-medium flex items-center gap-2 pb-2">
                                        <strong>Cancer Type:</strong>
                                        <span className="badge badge-accent text-base-content">
                                            Breast Cancer
                                        </span>
                                    </p>
                                    <p className="text-lg font-medium"><strong>Description:</strong></p>
                                    <div className="p-3 bg-base-100 rounded-lg border border-base-300 max-h-40 overflow-y-auto break-words mb-3">
                                    Prototype analysis using a breast cancer DNA methylation dataset in CancerSubtyper paper. Source dataset consists of TCGA-BRCA cohort, 1,060 primary breast tumor samples profiled using the Illumina Human Infinium 450K and 27K platforms, along with established PAM50 subtype annotations. For the unlabeled data, we integrated three publicly available breast cancer methylation datasets from the GeO: GSE69914, GSE75067, and GSE72245, collectively comprising 611 tumor samples.
                                    </div>

                                    <p className="text-lg font-medium mt-2 flex items-center gap-2">
                                        <strong>Status:</strong>
                                        <span className="flex items-center gap-1 text-red-500">
                                            <MdCancel className="text-xl" /> Inactive
                                        </span>
                                    </p>
                                </div>

                                {/* Timestamps */}
                                <div className="p-5 bg-base-200 rounded-lg shadow-md space-y-2">
                                    <h2 className="text-xl font-semibold text-base-content mb-3">Timestamps</h2>
                                    <p className="text-lg font-medium flex items-center gap-2">
                                        <FaCalendarAlt className="text-info" />
                                        <strong>Created At:</strong>
                                        01/01/2025, 12:00 PM
                                    </p>
                                    <p className="text-lg font-medium flex items-center gap-2">
                                        <FaClock className="text-info" />
                                        <strong>Last Edited:</strong>
                                        01/01/2025, 12:00 PM
                                    </p>
                                </div>
                            </div>

                            <div className="p-5 bg-base-200 rounded-lg shadow-md">
                                <h2 className="text-xl font-semibold text-base-content mb-3">Metadata File</h2>
                                <div className="flex items-center gap-4">
                                    <label
                                        htmlFor="metadata-upload"
                                        className={`btn btn-outline btn-primary flex items-center cursor-pointer btn-disabled`}
                                    >
                                        <FaFileUpload className="mr-2" />
                                        <span>Upload Metadata</span>
                                    </label>
                                    <input
                                        id="metadata-upload"
                                        type="file"
                                        className="hidden"
                                        accept=".csv"
                                    />

                                    <button
                                        className="btn btn-outline btn-info flex items-center"
                                        onClick={handleGetTemplate}
                                    >
                                        <FaFileDownload className="mr-2" />
                                        <span>Get Template</span>
                                    </button>

                                    <a
                                        href="https://drive.google.com/drive/folders/1q-1ctysnpSjzl6r85oIaNr02NtABvcW0"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="btn btn-outline btn-success flex items-center"
                                        title="Download demo metadata (45 KB)"
                                    >
                                        <FaFileDownload className="mr-2" />
                                        <span>Demo Metadata</span>
                                    </a>

                                    <div className="flex items-center gap-2 text-green-500">
                                        <FaFileCsv className="text-lg" />
                                        <span>Demo File.csv</span>
                                    </div>
                                </div>
                            </div>

                            <EditProject />
                        </div>

                        <div className="p-5 bg-base-200 rounded-lg shadow-md" >
                            <div className="flex justify-between items-center mb-4">
                                <h2 className="text-xl font-semibold text-base-content flex items-center gap-2">
                                    <AiOutlineFileText className="text-primary" />
                                    Dataset
                                </h2>
                                <button
                                    className="btn btn-outline btn-primary flex items-center cursor-pointer btn-disabled"
                                >
                                    <FaUpload />
                                    Upload Sample
                                </button>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {/* Source File */}
                                <div className="p-4 bg-base-100 rounded-lg border border-base-300 shadow-sm">
                                    <div className="flex justify-between items-center mb-2">
                                        <p className="text-lg font-medium">
                                            <strong>Labeled Data (Source):</strong>
                                        </p>
                                        <div className="flex gap-1">
                                            <button
                                                className="btn btn-sm btn-outline btn-info flex items-center"
                                                onClick={handleGetSourceTemplate}
                                            >
                                                <FaFileDownload className="mr-1" />
                                                Template
                                            </button>
                                            <a
                                                href="https://drive.google.com/drive/folders/1q-1ctysnpSjzl6r85oIaNr02NtABvcW0"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="btn btn-sm btn-outline btn-success flex items-center"
                                                title="Download demo source data (198.7 MB)"
                                            >
                                                <FaFileDownload className="mr-1" />
                                                Demo
                                            </a>
                                        </div>
                                    </div>
                                    <div className="mt-2 p-2 bg-base-200 rounded-md border border-base-300 max-h-16 overflow-y-auto break-words">
                                        <div className="flex items-center gap-2 text-blue-600">
                                            <FaFileCsv className="text-lg" />
                                            <span>Demo File.csv</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Target File */}
                                <div className="p-4 bg-base-100 rounded-lg border border-base-300 shadow-sm">
                                    <div className="flex justify-between items-center mb-2">
                                        <p className="text-lg font-medium">
                                            <strong>Unlabeled Data (Target):</strong>
                                        </p>
                                        <div className="flex gap-1">
                                            <button
                                                className="btn btn-sm btn-outline btn-info flex items-center"
                                                onClick={handleGetTargetTemplate}
                                            >
                                                <FaFileDownload className="mr-1" />
                                                Template
                                            </button>
                                            <a
                                                href="https://drive.google.com/drive/folders/1q-1ctysnpSjzl6r85oIaNr02NtABvcW0"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="btn btn-sm btn-outline btn-success flex items-center"
                                                title="Download demo target data (101.9 MB)"
                                            >
                                                <FaFileDownload className="mr-1" />
                                                Demo
                                            </a>
                                        </div>
                                    </div>
                                    <div className="mt-2 p-2 bg-base-200 rounded-md border border-base-300 max-h-16 overflow-y-auto break-words">
                                        <div className="flex items-center gap-2 text-blue-600">
                                            <FaFileCsv className="text-lg" />
                                            <span>Demo File.csv</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>


                        <div className="p-6 bg-base-200 rounded-lg shadow-md space-y-5">
                            <div className="flex flex-wrap justify-between items-center gap-4">
                                <h2 className="text-xl font-semibold text-base-content flex items-center gap-2">
                                    <FaRegFileAlt className="text-primary" />
                                    Job Details
                                </h2>

                                <div className="flex items-center gap-3">
                                    <select
                                        className="select select-bordered text-base-content w-56"
                                        value={""}
                                    >
                                        <option value="" disabled>Select a Model</option>
                                    </select>

                                    <button
                                        className="btn btn-outline btn-primary flex items-center"
                                        disabled={true}
                                    >
                                        <FaPlus className="mr-2" /> Create Job
                                    </button>
                                </div>
                            </div>

                            <div className="space-y-6">
                                <div className="p-6 bg-base-100 rounded-lg shadow-md border border-base-300">
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className="text-lg font-semibold text-base-content flex items-center gap-3">
                                            <span className="text-primary">Job ID: 1</span>
                                            <span className="text-gray-500 font-medium">| Model: BCtypeFinder</span>
                                        </h3>
                                        <div className="flex items-center gap-3">
                                            <JobStatusBadge status={"Completed"}/>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaRegCalendarAlt className="text-primary"/>
                                            <strong>Created:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaPlay className="text-green-500"/>
                                            <strong>Started:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaStopwatch className="text-yellow-500"/>
                                            <strong>Finished:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                    </div>

                                    <div className="flex justify-end gap-3 mt-5">
                                        <button
                                            className="btn btn-sm btn-outline btn-success flex items-center"
                                            onClick={() => downloadResultHandler('bctypefinder')}
                                            disabled={jobStatus === 'loading'}
                                        >
                                            <FaFileDownload className="mr-2"/>
                                            {jobStatus === 'loading' ? 'Downloading...' : 'Download Result'}
                                        </button>
                                        <button
                                            className="btn btn-sm btn-outline btn-info flex items-center"
                                            onClick={() => navigate(RouteConstants.visualizationExamples, { state: { modelName: "BCtypeFinder" } })}
                                        >
                                            <FaChartBar className="mr-2"/>
                                            Visualize
                                        </button>
                                    </div>
                                </div>

                                <div className="p-6 bg-base-100 rounded-lg shadow-md border border-base-300">
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className="text-lg font-semibold text-base-content flex items-center gap-3">
                                            <span className="text-primary">Job ID: 2</span>
                                            <span className="text-gray-500 font-medium">| Model: CancerSubminer</span>
                                        </h3>
                                        <div className="flex items-center gap-3">
                                            <JobStatusBadge status={"Completed"}/>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaRegCalendarAlt className="text-primary"/>
                                            <strong>Created:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaPlay className="text-green-500"/>
                                            <strong>Started:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                        <p className="flex items-center gap-2 text-base-content">
                                            <FaStopwatch className="text-yellow-500"/>
                                            <strong>Finished:</strong> 01/01/2025, 12:00 PM
                                        </p>
                                    </div>

                                    <div className="flex justify-end gap-3 mt-5">
                                        <button
                                            className="btn btn-sm btn-outline btn-success flex items-center"
                                            onClick={() => downloadResultHandler('cancersubminer')}
                                            disabled={jobStatus === 'loading'}
                                        >
                                            <FaFileDownload className="mr-2"/>
                                            {jobStatus === 'loading' ? 'Downloading...' : 'Download Result'}
                                        </button>
                                        <button
                                            className="btn btn-sm btn-outline btn-info flex items-center"
                                            onClick={() => navigate(RouteConstants.visualizationExamples, { state: { modelName: "CancerSubminer" } })}
                                        >
                                            <FaChartBar className="mr-2"/>
                                            Visualize
                                        </button>
                                    </div>
                                </div>
                            </div>

                        </div>

                    </div>
                </div>
            </div>
            <BackToTopButton />
        </div>
    );
};

export default DemoProject;