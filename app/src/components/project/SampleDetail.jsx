import React, { useState } from "react";
import {FaFileCsv, FaUpload} from "react-icons/fa";
import { AiOutlineFileText } from "react-icons/ai";
import { useSelector } from "react-redux";
import UploadSample from "./UploadSample.jsx";

const SampleDetail = ({ projectId, target, source }) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const used = useSelector((state) => state.job.jobList.length > 0);

    return (
        <div className="p-5 bg-base-200 rounded-lg shadow-md" >
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-base-content flex items-center gap-2">
                    <AiOutlineFileText className="text-primary" />
                    Sample Files
                </h2>
                <button
                    className="btn btn-outline btn-primary flex items-center cursor-pointer"
                    onClick={() => setIsModalOpen(true)}
                    disabled={used}
                >
                    <FaUpload />
                    Upload Sample
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Source File */}
                <div className="p-4 bg-base-100 rounded-lg border border-base-300 shadow-sm">
                    <p className="text-lg font-medium">
                        <strong>Source File:</strong>
                    </p>
                    <div className="mt-2 p-2 bg-base-200 rounded-md border border-base-300 max-h-16 overflow-y-auto break-words">
                        {source ? (
                            <div className="flex items-center gap-2 text-blue-600">
                                <FaFileCsv className="text-lg" />
                                <span>{source}</span>
                            </div>
                        ) : (
                            <span className="text-red-500">No source file uploaded</span>
                        )}
                    </div>
                </div>

                {/* Target File */}
                <div className="p-4 bg-base-100 rounded-lg border border-base-300 shadow-sm">
                    <p className="text-lg font-medium">
                        <strong>Target File:</strong>
                    </p>
                    <div className="mt-2 p-2 bg-base-200 rounded-md border border-base-300 max-h-16 overflow-y-auto break-words">
                        {target ? (
                            <div className="flex items-center gap-2 text-blue-600">
                                <FaFileCsv className="text-lg" />
                                <span>{target}</span>
                            </div>
                        ) : (
                            <span className="text-red-500">No target file uploaded</span>
                        )}
                    </div>
                </div>
            </div>

            {isModalOpen && <UploadSample projectId={projectId} onClose={() => setIsModalOpen(false)} />}
        </div>
    );
};

export default SampleDetail;
