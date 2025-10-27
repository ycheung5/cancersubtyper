import React, { useState } from "react";
import {FaFileCsv, FaUpload, FaDownload} from "react-icons/fa";
import { AiOutlineFileText } from "react-icons/ai";
import { useSelector } from "react-redux";
import UploadSample from "./UploadSample.jsx";

const SampleDetail = ({ projectId, target, source }) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const used = useSelector((state) => state.job.jobList.length > 0);

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
        <div className="p-5 bg-base-200 rounded-lg shadow-md" >
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-base-content flex items-center gap-2">
                    <AiOutlineFileText className="text-primary" />
                    Dataset
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
                    <div className="flex justify-between items-center mb-2">
                        <p className="text-lg font-medium">
                            <strong>Labeled Data (Source):</strong>
                        </p>
                        <button
                            className="btn btn-sm btn-outline btn-info flex items-center"
                            onClick={handleGetSourceTemplate}
                        >
                            <FaDownload className="mr-1" />
                            Template
                        </button>
                    </div>
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
                    <div className="flex justify-between items-center mb-2">
                        <p className="text-lg font-medium">
                            <strong>Unlabeled Data (Target):</strong>
                        </p>
                        <button
                            className="btn btn-sm btn-outline btn-info flex items-center"
                            onClick={handleGetTargetTemplate}
                        >
                            <FaDownload className="mr-1" />
                            Template
                        </button>
                    </div>
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
