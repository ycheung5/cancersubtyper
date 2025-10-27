import {
    AiOutlineClockCircle,
    AiOutlineDatabase,
    AiOutlineEye,
    AiOutlinePlusCircle,
    AiOutlineProject
} from "react-icons/ai";
import React from "react";
import {MdCancel} from "react-icons/md";
import {RouteConstants} from "../../shared/constants/RouteConstants.js";
import {useNavigate} from "react-router-dom";
import {useDispatch} from "react-redux";
import {showToast} from "../../redux/toastSlice.jsx";

const DemoDashboard = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();

    const createProjectHandler = () => {
        dispatch(showToast({ message: "Demo dashboard cannot create new project" , type: "error" }));
    }

    return (
        <div className="min-h-screen bg-base-100 p-10 flex flex-col items-center pt-[var(--navbar-height)] mt-5 overflow-auto">
            <h1 className="text-4xl font-bold mb-8 text-base-content flex items-center gap-2">
                Demo Dashboard
            </h1>

            <div className="bg-base-100 shadow-lg rounded-lg p-6 mb-8 border border-base-300 max-w-4xl w-full">
                <h2 className="text-xl font-semibold text-base-content mb-2 flex items-center gap-2">
                    <AiOutlineDatabase className="text-primary text-2xl" />
                    Storage Usage
                </h2>

                <div>
                    <p className="text-base-content text-lg">
                        Used:
                        <span className="font-bold text-primary"> 1 GB </span> /
                        <span className="font-bold"> 20 GB </span>
                    </p>
                    <progress
                        className="progress progress-primary w-full mt-3"
                        value={1}
                        max={20}
                    ></progress>
                </div>
            </div>

            <div className="bg-base-100 shadow-lg rounded-lg p-6 border border-gray-200 w-7xl">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-semibold text-base-content flex items-center gap-2">
                        <AiOutlineProject className="text-primary text-3xl" />
                        Your Projects
                    </h2>
                    <button
                        className="btn btn-primary px-6 py-2 font-semibold flex items-center gap-2 shadow-md hover:shadow-lg transition-all"
                        onClick={createProjectHandler}
                    >
                        <AiOutlinePlusCircle className="text-lg" />
                        Create Project
                    </button>
                </div>

                <div className="overflow-x-auto">
                    <table className="table w-full border border-gray-200 shadow-sm">
                        <thead className="bg-base-200">
                        <tr>
                            <th className="px-6 py-3 text-left text-base-content w-[20%]">Project Name</th>
                            <th className="px-6 py-3 text-left text-base-content w-[25%]">Cancer Type</th>
                            <th className="px-6 py-3 text-left text-base-content w-[25%]">Description</th>
                            <th className="px-6 py-3 text-left text-base-content w-[15%]">Status</th>
                            <th className="px-6 py-3 text-left text-base-content w-[15%]">Last Edited</th>
                            <th className="px-6 py-3 text-left text-base-content w-[10%]">Actions</th>
                        </tr>
                        </thead>
                        <tbody>
                            <tr className="hover:bg-base-200 transition-all">
                                {/* Project Name */}
                                <td className="px-6 py-3 font-semibold text-base-content relative max-w-[250px] truncate">
                                    <span className="block truncate">Breast Cancer Subtyping (DEMO)</span>
                                </td>

                                {/* Cancer Type */}
                                <td className="px-6 py-3 text-base-content relative max-w-[200px] truncate">
                                    <span className="block truncate">Breast Cancer</span>
                                </td>

                                {/* Description */}
                                <td className="px-6 py-3 text-base-content relative max-w-[250px] truncate">
                                    <span className="block truncate">Prototype analysis using a breast cancer DNA methylation dataset in CancerSubtyper paper. Source dataset consists of TCGA-BRCA cohort, 1,060 primary breast tumor samples profiled using the Illumina Human Infinium 450K and 27K platforms, along with established PAM50 subtype annotations. For the unlabeled data, we integrated three publicly available breast cancer methylation datasets from the GeO: GSE69914, GSE75067, and GSE72245, collectively comprising 611 tumor samples.</span>
                                </td>

                                {/* Status */}
                                <td className="px-6 py-3 whitespace-nowrap">
                                    <div className="flex items-center gap-2">
                                        <MdCancel className="text-error text-xl" />
                                        <span className={"badge-error"}>Inactive</span>
                                    </div>
                                </td>

                                {/* Last Edited */}
                                <td className="px-6 py-3 whitespace-nowrap text-base-content">
                                    <div className="flex items-center gap-2">
                                        <AiOutlineClockCircle className="text-primary text-lg" />
                                        01/01/2025, 12:00 PM
                                    </div>
                                </td>

                                {/* Actions */}
                                <td className="px-6 py-3">
                                    <button
                                        className="btn btn-primary btn-sm flex items-center gap-2 transition-all hover:shadow-md"
                                        onClick={() => navigate(RouteConstants.demoProject)}
                                    >
                                        <AiOutlineEye className="text-lg" />
                                        View
                                    </button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>


        </div>
    );
};

export default DemoDashboard;