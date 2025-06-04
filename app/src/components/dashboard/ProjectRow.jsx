import React from "react";
import { AiOutlineClockCircle, AiOutlineEye } from "react-icons/ai";
import { MdCheckCircle, MdCancel } from "react-icons/md";
import { formatDate, truncateString } from "../../shared/utils/utils.jsx";
import { RouteConstants } from "../../shared/constants/RouteConstants.js";
import { useNavigate } from "react-router-dom";

const ProjectRow = ({ project }) => {
    const navigate = useNavigate();

    return (
        <tr className="hover:bg-base-200 transition-all">
            {/* Project Name */}
            <td className="px-6 py-3 font-semibold text-base-content relative max-w-[250px] truncate">
                <span className="block truncate">{truncateString(20, project.name)}</span>
            </td>

            {/* Tumor Type */}
            <td className="px-6 py-3 text-base-content relative max-w-[200px] truncate">
                <span className="block truncate">
                    {project.tumor_type || <span className="italic text-gray-500">Unknown</span>}
                </span>
            </td>

            {/* Description */}
            <td className="px-6 py-3 text-base-content relative max-w-[250px] truncate">
                <span className="block truncate">
                    {truncateString(50, project.description) || "No description"}
                </span>
            </td>

            {/* Status */}
            <td className="px-6 py-3 whitespace-nowrap">
                <div className="flex items-center gap-2">
                    {project.active ? (
                        <MdCheckCircle className="text-success text-xl" />
                    ) : (
                        <MdCancel className="text-error text-xl" />
                    )}
                    <span className={`badge ${project.active ? "badge-success" : "badge-error"}`}>
                        {project.active ? "Active" : "Inactive"}
                    </span>
                </div>
            </td>

            {/* Last Edited */}
            <td className="px-6 py-3 whitespace-nowrap text-base-content">
                <div className="flex items-center gap-2">
                    <AiOutlineClockCircle className="text-primary text-lg" />
                    {formatDate(project.edited_at)}
                </div>
            </td>

            {/* Actions */}
            <td className="px-6 py-3">
                <button
                    className="btn btn-primary btn-sm flex items-center gap-2 transition-all hover:shadow-md"
                    onClick={() => navigate(`${RouteConstants.project}/${project.id}`)}
                >
                    <AiOutlineEye className="text-lg" />
                    View
                </button>
            </td>
        </tr>
    );
};

export default ProjectRow;
