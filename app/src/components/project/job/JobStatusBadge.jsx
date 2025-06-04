import React from "react";
import {
    FaHourglassHalf, FaCog, FaPlay, FaCheckCircle, FaTimesCircle
} from "react-icons/fa";

const statusConfig = {
    Pending: { icon: <FaHourglassHalf />, color: "bg-yellow-500 text-white", label: "Pending" },
    Preprocessing: { icon: <FaCog className="animate-spin" />, color: "bg-purple-500 text-white", label: "Preprocessing" },
    Running: { icon: <FaPlay className="animate-pulse" />, color: "bg-blue-500 text-white", label: "Running" },
    Completed: { icon: <FaCheckCircle />, color: "bg-green-500 text-white", label: "Completed" },
    Failed: { icon: <FaTimesCircle />, color: "bg-red-500 text-white", label: "Failed" },
};

const JobStatusBadge = ({ status }) => {
    const { icon, color, label } = statusConfig[status] || {
        icon: <FaHourglassHalf />,
        color: "bg-gray-500 text-white",
        label: "Unknown"
    };

    return (
        <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold ${color}`}>
            {icon}
            <span>{label}</span>
        </span>
    );
};

export default JobStatusBadge;
