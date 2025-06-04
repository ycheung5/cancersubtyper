import React from "react";
import { FaSpinner, FaTimesCircle, FaHourglassHalf } from "react-icons/fa";

const JobFeedback = ({ loading, error, jobs }) => {
    if (loading) {
        return (
            <div className="flex justify-center items-center space-x-2 text-primary">
                <FaSpinner className="animate-spin text-xl" />
                <span>Loading jobs...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="alert alert-error shadow-lg flex items-center p-3 border border-red-500">
                <FaTimesCircle className="text-lg text-white" />
                <span>{error}</span>
            </div>
        );
    }

    if (jobs.length === 0) {
        return (
            <div className="alert bg-base-100 border border-base-300 shadow-md flex items-center p-3">
                <FaHourglassHalf className="text-yellow-500" />
                <span>No jobs found for this project.</span>
            </div>
        );
    }

    return null;
};

export default JobFeedback;
