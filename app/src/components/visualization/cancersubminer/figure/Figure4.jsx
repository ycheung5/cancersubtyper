import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {getCSPlot4Table} from "../../../../redux/visualizationSlice.jsx";
import Plot4Table from "../plot/Plot4Table.jsx";
import { FaTable, FaSyncAlt } from "react-icons/fa";

const Figure4 = ({ id }) => {
    const dispatch = useDispatch();
    const { bc_plot4_table } = useSelector(state => state.visualization.plots);

    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        dispatch(getCSPlot4Table({ job_id: id }))
            .unwrap()
            .catch(error => console.error("Error fetching CSPlot4Table:", error))
            .finally(() => setLoading(false));
    }, [dispatch, id]);

    return (
        <div className="bg-base-200 p-5 rounded-lg shadow-md border border-base-300 mt-5">
            {/* Section Header */}
            <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                <FaTable className="text-primary" />
                Subtyping Results
            </h4>

            {/* Description */}
            <p className="text-sm text-gray-600 mb-3">
                This table displays the classification results for the given job.
                Each entry represents a prediction based on the selected model.
            </p>

            {/* Loading Indicator */}
            {loading && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading classification results...</span>
                </div>
            )}

            {/* Classification Table */}
            {!loading && bc_plot4_table && <Plot4Table />}
        </div>
    );
};

export default Figure4;
