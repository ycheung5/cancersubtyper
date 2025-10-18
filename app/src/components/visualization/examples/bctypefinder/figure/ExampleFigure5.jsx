import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
    getBCExamplePlot5Option,
    getBCExamplePlot5,
} from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot5 from "../plot/ExamplePlot5.jsx";
import { FaChartLine, FaFilter, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../../shared/utils/downloadPlot.jsx";

const ExampleFigure5 = () => {
    const dispatch = useDispatch();
    const { bc_plot5_option, bc_plot5 } = useSelector(
        (s) => s.visualizationExample.plots
    );

    const [loading, setLoading] = useState(true);
    const [selectedBatch, setSelectedBatch] = useState("");

    // fetch batches
    useEffect(() => {
        setLoading(true);
        dispatch(getBCExamplePlot5Option())
            .unwrap()
            .catch((e) => console.error("Error fetching example KM options:", e))
            .finally(() => setLoading(false));
    }, [dispatch]);

    // pick first batch by default
    useEffect(() => {
        if (bc_plot5_option?.length) setSelectedBatch((b) => b || bc_plot5_option[0]);
    }, [bc_plot5_option]);

    // fetch KM for batch
    useEffect(() => {
        if (!selectedBatch) return;
        setLoading(true);
        dispatch(getBCExamplePlot5({ batch: selectedBatch }))
            .unwrap()
            .catch((e) => console.error("Error fetching example KM data:", e))
            .finally(() => setLoading(false));
    }, [dispatch, selectedBatch]);

    return (
        <div className="bg-base-200 p-6 rounded-lg shadow-md border border-base-300 mt-5">
            {/* Section Header */}
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartLine className="text-primary" />
                Kaplan-Meier Survival Analysis
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                This Kaplan-Meier plot illustrates survival probabilities over time for different cancer subtypes
                within the selected batch. The x-axis represents survival time, while the y-axis shows the estimated
                probability of survival.
            </p>

            {/* Batch Selector */}
            <div className="flex flex-wrap gap-4 items-center bg-base-100 p-4 rounded-lg shadow-sm border border-base-300">
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Batch:
                </label>
                <select
                    className="select select-bordered w-48"
                    value={selectedBatch}
                    onChange={(e) => setSelectedBatch(e.target.value)}
                    disabled={loading}
                >
                    {bc_plot5_option && bc_plot5_option.length > 0 ? (
                        bc_plot5_option.map((batch) => (
                            <option key={batch} value={batch}>{batch}</option>
                        ))
                    ) : (
                        <option value="">No Batches Available</option>
                    )}
                </select>
            </div>

            {/* Loading Indicator */}
            <div className="mt-4">
                {loading && (
                    <div className="flex justify-center items-center mt-4 text-primary">
                        <FaSyncAlt className="animate-spin text-xl" />
                        <span className="ml-2">Loading Kaplan-Meier plot...</span>
                    </div>
                )}
            </div>

            {/* Plot Section */}
            {!loading && bc_plot5?.data.length > 0 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartLine className="text-primary" />
                        Kaplan-Meier Plot
                    </h4>

                    {/* Description + Plot */}
                    <div className="flex flex-col">
                        <p className="text-sm text-gray-500 mb-6">
                            Curves represent the survival distribution for each predicted subtype.
                            Divergence among curves may indicate the clinical relevance of subtype separation.
                        </p>
                        {/* Export Buttons */}
                        <div className="flex justify-end gap-4 mb-4">
                            <button className="btn btn-sm btn-outline" onClick={() => downloadSVG("bc-example-plot5", "plot5.svg")}>
                                Download SVG
                            </button>
                            <button className="btn btn-sm btn-outline" onClick={() => downloadPNG("bc-example-plot5", "plot5.png")}>
                                Download PNG
                            </button>
                        </div>
                        <ExamplePlot5 />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ExampleFigure5;
