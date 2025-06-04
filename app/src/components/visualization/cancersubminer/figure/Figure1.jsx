import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getCSPlot1Option, getCSPlot1, getCSPlot1Table } from "../../../../redux/visualizationSlice.jsx";
import Plot1 from "../plot/Plot1.jsx";
import Plot1Table from "../plot/Plot1Table.jsx";
import { FaFilter, FaChartBar, FaTable, FaSyncAlt } from "react-icons/fa";
import {downloadPNG, downloadSVG} from "../../../../shared/utils/downloadPlot.jsx";

const Figure1 = ({ id }) => {
    const dispatch = useDispatch();
    const { cs_plot1_option = {}, cs_plot1, cs_plot1_table } = useSelector(state => state.visualization.plots);

    const [loadingState, setLoadingState] = useState("loading-options");
    const [selectedBatch, setSelectedBatch] = useState("");
    const [selectedSubtype, setSelectedSubtype] = useState("");

    // Fetch batch-subtype options
    useEffect(() => {
        setLoadingState("loading-options");
        dispatch(getCSPlot1Option({ job_id: id }))
            .unwrap()
            .then(() => setLoadingState("idle"))
            .catch(error => {
                console.error("Error fetching CSPlot1Option:", error);
                setLoadingState("idle");
            });
    }, [dispatch, id]);

    // Auto-select first batch and subtype when options load
    useEffect(() => {
        if (cs_plot1_option && Object.keys(cs_plot1_option).length > 0) {
            const firstBatch = Object.keys(cs_plot1_option)[0];
            const firstSubtype = [...(cs_plot1_option[firstBatch] || [])].sort()[0] || "";
            setSelectedBatch(firstBatch);
            setSelectedSubtype(firstSubtype);
        }
    }, [cs_plot1_option]);

    // Fetch heatmap & table data when batch and subtype are selected
    useEffect(() => {
        if (selectedBatch && selectedSubtype) {
            setLoadingState("loading-heatmap");
            dispatch(getCSPlot1({ job_id: id, batch: selectedBatch, subtype: selectedSubtype }))
                .unwrap()
                .then((res) => {
                    // Extract unique clusters from rowLabel and colLabel
                    const clusters = new Set(
                        res.map(({ rowLabel, colLabel }) => [
                            rowLabel.replace("CpG Cluster ", "").trim(),
                            colLabel.replace("CpG Cluster ", "").trim(),
                        ]).flat()
                    );

                    // Convert Set to a comma-separated string and request table data
                    const clustersString = [...clusters].join(",");
                    setLoadingState("loading-table");
                    dispatch(getCSPlot1Table({ job_id: id, clusters: clustersString }))
                        .unwrap()
                        .then(() => setLoadingState("idle"))
                        .catch(error => {
                            console.error("Error fetching Plot1 table:", error);
                            setLoadingState("idle");
                        });
                })
                .catch(error => {
                    console.error("Error fetching heatmap:", error);
                    setLoadingState("idle");
                });
        }
    }, [dispatch, id, selectedBatch, selectedSubtype]);

    // Extract options
    const batchOptions = Object.keys(cs_plot1_option);
    const subtypes = selectedBatch ? (cs_plot1_option[selectedBatch] || []) : [];

    return (
        <div className="bg-base-200 p-6 rounded-lg shadow-md border border-base-300">
            {/* Section Title */}
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartBar className="text-primary" />
                CpG Cluster Analysis
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                This section provides interactive visualizations and data tables for exploring CpG clusters across different batches and subtypes.
                Use the selectors below to filter the heatmap and corresponding cluster table.
            </p>

            {/* Filters */}
            <div className="flex flex-wrap gap-4 items-center bg-base-100 p-4 rounded-lg shadow-sm border border-base-300">
                {/* Batch Selection */}
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Batch:
                </label>
                <select
                    value={selectedBatch}
                    onChange={(e) => {
                        setSelectedBatch(e.target.value);
                        setSelectedSubtype(cs_plot1_option[e.target.value]?.[0] || "");
                    }}
                    className="select select-bordered w-48"
                    disabled={loadingState === "loading-options"}
                >
                    {batchOptions.length > 0 ? (
                        batchOptions.map((batch) => (
                            <option key={batch} value={batch}>{batch}</option>
                        ))
                    ) : (
                        <option value="">No Batches Available</option>
                    )}
                </select>

                {/* Subtype Selection */}
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Subtype:
                </label>
                <select
                    value={selectedSubtype}
                    onChange={(e) => setSelectedSubtype(e.target.value)}
                    className="select select-bordered w-48"
                    disabled={loadingState === "loading-options"}
                >
                    {subtypes.length > 0 ? (
                        [...subtypes]
                            .sort((a, b) => a.localeCompare(b))
                            .map((subtype) => (
                            <option key={subtype} value={subtype}>{subtype}</option>
                        ))
                    ) : (
                        <option value="">No Subtypes Available</option>
                    )}
                </select>
            </div>

            {/* Loading States */}
            {loadingState === "loading-options" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading cluster options...</span>
                </div>
            )}
            {loadingState === "loading-heatmap" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading heatmap data...</span>
                </div>
            )}
            {loadingState === "loading-table" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading CpG table data...</span>
                </div>
            )}

            {/* Heatmap Plot */}
            {loadingState === "idle" && cs_plot1?.length > 0 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartBar className="text-primary" />
                        Correlation Heatmap
                    </h4>
                    <p className="text-sm text-gray-500 mb-6">
                        This heatmap visualizes the pairwise correlations among the top 30 CpG clusters with the highest absolute correlation values.
                    </p>

                    <div className="flex justify-end gap-4">
                        <button className="btn btn-sm btn-outline" onClick={() => downloadSVG("cs-plot1", "plot1.svg")}>
                            Download SVG
                        </button>
                        <button className="btn btn-sm btn-outline" onClick={() => downloadPNG("cs-plot1", "plot1.png")}>
                            Download PNG
                        </button>
                    </div>

                    <div className="flex justify-center mt-4" id="plot1">
                        <div className="w-full max-w-4xl">
                            <Plot1 />
                        </div>
                    </div>
                </div>
            )}

            {/* CpG Table */}
            {loadingState === "idle" && cs_plot1_table && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaTable className="text-primary" />
                        CpG Cluster Details
                    </h4>
                    <p className="text-sm text-gray-500 mb-6">
                        This table lists detailed attributes of the top 30 CpG clusters included in the correlation heatmap above.
                    </p>
                    <Plot1Table />
                </div>
            )}
        </div>
    );
};

export default Figure1;
