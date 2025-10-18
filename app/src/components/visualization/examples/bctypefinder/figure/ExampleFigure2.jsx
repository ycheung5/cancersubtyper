import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
    getBCExamplePlot2Option,
    getBCExamplePlot2,
} from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot2 from "../plot/ExamplePlot2.jsx";
import { FaChartBar, FaFilter, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../../shared/utils/downloadPlot.jsx";

const ExampleFigure2 = () => {
    const dispatch = useDispatch();
    const { bc_plot2_option, bc_plot2 } = useSelector(
        (s) => s.visualizationExample.plots
    );

    const [loading, setLoading] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState("");
    const [selectedBatch, setSelectedBatch] = useState("All");

    // Fetch available clusters/batches (example)
    useEffect(() => {
        setLoading(true);
        dispatch(getBCExamplePlot2Option())
            .unwrap()
            .catch((e) => console.error("Error fetching BCExamplePlot2Option:", e))
            .finally(() => setLoading(false));
    }, [dispatch]);

    // Auto-select first cluster
    useEffect(() => {
        if (bc_plot2_option?.cpg_groups?.length) {
            setSelectedCluster((prev) => prev || bc_plot2_option.cpg_groups[0]);
        }
    }, [bc_plot2_option?.cpg_groups]);

    // Fetch plot data when selection changes (example)
    useEffect(() => {
        if (!selectedCluster) return;
        setLoading(true);
        dispatch(getBCExamplePlot2({ option: selectedCluster, batch: selectedBatch }))
            .unwrap()
            .catch((e) => console.error("Error fetching BCExamplePlot2:", e))
            .finally(() => setLoading(false));
    }, [dispatch, selectedCluster, selectedBatch]);

    return (
        <div className="bg-base-200 p-6 rounded-lg shadow-md border border-base-300 mt-5">
            {/* Section Title */}
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartBar className="text-primary" />
                CpG Cluster Beta Value Distribution
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                This section allows you to explore the distribution of beta values for specific CpG clusters across batches.
                Use the selectors below to select a cluster and optionally narrow by batch.
            </p>

            {/* Filters */}
            <div className="flex flex-wrap gap-4 items-center bg-base-100 p-4 rounded-lg shadow-sm border border-base-300">
                {/* Cluster Selection */}
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Cluster:
                </label>
                <select
                    className="select select-bordered w-48"
                    value={selectedCluster}
                    onChange={(e) => setSelectedCluster(e.target.value)}
                    disabled={loading}
                >
                    {bc_plot2_option?.cpg_groups && bc_plot2_option.cpg_groups.length > 0 ? (
                        bc_plot2_option.cpg_groups.map((cluster) => (
                            <option key={cluster} value={cluster}>
                                CpG Cluster {cluster}
                            </option>
                        ))
                    ) : (
                        <option value="">No Clusters Available</option>
                    )}
                </select>

                {/* Batch Selection */}
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
                    {bc_plot2_option?.batches && bc_plot2_option.batches.length > 0 ? (
                        <>
                            <option key="All" value="All">All</option>
                            {bc_plot2_option.batches.map((batch) => (
                                <option key={batch} value={batch}>{batch}</option>
                            ))}
                        </>
                    ) : (
                        <option value="">No Batches Available</option>
                    )}
                </select>
            </div>

            {/* Loading State Message */}
            <div className="mt-4">
                {loading && (
                    <div className="flex justify-center items-center mt-4 text-primary">
                        <FaSyncAlt className="animate-spin text-xl" />
                        <span className="ml-2">Loading cluster data...</span>
                    </div>
                )}
            </div>

            {/* Visualization Section */}
            {!loading && bc_plot2 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartBar className="text-primary" />
                        Distribution Boxplot
                    </h4>
                    <p className="text-sm text-gray-500 mb-6">
                        This boxplot displays the distribution of beta values for the selected CpG cluster, grouped by batch.
                    </p>

                    <div className="flex justify-end gap-4 mt-4">
                        <button className="btn btn-sm btn-outline" onClick={() => downloadSVG("bc-example-plot2", "plot2.svg")}>
                            Download SVG
                        </button>
                        <button className="btn btn-sm btn-outline" onClick={() => downloadPNG("bc-example-plot2", "plot2.png")}>
                            Download PNG
                        </button>
                    </div>

                    <div className="flex justify-center mt-4">
                        <ExamplePlot2 />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ExampleFigure2;
