import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getCSPlot2Option, getCSPlot2 } from "../../../../redux/visualizationSlice.jsx";
import Plot2 from "../plot/Plot2.jsx";
import { FaChartBar, FaFilter, FaSyncAlt } from "react-icons/fa";
import {downloadPNG, downloadSVG} from "../../../../shared/utils/downloadPlot.jsx";

const Figure2 = ({ id }) => {
    const dispatch = useDispatch();
    const { cs_plot2_option, cs_plot2 } = useSelector(state => state.visualization.plots);

    const [loading, setLoading] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState("");
    const [selectedBatch, setSelectedBatch] = useState("All");

    // Fetch available clusters
    useEffect(() => {
        setLoading(true);
        dispatch(getCSPlot2Option({ job_id: id }))
            .unwrap()
            .catch(error => console.error("Error fetching CSPlot2Option:", error))
            .finally(() => setLoading(false));
    }, [dispatch, id]);

    // Auto-select first cluster
    useEffect(() => {
        if (cs_plot2_option?.cpg_groups && cs_plot2_option.cpg_groups.length > 0) {
            setSelectedCluster(cs_plot2_option.cpg_groups[0]);
        }
    }, [cs_plot2_option]);

    // Fetch plot data when a cluster is selected
    useEffect(() => {
        if (selectedCluster) {
            setLoading(true);
            dispatch(getCSPlot2({ job_id: id, cluster: selectedCluster, batch: selectedBatch }))
                .unwrap()
                .catch(error => console.error("Error fetching CSPlot2:", error))
                .finally(() => setLoading(false));
        }
    }, [dispatch, id, selectedCluster, selectedBatch]);

    return (
        <div className="bg-base-200 p-6 rounded-lg shadow-md border border-base-300 mt-5">
            {/* Section Title */}
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartBar className="text-primary" />
                CpG Cluster Visualization
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                Select a CpG Cluster to visualize its distribution across subtypes and batches.
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
                    {cs_plot2_option?.cpg_groups && cs_plot2_option.cpg_groups.length > 0 ? (
                        cs_plot2_option.cpg_groups.map((cluster) => (
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
                    {cs_plot2_option?.batches && cs_plot2_option.batches.length > 0 ? (
                        <>
                            <option key="All" value="All">All</option>
                            {cs_plot2_option.batches.map((batch) => (
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
                        <span className="ml-2">Loading cluster options...</span>
                    </div>
                )}
            </div>

            {/* Visualization Section */}
            {!loading && cs_plot2 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartBar className="text-primary" />
                        Cluster Distribution Plot
                    </h4>

                    {/* Only show when plot is loaded */}
                    <div className="flex justify-end gap-4 mt-4">
                        <button className="btn btn-sm btn-outline" onClick={() => downloadSVG("cs-plot2", "plot2.svg")}>
                            Download SVG
                        </button>
                        <button className="btn btn-sm btn-outline" onClick={() => downloadPNG("cs-plot2", "plot2.png")}>
                            Download PNG
                        </button>
                    </div>

                    <div className="flex justify-center mt-4">
                        <Plot2 />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Figure2;
